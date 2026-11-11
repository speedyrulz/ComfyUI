# Spectrum step forecasting for PAN 2.
#
# The method is Spectrum (Han, Shi, Li, Ye, Guo, Ermon, "Adaptive Spectral Feature Forecasting for Diffusion Sampling
# Acceleration", https://arxiv.org/abs/2603.01623, official code https://github.com/hanjq17/Spectrum, MIT): the hidden
# state after the last transformer block is a function of the sampling step. On actual steps the model runs and the
# state is kept. On forecast steps the blocks are skipped and the state is predicted from the kept ones, with a ridge
# fit of Chebyshev polynomials over the step index blended with a first-order step from the last two states; only
# the output layer runs.
#
# Both predictors are linear in the kept states, so a forecast is a weighted sum of them: the weights are a tiny
# solve on the CPU and the states are never copied into a coefficient matrix.

import copy
import logging
import math

import torch

TOKEN_CHUNK = 8192


def actual_step_plan(total_steps, warmup_steps, window_size, flex_window, final_steps):
    """True where the model runs. After the warm-up an actual step follows floor(window) - 1 forecast steps, and the
    window grows by flex_window after each actual step, so actual steps get sparser as sampling settles."""
    plan, forecasted, window = [], 0, float(window_size)
    for i in range(total_steps):
        actual = True
        if i >= warmup_steps:
            actual = (forecasted + 1) % max(1, math.floor(window)) == 0
            if actual:
                window += flex_window
        if i >= total_steps - final_steps:
            actual = True
        forecasted = 0 if actual else forecasted + 1
        plan.append(actual)
    return plan


def forecast_weights(steps, target, total_steps, degree, ridge, blend):
    """Weights over the kept states (oldest first) that give the forecast at step `target`."""
    k = len(steps)
    t = torch.tensor(steps, dtype=torch.float64)
    # last state plus a first-order step in units of the last spacing
    local = torch.zeros(k, dtype=torch.float64)
    local[-1] = 1.0
    if k >= 2:
        frac = (target - steps[-1]) / max(steps[-1] - steps[-2], 1e-8)
        local[-1] += frac
        local[-2] -= frac
    if blend <= 0.0:
        return local

    def design(x):
        tau = (x * 2.0 / max(total_steps, 1) - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, degree + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[:degree + 1], dim=1)

    x = design(t)
    gram = x.T @ x + ridge * torch.eye(x.shape[1], dtype=torch.float64)
    spectral = (design(torch.tensor([float(target)], dtype=torch.float64)) @ torch.linalg.solve(gram, x.T)).reshape(-1)
    return (1.0 - blend) * local + blend * spectral


class _Stream:
    def __init__(self):
        self.steps = []
        self.states = []
        self.last_step = -1
        self.total_steps = None
        self.actual = 0
        self.forecast = 0


class StepForecaster:
    def __init__(self, warmup_steps=5, window_size=2.0, flex_window=0.75, degree=4, ridge=0.1, blend=0.5, final_steps=1,
                 max_history=10, history_on_gpu=False):
        self.settings = dict(warmup_steps=int(warmup_steps), window_size=float(window_size), flex_window=float(flex_window),
                             degree=int(degree), ridge=float(ridge), blend=float(blend), final_steps=int(final_steps),
                             max_history=max(int(max_history), int(degree) + 2), history_on_gpu=bool(history_on_gpu))
        self.streams = {}
        self.plans = {}

    def __deepcopy__(self, memo):
        # the settings travel with a copied model, the kept states do not
        return StepForecaster(**copy.deepcopy(self.settings, memo))

    def _plan(self, total_steps):
        if total_steps not in self.plans:
            s = self.settings
            self.plans = {total_steps: actual_step_plan(total_steps, s["warmup_steps"], s["window_size"], s["flex_window"], s["final_steps"])}
        return self.plans[total_steps]

    @staticmethod
    def _locate(transformer_options):
        """(stream key, step index, steps of this run), or None when the call is not on the sampler's schedule."""
        schedule = transformer_options.get("sample_sigmas", None)
        current = transformer_options.get("sigmas", None)
        if schedule is None or current is None or schedule.numel() < 2:
            return None
        sigma = current.reshape(-1)[0].float()
        distance = (schedule.float().to(sigma.device) - sigma).abs()
        index = int(distance.argmin())
        total_steps = schedule.numel() - 1
        if index >= total_steps or float(distance[index]) > 1e-5 + 1e-4 * abs(float(sigma)):
            return None  # an in-between evaluation of a multi-stage sampler: always run the model
        key = (tuple(transformer_options.get("cond_or_uncond", ())), tuple(transformer_options.get("uuids", ())))
        return key, index, total_steps

    def _stream(self, key, index, total_steps, shape):
        stream = self.streams.get(key)
        fresh = stream is None or index < stream.last_step or stream.total_steps != total_steps \
            or (len(stream.states) > 0 and tuple(stream.states[-1].shape) != tuple(shape))
        if fresh:
            if index == 0:  # a new run: drop what earlier runs left behind
                self.streams = {k: v for k, v in self.streams.items() if v.last_step == 0}
            stream = self.streams[key] = _Stream()
            stream.total_steps = total_steps
        return stream

    def forecast(self, transformer_options, shape, device, dtype):
        """The forecast hidden state for this call, or None when the model has to run."""
        where = self._locate(transformer_options)
        if where is None:
            return None
        key, index, total_steps = where
        stream = self._stream(key, index, total_steps, shape)
        if index == stream.last_step or self._plan(total_steps)[index] or len(stream.states) < 2:
            return None
        s = self.settings
        weights = forecast_weights(stream.steps, index, total_steps, s["degree"], s["ridge"], s["blend"])
        tokens = shape[1]
        out = torch.empty(shape, device=device, dtype=dtype)
        for a in range(0, tokens, TOKEN_CHUNK):
            b = min(a + TOKEN_CHUNK, tokens)
            acc = torch.zeros((shape[0], b - a, shape[2]), device=device, dtype=torch.float32)
            for weight, state in zip(weights.tolist(), stream.states):
                if weight != 0.0:
                    acc.add_(state[:, a:b].to(device=device, dtype=torch.float32), alpha=weight)
            out[:, a:b] = acc.to(dtype)
        stream.last_step = index
        stream.forecast += 1
        self._report(stream, index, total_steps)
        return out

    def observe(self, transformer_options, x):
        """Keep the hidden state of an actual step."""
        where = self._locate(transformer_options)
        if where is None:
            return
        key, index, total_steps = where
        stream = self._stream(key, index, total_steps, x.shape)
        if index == stream.last_step:
            return
        state = x.detach()
        state = state.clone() if self.settings["history_on_gpu"] else state.to("cpu")
        stream.steps.append(index)
        stream.states.append(state)
        if len(stream.states) > self.settings["max_history"]:
            stream.steps.pop(0)
            stream.states.pop(0)
        stream.last_step = index
        stream.actual += 1
        self._report(stream, index, total_steps)

    @staticmethod
    def _report(stream, index, total_steps):
        if index == total_steps - 1:
            logging.info("PAN 2 Spectrum: %d steps, %d ran the model, %d forecast", total_steps, stream.actual, stream.forecast)
            stream.steps, stream.states = [], []  # the run is over, free the kept states
