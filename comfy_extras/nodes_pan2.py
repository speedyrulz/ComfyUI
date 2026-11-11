import json

import torch
from typing_extensions import override

import comfy.ldm.pan2.forecast
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import ComfyExtension, io

# The negative prompt PAN 2 was released with (from diffusers' PAN 2 pipeline, Apache-2.0). Every `time_range_sec` is
# filled in per clip so that it spans the whole video.
DEFAULT_NEGATIVE_PROMPT = {'video_style': 'low quality, blur, softness, reduced detail, noise, compression, clipped highlights, blurred sidebar, black border, letterboxing, '
                'watermark, subtitle, AIGC text, oily waxy plastic texture, oversaturation or color cast, color clipping or chroma bleeding, '
                'flicker, jitter, ghosting, motion smear, choppy or laggy motion, slow motion, Ken Burns effect, unmotivated slow zoom over a frozen '
                'scene, pan across a still image, global frame scaling or drifting, simulated movement from a static photograph, repeated '
                'micro-motion loop, discontinuous shot, scene cut, hard cut, crossfade, fade-in, fade-out, fading transition, scene replacement, '
                'spontaneous object or person pop-in or disappearance, duplicated or missing people or props, identity or count drift, rigid-object '
                'deformation or morphing, bending or melting rigid structures, structural shape or component drift',
 'lighting': 'overexposure, clipped highlights, underexposure, harsh lighting, overly deep skin shadows, brightness fading, excessive lighting '
             'saturation, chroma-clipped highlights, neon-like skin tones, colored spill overwhelming local material colors, oversaturated bloom, '
             'unnatural neon color casts, hue drift, chroma noise, bad white balance, color pulsing, flickering light, unnatural lighting',
 'perspective_view': 'warped geometry, distorted perspective, inconsistent spatial relationships, unmotivated layout changes, camera teleport or '
                     'jump, viewpoint reset, wrong depth or occlusion',
 'subjects_lifecycle': [{'subject_id': '<|Subject1|>',
                         'display_name': 'malformed inconsistent subject',
                         'subject_attribute': 'broken anatomy, distorted body proportions, reversed pelvis, twisted torso, deformed arms or legs, '
                                              'reversed joints, fused feet or toes, warped or collapsing faces, transparent or floating head and '
                                              'body, missing body parts, detached or disembodied body parts, floating hands or limbs, detached '
                                              'articulated parts, disconnected joints or supports, hands disconnected from wrists or forearms, '
                                              'discontinuous limb-to-torso connections, blurred or fused fingers, missing thumbs, extra or missing '
                                              'fingers, malformed wrists, extra limbs, duplicated hands, mirrored face, identity drift, wrong '
                                              'relative sizes, human-object fusion, unintended body-material fusion, loss of separate material '
                                              'boundaries, opaque white nail polish, gray nail polish, chalky opaque fingernails, artificial acrylic '
                                              'nails'}],
 'init_frame_visual_description': 'blur or softness, reduced detail, noise or compression artifacts, clipped highlights, blurred sidebar, black '
                                  'border or letterboxing, oily waxy plastic surfaces, oversaturation or color cast, exaggerated local contrast, '
                                  'exposure fading, bad white balance, smeared dense textures, blurred or distorted face, transparent or missing '
                                  'body parts, detached or disembodied body parts, floating hands or limbs, detached articulated parts, disconnected '
                                  'joints or supports, disconnected wrists or forearms, discontinuous limb-to-torso connections, twisted torso, '
                                  'fused or missing fingers, missing thumbs, malformed wrists, duplicated limbs, deformed arms or legs, reversed '
                                  'knees, fused toes, warped geometry, deformed or asymmetrically warped rigid objects, bent rigid structures, '
                                  'incorrect rigid part geometry, topology, proportions, or component count, wrong occlusion, floating or hovering '
                                  'objects, unsupported objects, object or human-object fusion, unintended fusion of distinct surfaces or layers, '
                                  'loss of separate material boundaries, merged objects, hand-object penetration, object-object penetration, '
                                  'penetration, collision, impossible contact, wrong sizes.',
 'events': [{'event_index': 0,
             'time_range_sec': '0 to 10',
             'event_description': 'blur, softness, reduced detail, compression artifacts, clipped highlights, flicker, jitter, ghosting, motion '
                                  'smear, choppy or laggy motion, slow motion, Ken Burns effect, unmotivated slow zoom over a frozen scene, pan '
                                  'across a still image, global frame scaling or drifting, simulated movement from a static photograph, repeated '
                                  'micro-motion loop, oily oversaturated or color-pulsing surfaces, shimmering surface detail, fingers fusing or '
                                  'changing count, hands smearing, facial features merging, reversed joints, upper body rotating independently of '
                                  'attached hips and feet, corkscrewed waist, limbs changing length or attachment after occlusion, transparent body '
                                  'or head, detached or disembodied body parts, floating hands or limbs, detached articulated parts, disconnected '
                                  'joints or supports, hands disconnected from wrists or forearms, body parts moving independently, vanishing '
                                  'limb-to-torso connections, merged people or limbs, limbs fused into torso, merged or fused feet and toes, '
                                  'floating or hovering objects, unsupported objects, object or human-object fusion, unintended fusion of distinct '
                                  'surfaces or layers, loss of separate material boundaries, merged objects, hand-object penetration, object-object '
                                  'penetration, penetration or clipping, objects colliding or morphing, rigid objects tearing, fragmenting, fusing, '
                                  'bending, melting or changing dimensions, topology, silhouette, component count or layout, subjects changing '
                                  'identity, splitting or duplicating, objects appearing or disappearing without entry, exit or occlusion, changed '
                                  'identity or count after occlusion, state changes inconsistent with the causing action, broken physics, uncaused '
                                  'deformation or motion, state changes preceding the causing interaction, loss of coupling during contact, pose '
                                  'jumps, discontinuous shot, camera jump, hard scene cut, crossfade, fade-in, fade-out, fading transition, scene '
                                  'replacement or viewpoint reset.',
             'camera_motion': [{'time_range_sec': '0 to 10', 'camera_motion': ['camera_jump', 'hard_cut', 'viewpoint_reset']}],
             'contained_actions': [{'time_range_sec': '0 to 10',
                                    'action': '<|Subject1|> fingers fusing during grip changes, palms merging with props, wrists collapsing, crossed '
                                              'limbs fusing, intersecting or swapping attachment, duplicated hands or feet, detached or disembodied '
                                              'hands or limbs, hands disconnected from wrists or forearms, discontinuous limb-to-torso connections, '
                                              'merged or fused feet and toes, unstable contact, unintended grip slip, hand-object penetration, '
                                              'body-object penetration, passage through solid surfaces instead of openings, object-object '
                                              'penetration, merged or fused objects, rigid objects fragmenting, fusing, smearing or changing '
                                              'dimensions during motion, unintended fusion of distinct surfaces or layers, loss of separate material '
                                              'boundaries, floating or hovering objects, unsupported objects, objects duplicating, splitting or '
                                              'disappearing during interaction or occlusion, unchanged action result, pose jump'}]}],
 'quality': {'video_quality': 'low resolution, blur, softness, reduced detail, noise, compression, clipped highlights, blurred sidebar, black '
                              'border, letterboxing, oily waxy plastic surfaces, greasy highlights, oversaturated colors, faded or washed-out '
                              'colors, exposure fading, smeared faces, hands or held objects, surface detail smearing or swimming, material or '
                              'existing markings changing without a physical cause, color or shape morphing, object pop-in or pop-out, sudden '
                              'disappearance, missing persistent objects, object duplication or count drift, rigid-object warping or deformation, '
                              'changing rigid proportions, topology, silhouette, or component layout',
             'text_logos_overlays': 'watermark, subtitle, captions, AIGC text or logo',
             'artifacts': ['watermark',
                           'template_or_text_dominated',
                           'talking_head',
                           'visual_quality_issue',
                           'edited_or_discontinuous',
                           'video_game',
                           'animation',
                           'ai_generated_content',
                           'black_border',
                           'ken_burn',
                           'time_lapse',
                           'blurred_sidebar',
                           'hard_scene_cuts']}}


def default_negative_prompt(length, fps):
    span = f"0 to {(length - 1) / fps:g}"

    def fill(node):
        if isinstance(node, dict):
            return {key: span if key == "time_range_sec" else fill(value) for key, value in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        return node

    return json.dumps(fill(DEFAULT_NEGATIVE_PROMPT), ensure_ascii=False, separators=(",", ":"))


class TextEncodePAN2(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodePAN2",
            display_name="PAN 2 Text Encode",
            category="model/conditioning/pan2",
            description="Encodes a PAN 2 prompt exactly as written. PAN 2 prompts are structured JSON captions; the regular text encode box would read their braces and | characters as dynamic prompt syntax and change them.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, tooltip="The structured world-state caption (JSON), or plain text."),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, prompt) -> io.NodeOutput:
        return io.NodeOutput(clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))


class PAN2NegativePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PAN2NegativePrompt",
            display_name="PAN 2 Default Negative Prompt",
            category="model/conditioning/pan2",
            description="The structured negative prompt PAN 2 was released with, its time ranges set to the length of the clip. Connect it to the prompt of a PAN 2 Text Encode node.",
            inputs=[
                io.Int.Input("length", default=241, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Number of video frames."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.01),
            ],
            outputs=[io.String.Output(display_name="negative_prompt")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, length, fps) -> io.NodeOutput:
        return io.NodeOutput(default_negative_prompt(length, fps))


class PAN2ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PAN2ImageToVideo",
            display_name="PAN 2 Text/Image To Video",
            category="model/conditioning/pan2",
            description="Empty PAN 2 video latent. With a start image it also sets the first-frame conditioning and makes the first latent frame follow the positive prediction without guidance, as in the reference sampler.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1248, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=704, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=241, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Number of video frames at 24 fps (241 is about 10 seconds, the model goes up to 481)."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True, tooltip="First frame for image-to-video. Leave empty for text-to-video."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, positive, negative, vae, width, height, length, batch_size, start_image=None) -> io.NodeOutput:
        frames = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 48, frames, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            # scaled to cover the canvas and center cropped, as PAN 2 was trained
            image = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            first = vae.encode(image[:, :, :, :3])
            concat_latent_image = torch.zeros([1, first.shape[1], frames, first.shape[-2], first.shape[-1]], device=first.device, dtype=first.dtype)
            concat_latent_image[:, :, :1] = first[:, :, :1]
            mask = torch.ones((1, 1, frames, first.shape[-2], first.shape[-1]), device=first.device, dtype=first.dtype)
            mask[:, :, :1] = 0.0  # the first latent frame is given

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

            def first_frame_unguided(args):
                denoised = args["denoised"]
                cond = args["cond_denoised"]
                if denoised is not cond and denoised.ndim == 5:
                    denoised[:, :, :1] = cond[:, :, :1]
                return denoised

            model = model.clone()
            model.set_model_sampler_post_cfg_function(first_frame_unguided)

        return io.NodeOutput(model, positive, negative, {"samples": latent})


class PAN2Spectrum(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PAN2Spectrum",
            display_name="PAN 2 Spectrum (Step Forecast)",
            category="model/conditioning/pan2",
            description="Speeds up PAN 2 sampling with Spectrum: on some steps the transformer is skipped and its result is forecast from earlier steps. The video differs somewhat from a normal run. Made for samplers that call the model once per step (euler and similar).",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("warmup_steps", default=5, min=2, max=1000, tooltip="Steps at the start that always run the model."),
                io.Float.Input("window_size", default=2.0, min=1.0, max=64.0, step=0.05, tooltip="After the warm-up the model runs once every this many steps (rounded down). 1 with flex_window 0 turns forecasting off."),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=16.0, step=0.05, tooltip="The window grows by this much after every step that runs the model. 0 keeps a fixed spacing; higher is faster and less exact."),
                io.Float.Input("blend", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="Share of the Chebyshev forecast; the rest is a straight step from the last two results."),
                io.Int.Input("degree", default=4, min=1, max=8, tooltip="Degree of the Chebyshev polynomials.", advanced=True),
                io.Float.Input("ridge", default=0.1, min=0.0, max=100.0, step=0.01, tooltip="Regularization of the fit.", advanced=True),
                io.Int.Input("final_steps", default=1, min=0, max=1000, tooltip="Steps at the end that always run the model.", advanced=True),
                io.Combo.Input("history", options=["system RAM", "GPU"], tooltip="Where the results of earlier steps are kept (up to 10 per prompt). GPU avoids copies but needs the VRAM.", advanced=True),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, warmup_steps, window_size, flex_window, blend, degree=4, ridge=0.1, final_steps=1, history="system RAM") -> io.NodeOutput:
        model = model.clone()
        options = model.model_options["transformer_options"] = model.model_options.get("transformer_options", {}).copy()
        options["pan2_step_forecaster"] = comfy.ldm.pan2.forecast.StepForecaster(
            warmup_steps=warmup_steps, window_size=window_size, flex_window=flex_window, degree=degree, ridge=ridge, blend=blend,
            final_steps=final_steps, history_on_gpu=history == "GPU")
        return io.NodeOutput(model)


class PAN2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodePAN2,
            PAN2NegativePrompt,
            PAN2ImageToVideo,
            PAN2Spectrum,
        ]


async def comfy_entrypoint() -> PAN2Extension:
    return PAN2Extension()
