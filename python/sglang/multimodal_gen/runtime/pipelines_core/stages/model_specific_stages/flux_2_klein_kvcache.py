# SPDX-License-Identifier: Apache-2.0
"""
KV cache denoising stage for FLUX.2 Klein.

On step 0, reference image tokens are included in the forward pass and their
post-RoPE K/V projections are cached.  On steps 1+, the cached K/V is reused
so the reference tokens need not be recomputed, saving significant compute.
"""

import time

import torch

from sglang.multimodal_gen.runtime.models.dits.flux_2 import Flux2KVCache
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class Flux2KleinKVCacheDenoisingStage(DenoisingStage):
    """Denoising stage that caches reference image K/V on step 0.

    Step 0: concatenate ref image latent, run full forward with
    ``kv_cache_mode="extract"`` so each attention layer stores ref K/V.

    Steps 1+: omit ref image latent, run forward with
    ``kv_cache_mode="cached"`` so attention layers append cached ref K/V
    to the current K/V.

    When there are no reference images the stage delegates entirely to the
    parent ``DenoisingStage.forward``.
    """

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # No reference images -> standard behaviour
        if batch.image_latent is None:
            return super().forward(batch, server_args)

        # ---- prepare (reuse parent infrastructure) ----
        prepared_vars = self._prepare_denoising_loop(batch, server_args)

        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        image_kwargs = prepared_vars["image_kwargs"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        seq_len = prepared_vars["seq_len"]
        guidance = prepared_vars["guidance"]

        # ---- KV cache setup ----
        num_double_layers = len(self.transformer.transformer_blocks)
        num_single_layers = len(self.transformer.single_transformer_blocks)
        kv_cache = Flux2KVCache(num_double_layers, num_single_layers)

        # Number of ref tokens (already SP-sharded if applicable)
        num_ref_tokens = batch.image_latent.shape[1]

        # Prepare pos_cond_kwargs *without* ref positions for steps 1+.
        # freqs_cis layout is [txt, img, ref]; slice off the last num_ref_tokens.
        cos_full, sin_full = pos_cond_kwargs["freqs_cis"]
        freqs_cis_without_ref = (
            cos_full[:-num_ref_tokens],
            sin_full[:-num_ref_tokens],
        )
        pos_cond_kwargs_without_ref = dict(pos_cond_kwargs)
        pos_cond_kwargs_without_ref["freqs_cis"] = freqs_cis_without_ref

        # ---- denoising loop ----
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        denoising_start_time = time.time()

        is_warmup = batch.is_warmup
        self.scheduler.set_begin_index(0)
        timesteps_cpu = timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{i}",
                        logger=logger,
                        metrics=batch.metrics,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                    ):
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )

                        latent_model_input = latents.to(target_dtype)

                        if i == 0:
                            # Step 0: include ref image tokens, extract KV
                            latent_model_input = torch.cat(
                                [latent_model_input, batch.image_latent], dim=1
                            ).to(target_dtype)
                            step_pos_cond_kwargs = dict(pos_cond_kwargs)
                            step_pos_cond_kwargs["kv_cache"] = kv_cache
                            step_pos_cond_kwargs["kv_cache_mode"] = "extract"
                            step_pos_cond_kwargs["num_ref_tokens"] = num_ref_tokens
                            step_pos_cond_kwargs["ref_fixed_timestep"] = 0.0
                        else:
                            # Steps 1+: reuse cached ref KV, no ref tokens
                            step_pos_cond_kwargs = dict(pos_cond_kwargs_without_ref)
                            step_pos_cond_kwargs["kv_cache"] = kv_cache
                            step_pos_cond_kwargs["kv_cache_mode"] = "cached"
                            step_pos_cond_kwargs["num_ref_tokens"] = 0

                        timestep = self.expand_timestep_before_forward(
                            batch,
                            server_args,
                            t_device,
                            target_dtype,
                            seq_len,
                            reserved_frames_mask,
                        )

                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t_device
                        )

                        attn_metadata = self._build_attn_metadata(
                            i,
                            batch,
                            server_args,
                            timestep_value=t_int,
                            timesteps=timesteps_cpu,
                        )

                        noise_pred = self._predict_noise_with_cfg(
                            current_model=current_model,
                            latent_model_input=latent_model_input,
                            timestep=timestep,
                            batch=batch,
                            timestep_index=i,
                            attn_metadata=attn_metadata,
                            target_dtype=target_dtype,
                            current_guidance_scale=current_guidance_scale,
                            image_kwargs=image_kwargs,
                            pos_cond_kwargs=step_pos_cond_kwargs,
                            neg_cond_kwargs=neg_cond_kwargs,
                            server_args=server_args,
                            guidance=guidance,
                            latents=latents,
                        )

                        if server_args.comfyui_mode:
                            batch.noise_pred = noise_pred

                        latents = self.scheduler.step(
                            model_output=noise_pred,
                            timestep=t_device,
                            sample=latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        latents = self.post_forward_for_ti2v_task(
                            batch, server_args, reserved_frames_mask, latents, z
                        )

                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)

                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        if not is_warmup:
                            self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0 and not is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )

        # Clean up cache
        kv_cache.clear()

        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )
        return batch
