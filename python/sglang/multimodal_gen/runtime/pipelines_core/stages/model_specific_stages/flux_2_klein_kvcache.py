# SPDX-License-Identifier: Apache-2.0
"""
KV cache denoising stage for FLUX.2 Klein.

On step 0, reference image tokens are included in the forward pass and their
post-RoPE K/V projections are cached.  On steps 1+, the cached K/V is reused
so the reference tokens need not be recomputed, saving significant compute.

Adapts the standard DenoisingStage by overriding per-step methods rather than
duplicating the denoising loop.
"""

from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.flux_2 import Flux2KVCache
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Flux2KleinKVCacheDenoisingStage(DenoisingStage):
    """Denoising stage that caches reference image K/V on step 0.

    Reuses the parent ``DenoisingStage`` loop by overriding:
    - ``_prepare_denoising_loop``: sets up KV cache and computes freqs_cis
      without ref positions for steps 1+.
    - ``_predict_noise_with_cfg``: injects per-step KV cache params and
      switches from extract to cached mode after step 0.
    - ``_post_denoising_loop``: clears the KV cache.

    When there are no reference images, all overrides are no-ops and the
    parent behaviour is used unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_cache: Flux2KVCache | None = None
        self._num_ref_tokens: int = 0
        self._freqs_cis_without_ref: tuple[torch.Tensor, torch.Tensor] | None = None
        self._saved_image_latent: torch.Tensor | None = None

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        prepared = super()._prepare_denoising_loop(batch, server_args)

        if batch.image_latent is not None:
            num_double = len(self.transformer.transformer_blocks)
            num_single = len(self.transformer.single_transformer_blocks)
            self._kv_cache = Flux2KVCache(num_double, num_single)
            self._num_ref_tokens = batch.image_latent.shape[1]

            # Compute freqs_cis without ref by slicing off the last num_ref positions.
            # freqs_cis layout: [txt_positions, img_positions, ref_positions]
            cos, sin = prepared["pos_cond_kwargs"]["freqs_cis"]
            self._freqs_cis_without_ref = (
                cos[: -self._num_ref_tokens],
                sin[: -self._num_ref_tokens],
            )
        else:
            self._kv_cache = None

        return prepared

    def _predict_noise_with_cfg(
        self,
        current_model: nn.Module,
        latent_model_input: torch.Tensor,
        timestep,
        batch: Req,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ):
        if self._kv_cache is not None:
            # Build step-specific kwargs (don't mutate the original dict)
            pos_cond_kwargs = dict(pos_cond_kwargs)
            pos_cond_kwargs["kv_cache"] = self._kv_cache

            if timestep_index == 0:
                # Step 0: extract mode (ref tokens are in latent_model_input
                # via the parent's image_latent concat)
                pos_cond_kwargs["kv_cache_mode"] = "extract"
                pos_cond_kwargs["num_ref_tokens"] = self._num_ref_tokens
                pos_cond_kwargs["ref_fixed_timestep"] = 0.0
            else:
                # Steps 1+: cached mode, no ref tokens in input
                pos_cond_kwargs["kv_cache_mode"] = "cached"
                pos_cond_kwargs["num_ref_tokens"] = 0
                pos_cond_kwargs["freqs_cis"] = self._freqs_cis_without_ref

        result = super()._predict_noise_with_cfg(
            current_model=current_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=timestep_index,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            current_guidance_scale=current_guidance_scale,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            server_args=server_args,
            guidance=guidance,
            latents=latents,
        )

        # After step 0: clear image_latent so the parent loop won't concat
        # ref tokens on subsequent steps
        if self._kv_cache is not None and timestep_index == 0:
            self._saved_image_latent = batch.image_latent
            batch.image_latent = None

        return result

    def _post_denoising_loop(
        self,
        batch,
        latents,
        trajectory_latents,
        trajectory_timesteps,
        server_args,
        is_warmup=False,
    ):
        # Restore image_latent and clean up KV cache
        if self._saved_image_latent is not None:
            batch.image_latent = self._saved_image_latent
            self._saved_image_latent = None
        if self._kv_cache is not None:
            self._kv_cache.clear()
            self._kv_cache = None

        return super()._post_denoising_loop(
            batch,
            latents,
            trajectory_latents,
            trajectory_timesteps,
            server_args,
            is_warmup,
        )
