from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux_2_klein_kvcache import (
    Flux2KleinKVCacheDenoisingStage,
)


class Flux2KleinPipeline(Flux2Pipeline):
    pipeline_name = "Flux2KleinPipeline"


class Flux2KleinKVCachePipeline(Flux2Pipeline):
    """FLUX.2 Klein pipeline with KV cache for reference image conditioning.

    Overrides ``add_standard_denoising_stage`` to wire in
    ``Flux2KleinKVCacheDenoisingStage``. All other stages are inherited from
    ``Flux2Pipeline``.
    """

    pipeline_name = "Flux2KleinKVCachePipeline"

    def add_standard_denoising_stage(
        self,
        transformer_key: str = "transformer",
        transformer_2_key: str | None = "transformer_2",
        scheduler_key: str = "scheduler",
        vae_key: str | None = "vae",
    ):
        """Wire in the KV-cache denoising stage instead of the default one."""
        kwargs = {
            "transformer": self.get_module(transformer_key),
            "scheduler": self.get_module(scheduler_key),
        }

        if transformer_2_key:
            transformer_2 = self.get_module(transformer_2_key, None)
            if transformer_2 is not None:
                kwargs["transformer_2"] = transformer_2

        if vae_key:
            vae = self.get_module(vae_key, None)
            if vae is not None:
                kwargs["vae"] = vae
                kwargs["pipeline"] = self

        return self.add_stage(Flux2KleinKVCacheDenoisingStage(**kwargs))


EntryClass = [Flux2KleinPipeline, Flux2KleinKVCachePipeline]
