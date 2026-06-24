"""BE-FR YOLO: training-time blur evolution and feature-level frequency refinement."""

__all__ = [
    "BlurEvolution",
    "BlurEvolutionConfig",
    "FRBlock",
    "FRNet",
    "SpectralConsistencyLoss",
]


def __getattr__(name: str):
    if name in {"BlurEvolution", "BlurEvolutionConfig"}:
        from .blur.blur_evolution import BlurEvolution, BlurEvolutionConfig

        return {"BlurEvolution": BlurEvolution, "BlurEvolutionConfig": BlurEvolutionConfig}[name]
    if name in {"FRBlock", "FRNet"}:
        from .modules.frnet import FRBlock, FRNet

        return {"FRBlock": FRBlock, "FRNet": FRNet}[name]
    if name == "SpectralConsistencyLoss":
        from .losses.spectral_consistency import SpectralConsistencyLoss

        return SpectralConsistencyLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
