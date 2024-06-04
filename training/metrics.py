import torch


class MetricsClass:
    """Metrics class."""
    @staticmethod
    @torch.no_grad()
    def get_parameter_count(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
