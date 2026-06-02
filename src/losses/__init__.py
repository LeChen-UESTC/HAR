"""Loss functions."""

from .classwise_infonce import classwise_infonce, classwise_multibank_infonce

__all__ = [
    "classwise_infonce",
    "classwise_multibank_infonce",
]
