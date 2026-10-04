"""Lightweight package: importing this module never loads a model."""

__version__ = "0.1.0"


def load_environment(**kwargs):
    """Load the optional verifiers adapter only when explicitly requested."""
    from .environment import load_environment as loader

    return loader(**kwargs)
