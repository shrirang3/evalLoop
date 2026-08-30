"""EvalLoop — a configurable evaluation and improvement control plane for AI products."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evalloop")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
