import logging
import os

import torch

logger = logging.getLogger(__name__)

#: Name of the environment variable used to request a specific torch device.
TORCH_DEVICE_ENV = "TORCH_DEVICE"

#: Auto-selection preference order when no device is explicitly requested.
_AUTO_PREFERENCE: tuple[str, ...] = ("mps", "cuda", "cpu")


def _is_available(backend: str) -> bool:
    """Return whether a torch device backend is usable on this machine.

    Args:
        backend: The device backend name (`"cpu"`, `"cuda"` or `"mps"`).

    Returns:
        `True` if the backend is available, `False` otherwise.
    """
    if backend == "cpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "mps":
        return torch.backends.mps.is_available()
    return False


def get_torch_device(preferred: str | None = None) -> torch.device:
    """Resolve the torch device to use, with graceful fallback to CPU.

    Resolution order:

    1. The `preferred` argument, if given.
    2. Otherwise the `TORCH_DEVICE` environment variable.
    3. Otherwise auto-select following `mps -> cuda -> cpu`.

    A requested device whose backend is unavailable falls back to CPU with a
    warning, so callers always receive a usable device.

    Args:
        preferred: Optional explicit device string (e.g. `"mps"`, `"cuda"`,
            `"cuda:0"` or `"cpu"`). Overrides the environment variable.

    Returns:
        The resolved :class:`torch.device`.
    """
    requested = (preferred or os.environ.get(TORCH_DEVICE_ENV) or "").strip().lower()
    if requested:
        backend = requested.split(":", 1)[0]
        if _is_available(backend):
            logger.info("Using requested torch device: %s", requested)
            return torch.device(requested)
        logger.warning("Requested torch device %r is unavailable; falling back to cpu.", requested)
        return torch.device("cpu")

    for backend in _AUTO_PREFERENCE:
        if _is_available(backend):
            logger.info("Auto-selected torch device: %s", backend)
            return torch.device(backend)
    return torch.device("cpu")
