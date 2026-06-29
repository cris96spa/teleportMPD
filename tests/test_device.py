import torch

from teleport_mdp.utils.device import TORCH_DEVICE_ENV, get_torch_device


def test_explicit_cpu():
    """An explicit cpu request is honored."""
    assert get_torch_device("cpu") == torch.device("cpu")


def test_unavailable_device_falls_back_to_cpu():
    """A request for an unknown/unavailable backend falls back to cpu."""
    assert get_torch_device("not_a_real_device") == torch.device("cpu")


def test_env_var_is_used(monkeypatch):
    """The TORCH_DEVICE environment variable selects the device when set."""
    monkeypatch.setenv(TORCH_DEVICE_ENV, "cpu")
    assert get_torch_device() == torch.device("cpu")


def test_explicit_arg_overrides_env(monkeypatch):
    """The explicit argument takes precedence over the environment variable."""
    monkeypatch.setenv(TORCH_DEVICE_ENV, "not_a_real_device")
    assert get_torch_device("cpu") == torch.device("cpu")


def test_auto_returns_available_device(monkeypatch):
    """Auto-selection returns a usable, known backend."""
    monkeypatch.delenv(TORCH_DEVICE_ENV, raising=False)
    device = get_torch_device()
    assert device.type in {"cpu", "cuda", "mps"}
