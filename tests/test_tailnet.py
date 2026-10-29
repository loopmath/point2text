import subprocess
from types import SimpleNamespace

import pytest

from point2text.server import tailnet_ipv4


def test_resolves_direct_tailnet_binding_without_serve_changes(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="100.101.102.103\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert tailnet_ipv4() == "100.101.102.103"
    assert calls == [["tailscale", "ip", "-4"]]


@pytest.mark.parametrize("value", ["", "0.0.0.0", "192.168.1.2", "not-an-ip", "100.1.2.3\n100.4.5.6"])
def test_refuses_invalid_or_non_tailnet_addresses(monkeypatch, value):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=value))
    with pytest.raises(RuntimeError, match="Tailscale IPv4"):
        tailnet_ipv4()


def test_missing_cli_has_actionable_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("tailscale")
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="connect to your tailnet"):
        tailnet_ipv4()
