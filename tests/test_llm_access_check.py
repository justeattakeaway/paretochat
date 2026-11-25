from langchain_core.messages import AIMessage
import pytest

from paretochat.infrastructure.llm import provider_factory
from paretochat.infrastructure.llm.provider_factory import LLMAccessError, create_llm_model, ensure_llm_access


class _FakeModel:
    model = "fake-model"

    def __init__(self, reply="OK", error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    provider_factory._verified.clear()
    yield
    provider_factory._verified.clear()


def _use(monkeypatch, model):
    monkeypatch.setattr(provider_factory, "_build_llm_model", lambda: model)


def test_check_runs_only_once_across_calls(monkeypatch):
    model = _FakeModel()
    _use(monkeypatch, model)
    ensure_llm_access()
    create_llm_model()
    create_llm_model()
    assert model.calls == 1


def test_access_error_is_reported_with_provider_and_model(monkeypatch):
    _use(monkeypatch, _FakeModel(error=RuntimeError("401 not logged in")))
    with pytest.raises(LLMAccessError, match=r"'fake'.*'fake-model'.*401 not logged in"):
        create_llm_model()


def test_empty_response_is_an_access_error(monkeypatch):
    _use(monkeypatch, _FakeModel(reply=""))
    with pytest.raises(LLMAccessError, match="empty response"):
        create_llm_model()


def test_failure_is_not_cached(monkeypatch):
    model = _FakeModel(error=RuntimeError("boom"))
    _use(monkeypatch, model)
    with pytest.raises(LLMAccessError):
        create_llm_model()
    model.error = None
    create_llm_model()
    assert model.calls == 2


def test_claude_login_check_reports_logged_out(monkeypatch):
    from types import SimpleNamespace

    from paretochat.infrastructure.llm import claude_cli_provider

    monkeypatch.setattr(claude_cli_provider.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_cli_provider.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout='{"loggedIn": false}', stderr=""),
    )
    with pytest.raises(RuntimeError, match="not logged in"):
        claude_cli_provider.check_claude_login()


def test_claude_logged_out_is_reported_even_if_model_replies_with_text(monkeypatch):
    # The CLI answers "Not logged in" as ordinary text, so only the explicit login check can catch it.
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    model = _FakeModel(reply="Not logged in · Please run /login")
    _use(monkeypatch, model)

    def logged_out():
        raise RuntimeError("the Claude CLI is not logged in; run `claude auth login`")

    from paretochat.infrastructure.llm import claude_cli_provider

    monkeypatch.setattr(claude_cli_provider, "check_claude_login", logged_out)
    with pytest.raises(LLMAccessError, match="not logged in"):
        create_llm_model()
    assert model.calls == 0
