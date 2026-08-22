from llm.provider import DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, resolve_openai_config


def test_openai_api_key_is_preferred(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")
    key, base_url, model = resolve_openai_config()
    assert key == "openai-key"
    assert base_url == DEFAULT_OPENAI_BASE_URL
    assert model == DEFAULT_OPENAI_MODEL


def test_openai_configuration_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    key, base_url, model = resolve_openai_config()
    assert key == "openai-key"
    assert base_url == "https://example.invalid/v1"
    assert model == "test-model"


def test_missing_key_has_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    try:
        resolve_openai_config()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing API key to raise RuntimeError")
