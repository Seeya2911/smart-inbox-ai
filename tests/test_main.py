import main


def test_openai_api_key_selects_compatible_provider(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeProvider:
        def __init__(self):
            self.model = "fake"

        def analyze(self, message):
            raise AssertionError("provider should not be called during construction")

    monkeypatch.setattr(main, "OpenAICompatibleProvider", FakeProvider)

    assistant = main.SmartInboxAssistant()

    assert isinstance(assistant.analyzer.provider, FakeProvider)


def test_legacy_llm_api_key_still_selects_compatible_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    class FakeProvider:
        def __init__(self):
            self.model = "fake"

        def analyze(self, message):
            raise AssertionError("provider should not be called during construction")

    monkeypatch.setattr(main, "OpenAICompatibleProvider", FakeProvider)

    assistant = main.SmartInboxAssistant()

    assert isinstance(assistant.analyzer.provider, FakeProvider)


def test_without_api_keys_uses_mock_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    assistant = main.SmartInboxAssistant()

    assert isinstance(assistant.analyzer.provider, main.MockLLMProvider)
