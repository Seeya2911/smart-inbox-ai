from llm.analyzer import EmailAnalyzer
from llm.language import detect_language
from llm.provider import MockLLMProvider


CASES = {
    "en": "I have an appointment tomorrow. Could you send the signed form today so I can submit it before the appointment?",
    "de": "Ich habe morgen einen Termin. Könnten Sie mir das unterschriebene Formular heute schicken, damit ich es rechtzeitig einreichen kann?",
    "fr": "J'ai un rendez-vous demain. Pourriez-vous m'envoyer le formulaire signé aujourd'hui afin que je puisse le déposer à temps ?",
    "es": "Tengo una cita mañana. ¿Podría enviarme hoy el formulario firmado para poder entregarlo a tiempo?",
}


def test_supported_languages_are_detected():
    for expected, text in CASES.items():
        result = detect_language(text)
        assert result.language == expected
        assert result.supported
        assert result.confidence > 0.5


def test_short_text_is_not_overclaimed():
    result = detect_language("OK")
    assert result.language == "unknown"
    assert not result.supported


def test_analyzer_preserves_original_text_and_detection_signal():
    analyzer = EmailAnalyzer(MockLLMProvider())
    result = analyzer.analyze({"subject": "Formulaire", "message_text": CASES["fr"]})
    assert result.language == "fr"
    assert result.detected_language == "fr"
    assert not result.language_disagreement
    assert result.summary


def test_schema_rejects_invalid_semantic_labels():
    from llm.schemas import EmailAnalysis

    try:
        EmailAnalysis(
            summary="example",
            intent="urgent_keyword",
            urgency="high",
            sentiment="neutral",
            priority="high",
        )
    except ValueError as exc:
        assert "Unsupported intent" in str(exc)
    else:
        raise AssertionError("Invalid semantic labels must be rejected")
