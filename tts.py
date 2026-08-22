import threading

import pyttsx3


class TTSEngine:
    """Optional local text-to-speech engine."""

    def __init__(self):
        self.engine = None
        self.is_speaking = False
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
            self.engine.setProperty("volume", 0.9)
        except Exception:
            self.engine = None

    def speak(self, text, blocking=True):
        if not self.engine or not text or self.is_speaking:
            return False
        try:
            self.is_speaking = True
            clean_text = " ".join(str(text).replace("\n", " ").split())[:1000]
            if blocking:
                self.engine.say(clean_text)
                self.engine.runAndWait()
                self.is_speaking = False
            else:
                def run():
                    try:
                        self.engine.say(clean_text)
                        self.engine.runAndWait()
                    finally:
                        self.is_speaking = False
                threading.Thread(target=run, daemon=True).start()
            return True
        except Exception:
            self.is_speaking = False
            return False

    def set_rate(self, rate):
        if not self.engine:
            return False
        self.engine.setProperty("rate", max(50, min(300, int(rate))))
        return True

    def set_volume(self, volume):
        if not self.engine:
            return False
        self.engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
        return True

    def get_available_voices(self):
        if not self.engine:
            return []
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "languages": getattr(voice, "languages", []),
                "gender": getattr(voice, "gender", "unknown"),
            }
            for voice in self.engine.getProperty("voices")
        ]

    def set_voice(self, voice_id):
        if not self.engine:
            return False
        self.engine.setProperty("voice", voice_id)
        return True

    def stop(self):
        if not self.engine:
            return False
        try:
            self.engine.stop()
            self.is_speaking = False
            return True
        except Exception:
            return False

    def is_available(self):
        return self.engine is not None


# Backward-compatible name used by the original application entry point.
TextToSpeechEngine = TTSEngine

_tts_engine = TTSEngine()


def read_text(text, rate=175, volume=0.9, blocking=True):
    _tts_engine.set_rate(rate)
    _tts_engine.set_volume(volume)
    return _tts_engine.speak(text, blocking=blocking)


def stop_speech():
    return _tts_engine.stop()


def get_voices():
    return _tts_engine.get_available_voices()


def set_voice(voice_id):
    return _tts_engine.set_voice(voice_id)


def is_speaking():
    return _tts_engine.is_speaking
