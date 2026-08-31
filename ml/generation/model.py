"""Local pretrained generative model loader and inference wrapper (FLAN-T5-base)."""
from __future__ import annotations

import time
from typing import Optional

DEFAULT_MODEL_NAME = "google/flan-t5-base"

_CACHED_MODEL: Optional["LocalGenerationModel"] = None


class LocalGenerationModel:
    """Wrapper around local HuggingFace Seq2Seq model (FLAN-T5)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required for local generation. "
                "Please install them via: pip install torch transformers"
            ) from e

        self._torch = torch
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"  [INIT] Loading local model '{model_name}' on {self.device}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"  [OK] Model loaded in {time.time() - t0:.2f}s.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        min_length: int = 5,
        num_beams: int = 2,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Generate text completion from a prompt using greedy/beam search."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> LocalGenerationModel:
    """Load or retrieve the cached singleton LocalGenerationModel."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None or force_reload or _CACHED_MODEL.model_name != model_name:
        _CACHED_MODEL = LocalGenerationModel(model_name=model_name, device=device)
    return _CACHED_MODEL
