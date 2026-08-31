"""Local Generative AI foundation for Smart Inbox AI."""
from ml.generation.inference import extract_action, process_email, summarize_email
from ml.generation.model import LocalGenerationModel, load_model
from ml.generation.schemas import (
    ALLOWED_ACTION_TYPES,
    GenerationOutput,
    GenerationTrainingExample,
    SuggestedAction,
    UserFeedbackExample,
)

__all__ = [
    "load_model",
    "LocalGenerationModel",
    "summarize_email",
    "extract_action",
    "process_email",
    "SuggestedAction",
    "GenerationOutput",
    "GenerationTrainingExample",
    "UserFeedbackExample",
    "ALLOWED_ACTION_TYPES",
]
