import os
from pydantic import BaseModel, Field, ValidationError


class ServiceSettings(BaseModel):
    app_name: str = Field(default="track1-inference-service", min_length=1)
    model_id: str = Field(default="fake-llm-v1", min_length=1)
    max_prompt_chars: int = Field(default=4096, ge=1, le=32768)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    startup_delay_sec: float = Field(default=0.0, ge=0.0, le=30.0)
    force_unready: bool = Field(default=False)

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        raw = {
            "app_name": os.getenv("INFERENCE_APP_NAME", "track1-inference-service"),
            "model_id": os.getenv("INFERENCE_MODEL_ID", "fake-llm-v1"),
            "max_prompt_chars": int(os.getenv("INFERENCE_MAX_PROMPT_CHARS", "4096")),
            "max_new_tokens": int(os.getenv("INFERENCE_MAX_NEW_TOKENS", "256")),
            "startup_delay_sec": float(os.getenv("INFERENCE_STARTUP_DELAY_SEC", "0.0")),
            "force_unready": os.getenv("INFERENCE_FORCE_UNREADY", "false").lower() in ("1", "true", "yes"),
        }
        return cls(**raw)


def validate_settings_or_raise() -> ServiceSettings:
    try:
        return ServiceSettings.from_env()
    except (ValueError, ValidationError) as exc:
        raise RuntimeError(f"Invalid service configuration: {exc}") from exc
