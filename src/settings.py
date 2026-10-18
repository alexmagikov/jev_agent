from urllib.parse import urlsplit
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JevClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: SecretStr = Field(validation_alias="JEV_API_KEY")


class LLMSettings(BaseSettings):
    """Настройки полного режима. Имя модели выбирается явно, а не скрытым default."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: SecretStr = Field(validation_alias="LLM_API_KEY", min_length=1)
    model: str = Field(validation_alias="LLM_MODEL", min_length=1)
    base_url: str = Field(default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL")
    api_protocol: Literal["openai", "anthropic"] = Field(
        default="openai", validation_alias="LLM_API_PROTOCOL",
    )
    opencode_session: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$",
        validation_alias="LLM_OPENCODE_SESSION",
    )
    timeout_seconds: float = Field(default=60, gt=0, le=300, validation_alias="LLM_TIMEOUT_SECONDS")
    max_output_tokens: int = Field(default=4096, ge=256, le=16384, validation_alias="LLM_MAX_OUTPUT_TOKENS")
    json_mode: bool = Field(default=True, validation_alias="LLM_JSON_MODE")
    response_format: Literal["text", "json_object", "json_schema"] | None = Field(
        default=None, validation_alias="LLM_RESPONSE_FORMAT",
    )
    temperature: float | None = Field(
        default=None, ge=0, le=2, allow_inf_nan=False, validation_alias="LLM_TEMPERATURE",
    )
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = Field(
        default="max_completion_tokens", validation_alias="LLM_TOKEN_LIMIT_PARAMETER"
    )
    max_retries: int = Field(default=2, ge=0, le=5, validation_alias="LLM_MAX_RETRIES")
    min_request_interval_seconds: float = Field(
        default=0, ge=0, le=60, allow_inf_nan=False,
        validation_alias="LLM_MIN_REQUEST_INTERVAL_SECONDS",
    )
    rate_limit_retries: int | None = Field(
        default=None, ge=0, le=5, validation_alias="LLM_RATE_LIMIT_RETRIES",
    )
    rate_limit_backoff_seconds: float = Field(
        default=30, ge=1, le=60, allow_inf_nan=False,
        validation_alias="LLM_RATE_LIMIT_BACKOFF_SECONDS",
    )
    input_price_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, validation_alias="LLM_INPUT_PRICE_PER_MILLION",
    )
    output_price_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, validation_alias="LLM_OUTPUT_PRICE_PER_MILLION",
    )

    @model_validator(mode="after")
    def validate_protocol_options(self) -> "LLMSettings":
        if self.api_protocol == "anthropic" and self.response_format not in (None, "text"):
            raise ValueError("Для Messages API задайте LLM_RESPONSE_FORMAT=text; JSON проверяется локально")
        return self

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        url = urlsplit(value)
        local_http = url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        if (not url.hostname or (url.scheme != "https" and not local_http)
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("LLM_BASE_URL: нужен HTTPS либо HTTP localhost, без credentials/query/fragment")
        return value.rstrip("/")


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_attempts: int = Field(default=2, ge=1, le=3, validation_alias="EXTRACTION_MAX_ATTEMPTS")
    wrong_threshold: float = Field(default=0.7, gt=0, lt=1, validation_alias="JEV_WRONG_THRESHOLD")
