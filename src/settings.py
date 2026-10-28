from urllib.parse import urlsplit
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_base_url(value: str, env_name: str) -> str:
    url = urlsplit(value)
    local_http = url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
    if (not url.hostname or (url.scheme != "https" and not local_http)
            or url.username or url.password or url.query or url.fragment):
        raise ValueError(f"{env_name}: нужен HTTPS либо HTTP localhost, без credentials/query/fragment")
    return value.rstrip("/")


class DecisionModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: SecretStr = Field(min_length=1, validation_alias=AliasChoices("DECISION_API_KEY", "JEV_API_KEY"))
    base_url: str = Field(default="https://api.typesafe.ai", validation_alias=AliasChoices("DECISION_BASE_URL", "JEV_BASE_URL"))
    model: str = Field(default="jev-latest", min_length=1, validation_alias=AliasChoices("DECISION_MODEL", "JEV_MODEL"))
    timeout_seconds: float = Field(default=120, gt=0, le=600, validation_alias="DECISION_TIMEOUT_SECONDS")
    max_retries: int = Field(default=0, ge=0, le=5, validation_alias="DECISION_MAX_RETRIES")
    input_price_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
        validation_alias=AliasChoices("DECISION_INPUT_PRICE_PER_MILLION", "JEV_INPUT_PRICE_PER_MILLION"),
    )
    output_price_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False,
        validation_alias=AliasChoices("DECISION_OUTPUT_PRICE_PER_MILLION", "JEV_OUTPUT_PRICE_PER_MILLION"),
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value, "DECISION_BASE_URL")


class JeffSettings(BaseSettings):
    """Подключение к jeff-serve; не связано с ключами Jev или CLM."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(default="http://127.0.0.1:8765", validation_alias="JEFF_BASE_URL")
    model: str = Field(default="jeff-latest", min_length=1, validation_alias="JEFF_MODEL")
    api_key: SecretStr | None = Field(default=None, min_length=1, validation_alias="JEFF_API_KEY")
    timeout_seconds: float = Field(default=120, gt=0, le=600, validation_alias="JEFF_TIMEOUT_SECONDS")
    max_retries: int = Field(default=0, ge=0, le=5, validation_alias="JEFF_MAX_RETRIES")
    wrong_threshold: float = Field(default=0.8, gt=0, lt=1, validation_alias="JEFF_WRONG_THRESHOLD")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value, "JEFF_BASE_URL")


class CLMSettings(BaseSettings):
    """Подключение к собственному clm-serve, отдельно от генеративной LLM."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(default="http://127.0.0.1:8700", validation_alias="CLM_BASE_URL")
    model: str = Field(default="clm-latest", min_length=1, validation_alias="CLM_MODEL")
    api_key: SecretStr | None = Field(default=None, min_length=1, validation_alias="CLM_API_KEY")
    timeout_seconds: float = Field(default=120, gt=0, le=600, validation_alias="CLM_TIMEOUT_SECONDS")
    max_retries: int = Field(default=0, ge=0, le=5, validation_alias="CLM_MAX_RETRIES")
    wrong_threshold: float = Field(default=0.8, gt=0, lt=1, validation_alias="CLM_WRONG_THRESHOLD")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value, "CLM_BASE_URL")


class LayaSettings(BaseSettings):
    """Отдельный laya-serve; multilingual выбирается явно, без автороутинга."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(default="http://127.0.0.1:8770", validation_alias="LAYA_BASE_URL")
    model: str = Field(default="multilingual", min_length=1, validation_alias="LAYA_MODEL")
    api_key: SecretStr | None = Field(default=None, min_length=1, validation_alias="LAYA_API_KEY")
    timeout_seconds: float = Field(default=120, gt=0, le=600, validation_alias="LAYA_TIMEOUT_SECONDS")
    max_retries: int = Field(default=0, ge=0, le=5, validation_alias="LAYA_MAX_RETRIES")
    wrong_threshold: float = Field(default=0.8, gt=0, lt=1, validation_alias="LAYA_WRONG_THRESHOLD")
    max_len: int = Field(default=8192, ge=1, le=8192, validation_alias="LAYA_MAX_LEN")
    head_max_len: int = Field(default=2048, ge=1, le=8192, validation_alias="LAYA_HEAD_MAX_LEN")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value, "LAYA_BASE_URL")


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
        return _validate_base_url(value, "LLM_BASE_URL")


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_attempts: int = Field(default=2, ge=1, le=3, validation_alias="EXTRACTION_MAX_ATTEMPTS")
    wrong_threshold: float = Field(default=0.8, gt=0, lt=1, validation_alias=AliasChoices("DECISION_WRONG_THRESHOLD", "JEV_WRONG_THRESHOLD"))
