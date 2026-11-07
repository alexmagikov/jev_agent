from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ValidationError
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeAPIError, TypeSafeError, TypeSafeRateLimitError

from src.settings import DecisionModelSettings


class DecisionModelError(RuntimeError):
    """Ошибка SDK с безопасным описанием, без тела ответа сервера."""


class DecisionModelRateLimitError(DecisionModelError):
    """Лимит сервиса решений исчерпан; новые кейсы запускать нельзя."""


@dataclass
class DecisionUsage:
    # Это вызовы SDK: его внутренние HTTP-повторы здесь не наблюдаются.
    sdk_calls: int = 0
    responses: int = 0
    # Ответы сервера с HTTP-ошибкой (включая 429): вызов отклонён и не тарифицируется.
    error_responses: int = 0
    usage_reports: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0


def decision_sdk_options(settings: DecisionModelSettings, *, max_retries: int | None = None) -> dict:
    return {
        "api_key": settings.api_key.get_secret_value(),
        "base_url": settings.base_url,
        "model": settings.model,
        "timeout": settings.timeout_seconds,
        "retry": RetryPolicy(max_retries=settings.max_retries if max_retries is None else max_retries),
    }


class DecisionModelClient:
    """Клиент TypeSafe-совместимого API моделей решений."""

    name = "Decision model"

    def __init__(self, api_key: str, *, sdk_options: dict | None = None, name: str | None = None):
        if name is not None:
            self.name = name
        self.api_key = api_key
        self._sdk_options = sdk_options or {}
        self.usage = DecisionUsage()

    async def ainvoke[ResponseT: BaseModel](
        self,
        state: BaseModel,
        questions: BaseModel,
        response_model: type[ResponseT],
    ) -> ResponseT:
        started = perf_counter()
        self.usage.sdk_calls += 1
        try:
            async with AsyncTypeSafeClient(api_key=self.api_key, **self._sdk_options) as client:
                # Без response_model SDK сохраняет usage; типизацию ответа выполняем ниже.
                response = await client.system_one(
                    state=state.model_dump(mode="json", by_alias=True),
                    questions=questions.model_dump(mode="json", by_alias=True),
                )

                self.usage.responses += 1
                usage = getattr(response, "usage", None)
                counts = (getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))

                if all(type(count) is int and count >= 0 for count in counts):
                    self.usage.usage_reports += 1
                    self.usage.input_tokens += counts[0]
                    self.usage.output_tokens += counts[1]
                return response_model.model_validate(response.model_dump(mode="json"))
        except TypeSafeRateLimitError:
            self.usage.error_responses += 1
            raise DecisionModelRateLimitError(f"{self.name}: HTTP 429, лимит API не снят; повторите запуск позже") from None
        except (TypeSafeError, ValidationError) as error:
            if isinstance(error, TypeSafeAPIError):
                self.usage.error_responses += 1
            raise DecisionModelError(f"Ошибка {self.name}: {type(error).__name__}") from None
        finally:
            self.usage.elapsed_seconds += perf_counter() - started
