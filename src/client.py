from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ValidationError
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError, TypeSafeRateLimitError


class JevError(RuntimeError):
    """Ошибка SDK с безопасным описанием, без тела ответа сервера."""


class JevRateLimitError(JevError):
    """Лимит Jev исчерпан после повторов SDK; новые кейсы запускать нельзя."""


@dataclass
class JevUsage:
    # Это вызовы SDK: его внутренние HTTP-повторы здесь не наблюдаются.
    sdk_calls: int = 0
    responses: int = 0
    usage_reports: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0


class AsyncJev:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.usage = JevUsage()

    async def ainvoke[ResponseT: BaseModel](
        self,
        state: BaseModel,
        questions: BaseModel,
        response_model: type[ResponseT],
    ) -> ResponseT:
        started = perf_counter()
        self.usage.sdk_calls += 1
        try:
            async with AsyncTypeSafeClient(api_key=self.api_key) as client:
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
            raise JevRateLimitError("Jev: HTTP 429, лимит API не снят; повторите запуск позже") from None
        except (TypeSafeError, ValidationError) as error:
            raise JevError(f"Ошибка Jev: {type(error).__name__}") from None
        finally:
            self.usage.elapsed_seconds += perf_counter() - started
