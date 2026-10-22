from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ValidationError
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError, TypeSafeRateLimitError

from src.settings import DecisionModelSettings, CLMSettings, JeffSettings, LayaSettings


class DecisionModelError(RuntimeError):
    """Ошибка SDK с безопасным описанием, без тела ответа сервера."""


class DecisionModelRateLimitError(DecisionModelError):
    """Лимит сервиса решений исчерпан; новые кейсы запускать нельзя."""


@dataclass
class DecisionUsage:
    # Это вызовы SDK: его внутренние HTTP-повторы здесь не наблюдаются.
    sdk_calls: int = 0
    responses: int = 0
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


def clm_sdk_options(settings: CLMSettings, *, max_retries: int | None = None) -> dict:
    return {
        # SDK требует ключ. Заглушка исключает подхват чужого TYPESAFE_API_KEY
        # из окружения, когда локальный clm-serve работает без аутентификации.
        "api_key": settings.api_key.get_secret_value() if settings.api_key else "local-clm",
        "base_url": settings.base_url, 
        "model": settings.model,
        "timeout": settings.timeout_seconds,
        "retry": RetryPolicy(max_retries=settings.max_retries if max_retries is None else max_retries),
    }


def jeff_sdk_options(settings: JeffSettings, *, max_retries: int | None = None) -> dict:
    return {
        "api_key": settings.api_key.get_secret_value() if settings.api_key else "local-jeff",
        "base_url": settings.base_url, 
        "model": settings.model,
        "timeout": settings.timeout_seconds,
        # RetryPolicy SDK включает 529 и уважает Retry-After. Аудит отключает повторы.
        "retry": RetryPolicy(max_retries=settings.max_retries if max_retries is None else max_retries),
    }


def laya_sdk_options(settings: LayaSettings, *, max_retries: int | None = None) -> dict:
    return {
        "api_key": settings.api_key.get_secret_value() if settings.api_key else "local-laya",
        "base_url": settings.base_url, 
        "model": settings.model,
        "timeout": settings.timeout_seconds,
        "retry": RetryPolicy(max_retries=settings.max_retries if max_retries is None else max_retries),
    }


def laya_request_options(settings: LayaSettings) -> dict:
    # Без этих полей сервер наследует короткие лимиты checkpoint (обычно 1024/256).
    return {"extra_body": {"max_len": settings.max_len, "head_max_len": settings.head_max_len}}


def prepare_clm_questions(questions: dict) -> dict:
    """CLM кодирует instructions и ответы отдельно: вопрос нужен в обоих контекстах.

    Сохраняем явный instructions.question; для старых вопросов берём criteria.true.
    Оба ответа содержат текст конкретной проверки для контрастивного ранжирования.
    Работаем с отдельным JSON-словарём, не изменяя Pydantic-модель вызывающего кода.
    """
    for question in questions.values():
        instructions, criteria = question.get("instructions"), question.get("criteria")
        if (question.get("type") == "noul" and isinstance(instructions, dict)
                and isinstance(criteria, dict) and isinstance(criteria.get("true"), str)):
            explicit_question = instructions.get("question")
            check = explicit_question if isinstance(explicit_question, str) else criteria["true"]
            question["instructions"] = {**instructions, "question": check}
            question["criteria"] = {**criteria, "false": "No. " + check + " " + criteria.get("false", "")}
            if isinstance(explicit_question, str):
                question["criteria"]["true"] = "Yes. " + check + " " + criteria["true"]
    return questions


class DecisionModelClient:
    """Клиент TypeSafe-совместимого API моделей решений."""

    name = "Decision model"

    def __init__(self, api_key: str, *, sdk_options: dict | None = None,
                 request_options: dict | None = None, name: str | None = None):
        if name is not None:
            self.name = name
        self.api_key = api_key
        self._sdk_options = sdk_options or {}
        self._request_options = request_options or {}
        self.usage = DecisionUsage()

    def prepare_questions(self, questions: dict) -> dict:
        return questions

    def safe_error(self, error: Exception) -> DecisionModelError:
        return DecisionModelError(f"Ошибка {self.name}: {type(error).__name__}")

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
                    questions=self.prepare_questions(questions.model_dump(mode="json", by_alias=True)),
                    **self._request_options,
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
            raise DecisionModelRateLimitError(f"{self.name}: HTTP 429, лимит API не снят; повторите запуск позже") from None
        except (TypeSafeError, ValidationError) as error:
            raise self.safe_error(error) from None
        finally:
            self.usage.elapsed_seconds += perf_counter() - started


class AsyncCLM(DecisionModelClient):
    name = "CLM"

    def __init__(self, settings: CLMSettings):
        options = clm_sdk_options(settings)
        super().__init__(options.pop("api_key"), sdk_options=options)

    def prepare_questions(self, questions: dict) -> dict:
        return prepare_clm_questions(questions)


class AsyncJeff(DecisionModelClient):
    name = "Jeff"

    def __init__(self, settings: JeffSettings):
        options = jeff_sdk_options(settings)
        super().__init__(options.pop("api_key"), sdk_options=options)

    def safe_error(self, error: Exception) -> DecisionModelError:
        status = getattr(error, "status", None)
        if status == 529:
            return DecisionModelRateLimitError("Jeff: HTTP 529, сервер занят; повторите запуск позже")
        if status == 422:
            return DecisionModelError("Jeff: HTTP 422; проверьте модель, формат вопросов и JEFF_MAX_TOKENS сервера")
        return super().safe_error(error)


class AsyncLaya(DecisionModelClient):
    name = "Laya"

    def __init__(self, settings: LayaSettings):
        options = laya_sdk_options(settings)
        super().__init__(options.pop("api_key"), sdk_options=options,
                         request_options=laya_request_options(settings))

    def safe_error(self, error: Exception) -> DecisionModelError:
        status = getattr(error, "status", None)
        if status == 503:
            return DecisionModelRateLimitError("Laya: HTTP 503, сервер занят; повторите запуск позже")
        if status in (413, 422):
            return DecisionModelError(f"Laya: HTTP {status}; проверьте модель, формат и лимиты контекста сервера")
        return super().safe_error(error)
