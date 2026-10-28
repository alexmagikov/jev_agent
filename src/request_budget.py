"""Общие лимиты фактических вызовов для режимов и парного аудита."""

import asyncio
from dataclasses import dataclass

from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

from src.client import DecisionModelClient, DecisionModelError, decision_sdk_options
from src.llm import AsyncLLMExtractor


class APICallBudgetError(RuntimeError):
    """Лимит исчерпан до следующего HTTP-запроса."""


@dataclass
class APICallBudget:
    maximum: int
    used: int = 0

    def __post_init__(self):
        if self.maximum < 1:
            raise ValueError("Бюджет должен быть положительным")

    def spend(self):
        if self.used >= self.maximum:
            raise APICallBudgetError("Бюджет API-вызовов исчерпан")
        self.used += 1


class BudgetedExtractor(AsyncLLMExtractor):
    def __init__(self, settings, budget, **kwargs):
        super().__init__(settings, **kwargs)
        self.budget = budget

    async def _send(self, payload):
        self.budget.spend()
        return await super()._send(payload)


class BudgetedDecisionModel(DecisionModelClient):
    """Внутренние повторы SDK выключены; каждый явный повтор расходует бюджет."""

    def __init__(self, settings, budget, *, api_retries=2):
        if not 0 <= api_retries <= 5:
            raise ValueError("Число сетевых повторов должно быть от 0 до 5")
        options = decision_sdk_options(settings, max_retries=0)
        super().__init__(options.pop("api_key"), sdk_options=options, name=settings.model)
        self.budget, self.api_retries = budget, api_retries

    async def ainvoke(self, state, questions, response_model):
        for retry in range(self.api_retries + 1):
            self.budget.spend()
            try:
                return await super().ainvoke(state, questions, response_model)
            except DecisionModelError as error:
                # Базовый клиент скрывает ответ API через raise ... from None.
                # Контекст нужен только для политики повторов/остановки, не для отчётов.
                cause = error.__cause__ or error.__context__
                temporary = (isinstance(cause, TypeSafeAPIConnectionError)
                             or isinstance(cause, TypeSafeAPIError) and (cause.status == 408 or cause.status >= 500))
                if not temporary or retry == self.api_retries:
                    raise error from cause
                await asyncio.sleep(min(2 ** retry, 30))
