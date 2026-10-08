import asyncio
from argparse import ArgumentParser

from src.cases.conversation import ConversationCase
from src.cases.refund import RefundCase
from src.cases.ticket import TicketCase
from src.client import AsyncJev
from src.settings import JevClientSettings


async def main(case_name: str = "ticket") -> None:
    settings = JevClientSettings()
    client = AsyncJev(api_key=settings.api_key.get_secret_value())
    match case_name:
        case "ticket":
            ticket = await TicketCase().run(client)
            print(ticket.answers.department.choice)
            print(ticket.answers.frustration.score)
            print(ticket.answers.is_urgent.noul)
        case "refund":
            refund = await RefundCase().run(client)
            print(refund.answers.refund_requested.noul)
            print(refund.answers.duplicate_charge.noul)
            print(refund.answers.policy_supports_refund.noul)
        case "conversation":
            conversation = await ConversationCase().run(client)
            print(conversation.answers.human_requested.noul)
            print(conversation.answers.repeat_contact.noul)
        
        case _:
            raise ValueError(f"Unknown case: {case_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("case", choices=("ticket", "refund", "conversation"), nargs="?", default="ticket")
    asyncio.run(main(parser.parse_args().case))
