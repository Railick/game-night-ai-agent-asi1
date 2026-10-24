import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from uagents import Agent, Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

# Configure these with the actual addresses of your agents.
INTENT_AGENT = os.getenv("INTENT_AGENT_ADDRESS")
CATALOG_AGENT = os.getenv("CATALOG_AGENT_ADDRESS")
RANKING_AGENT = os.getenv("RANKING_AGENT_ADDRESS")


# Shared internal message contracts.
# The other agents must use matching schemas.


class IntentRequest(Model):
    task_id: str
    text: str


class IntentResponse(Model):
    task_id: str
    requirements: dict
    error: str = ""


class CatalogRequest(Model):
    task_id: str
    requirements: dict


class CatalogResponse(Model):
    task_id: str
    games: list[dict]
    error: str = ""


class RankingRequest(Model):
    task_id: str
    requirements: dict
    games: list[dict]


class RankingResponse(Model):
    task_id: str
    games: list[dict]
    error: str = ""


agent = Agent(
    name="gamenight-orchestrator",
    seed=os.getenv("AGENT_SEED"),
    mailbox=True,
)

chat = Protocol(spec=chat_protocol_spec)

# Tracks which user and stage belong to each workflow.
workflows = {}


def send_user_reply(ctx, user_address, text):
    return ctx.send(
        user_address,
        ChatMessage(
            content=[
                TextContent(type="text", text=text),
                EndSessionContent(type="end-session"),
            ]
        ),
    )


@chat.on_message(ChatMessage)
async def handle_user_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(acknowledged_msg_id=msg.msg_id),
    )

    text = " ".join(
        item.text for item in msg.content if isinstance(item, TextContent)
    ).strip()

    if not text:
        await send_user_reply(
            ctx,
            sender,
            "Tell me your group size, platform, budget, and preferred game mode.",
        )
        return

    task_id = str(uuid4())

    workflows[task_id] = {
        "user": sender,
        "stage": "intent",
    }

    ctx.logger.info(f"Starting workflow {task_id}")

    await ctx.send(
        INTENT_AGENT,
        IntentRequest(task_id=task_id, text=text),
    )


@agent.on_message(model=IntentResponse)
async def handle_intent_result(ctx: Context, sender: str, msg: IntentResponse):
    task = workflows.get(msg.task_id)

    if not task or task["stage"] != "intent":
        return

    if msg.error:
        await send_user_reply(ctx, task["user"], f"Intent error: {msg.error}")
        workflows.pop(msg.task_id, None)
        return

    requirements = msg.requirements
    missing = requirements.get("missing_fields", [])

    if missing:
        await send_user_reply(
            ctx,
            task["user"],
            "Please clarify: " + ", ".join(missing),
        )
        workflows.pop(msg.task_id, None)
        return

    task["stage"] = "catalog"
    task["requirements"] = requirements

    await ctx.send(
        CATALOG_AGENT,
        CatalogRequest(
            task_id=msg.task_id,
            requirements=requirements,
        ),
    )


@agent.on_message(model=CatalogResponse)
async def handle_catalog_result(ctx: Context, sender: str, msg: CatalogResponse):
    task = workflows.get(msg.task_id)

    if not task or task["stage"] != "catalog":
        return

    if msg.error:
        await send_user_reply(ctx, task["user"], f"Catalog error: {msg.error}")
        workflows.pop(msg.task_id, None)
        return

    if not msg.games:
        await send_user_reply(
            ctx,
            task["user"],
            "No games matched all your requirements. "
            "Try relaxing your budget, genre, or player constraints.",
        )
        workflows.pop(msg.task_id, None)
        return

    task["stage"] = "ranking"

    await ctx.send(
        RANKING_AGENT,
        RankingRequest(
            task_id=msg.task_id,
            requirements=task["requirements"],
            games=msg.games,
        ),
    )


@agent.on_message(model=RankingResponse)
async def handle_ranking_result(ctx: Context, sender: str, msg: RankingResponse):
    task = workflows.get(msg.task_id)

    if not task or task["stage"] != "ranking":
        return

    if msg.error:
        await send_user_reply(ctx, task["user"], f"Ranking error: {msg.error}")
        workflows.pop(msg.task_id, None)
        return

    games = msg.games[:3]

    if not games:
        answer = "No suitable games could be ranked."
    else:
        lines = ["Here are your top GameNight AI recommendations:\n"]

        for i, game in enumerate(games, start=1):
            lines.append(
                f"{i}. {game.get('name', 'Unknown game')}\n"
                f"   Score: {game.get('score', 'N/A')}\n"
                f"   Reason: {game.get('reason', 'Matches your requirements')}\n"
                f"   Link: {game.get('url', 'Not available')}"
            )

        answer = "\n\n".join(lines)

    await send_user_reply(ctx, task["user"], answer)
    workflows.pop(msg.task_id, None)
    ctx.logger.info(f"Completed workflow {msg.task_id}")


agent.include(chat, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
