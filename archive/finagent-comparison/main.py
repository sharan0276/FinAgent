import os
import asyncio
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import ComparisonResult


SYSTEM_PROMPT = """
You are a financial comparison assistant for FinAgent.

Your task is to compare a target company against historical financial-risk cases.

Rules:
1. Be concise and structured.
2. Do not invent facts, numbers, filings, or company history.
3. Only use the information given in the prompt.
4. If evidence is weak, say the resemblance is weak or superficial.
5. Always mention important caveats and missing data.
"""


def build_agent() -> Agent:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env")

    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
        ),
    )

    agent = Agent(
        model=model,
        output_type=ComparisonResult,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

def load_prompt_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def run():
    agent = build_agent()

    print("FinAgent Comparison Prototype")
    print("Type a file path (e.g., prompt.txt) or type 'manual'\n")

    while True:
        user_input = input("Input: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            if user_input.lower() == "manual":
                prompt = input("Enter prompt:\n")
            else:
                prompt = load_prompt_from_file(user_input)

            result = await agent.run(prompt)

            print("\nValidated Output:")
            print(result.output.model_dump_json(indent=2))
            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(run())