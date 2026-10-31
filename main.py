import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "user_profile.json"
OUTPUT_PATH = BASE_DIR / "fitness.md"

client = Anthropic()
model = "claude-haiku-5-5"


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system_prompt=None, temperature=1.0, stop_sequences=None):
    params = {
        "model": model,
        "max_tokens": 4096,
        "messages": messages,
        "extra_body": {"temperature": temperature}
    }

    if system_prompt:
        params["system"] = system_prompt

    if stop_sequences is not None:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            f"The API response reached the {params['max_tokens']}-token limit; "
            "increase max_tokens before saving the plan."
        )

    text = "".join(
        block.text for block in message.content if block.type == "text"
    )
    if not text:
        raise RuntimeError(
            f"The API response contained no text content "
            f"(stop_reason={message.stop_reason!r})."
        )
    return text


def main():
    with PROFILE_PATH.open(encoding="utf-8") as profile_file:
        user_profile = json.load(profile_file)
    if not isinstance(user_profile, dict):
        raise ValueError(f"{PROFILE_PATH} must contain a JSON object.")

    messages = []
    system_prompt = """Act as an expert dietitian and CSCS trainer. Generate a personalized plan based on user info.

Output structure:
1. Diet Plan: Breakfast, Lunch, Dinner, Snacks (include macros & dietary limits).
2. Workout Plan: Weekly Cardio + Strength routine (specify exercises, sets, reps, intensity, duration)."""

    add_user_message(messages, json.dumps(user_profile, indent=2))
    response = chat(
        messages=messages,
        system_prompt=system_prompt,
    )

    OUTPUT_PATH.write_text(response, encoding="utf-8")
    print(f"Fitness plan written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
