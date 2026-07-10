# %%
import openai
import os
import re

from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

# %%
orfeo_api_key = os.getenv("ORFEO_API_KEY")
orfeo_client = openai.OpenAI(
    api_key=orfeo_api_key,
    base_url="https://orfeo-llm.areasciencepark.it/vllm/v1",
)

# %%
# List available models
if orfeo_client:
    models = orfeo_client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"- {model.id}")
else:
    print("ORFEO_API_KEY not found. Please set the environment variable to use the ORFEO API.")

# %%
MODEL_NAME = "google/gemma-4-26B-A4B-it"

feats_desc = [
    "historical or political figures",
    "geographical location",
    "cultural or artistic concepts",
    "scientific or technological concepts",
    "events or time periods",
    "social or economic concepts",
    "emotions or feelings",
    "other abstract concepts",
]
contexts = [
    "Macron is the president of",
    "The Eiffel Tower is located in",
    "The Mona Lisa is a famous artwork in",
    "The theory of relativity was developed by",
    "World War II ended in",
    "Capitalism is an example of",
    "Happiness is an example of",
    "Photosynthesis is a process in",
]

# %%
# --- System prompt with few-shot examples ---
SYSTEM_PROMPT = (
    "You need to evaluate from 0 to 1 how much a provided <description> "
    "is relevant to a provided <context>.\n"
    "Your answer must have the following format: '**score**: <your score>'\n\n"
    "Examples of <description> and <context>:\n\n"
    "1. **description**: 'geographical location'\n"
    "   **context**: 'Obama grew up in'\n"
    "   **score**: 0.9\n\n"
    "2. **description**: 'color of the sky'\n"
    "   **context**: 'Obama grew up in'\n"
    "   **score**: 0.1\n\n"
    "3. **description**: 'technology and computer science'\n"
    "   **context**: 'Apple was founded by'\n"
    "   **score**: 0.8\n"
)


def build_user_turn(description: str, context: str) -> str:
    """Format a single evaluation request."""
    return (
        f"Provide the score for the following <description> and <context>:\n\n"
        f"**description**: '{description}'\n"
        f"**context**: '{context}'"
    )


def parse_score(response_text: str) -> float:
    """Extract the float score from a model response.

    Looks for patterns like '**score**: 0.85' and returns the float value.
    Falls back to NaN if parsing fails.
    """
    match = re.search(r"\*?\*?score\*?\*?\s*:\s*([0-9]*\.?[0-9]+)", response_text)
    if match:
        return float(match.group(1))
    rprint(f"[yellow]Warning: could not parse score from: {response_text!r}[/yellow]")
    return float("nan")


def evaluate_pairs(
    client: openai.OpenAI,
    descriptions: list[str],
    contexts: list[str],
    model: str = MODEL_NAME,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> list[float]:
    """Run a multi-turn conversation scoring each (description, context) pair.

    The first message sets the system prompt with few-shot examples.
    Each subsequent turn appends the new user query and the assistant's
    prior answer, keeping full conversation history so the model stays
    on-task.

    Args:
        client: OpenAI-compatible API client.
        descriptions: List of feature descriptions to evaluate.
        contexts: List of context strings (same length as descriptions).
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.

    Returns:
        List of float scores, one per (description, context) pair.
    """
    assert len(descriptions) == len(contexts), "descriptions and contexts must have the same length"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    scores: list[float] = []

    for i, (desc, ctx) in enumerate(zip(descriptions, contexts)):
        user_msg = build_user_turn(desc, ctx)
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        score = parse_score(assistant_text)
        scores.append(score)
        rprint(f"[{i + 1}/{len(descriptions)}] desc='{desc}' | ctx='{ctx}' → [bold green]{score}[/bold green]")

    return scores


# %%
scores = evaluate_pairs(orfeo_client, feats_desc, contexts)
rprint(f"\n[bold]Final scores:[/bold] {scores}")

# %%
