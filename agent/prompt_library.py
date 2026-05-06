from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


class PromptLibrary:
    """Loads LLM prompts from agent/prompts.

    Prompt files are plain Markdown. User prompt templates use simple
    {{PLACEHOLDER}} replacement so JSON examples can keep normal braces.
    """

    def __init__(self, prompt_dir: Path = PROMPT_DIR) -> None:
        self.prompt_dir = prompt_dir

    def pair(self, name: str) -> PromptPair:
        return PromptPair(
            system=self.load(f"{name}.system.md"),
            user=self.load(f"{name}.user.md"),
        )

    def load(self, filename: str) -> str:
        return (self.prompt_dir / filename).read_text(encoding="utf-8").strip()

    def render(self, template: str, values: dict[str, str]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered

