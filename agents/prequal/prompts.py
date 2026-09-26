"""Builds the rewriter and router messages from prompts/prequal/*.md.

The static part (instructions + examples) is the SystemMessage and is
byte-identical across requests, which is what provider prompt caching keys
on. The trailing HISTORY / LAST_PRODUCTS / QUERY block is the HumanMessage,
so the query always comes last.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agents.prequal.session_context import format_history, format_last_products
from llm.prompt_loader import load_prompt

PROMPT_FILES = {"rewriter": "prequal/rewriter.md", "router": "prequal/router.md"}
_HISTORY_MARKER = "{{history}}"


@dataclass(frozen=True)
class PrequalTemplate:
    key: str
    system: str
    human: str      # holds {{history}}, {{last_products}}, {{query}}

    def messages(self, query: str, history: Sequence[Dict[str, str]] = (),
                 last_products: Sequence[str] = ()) -> list:
        human = (self.human
                 .replace("{{history}}", format_history(history))
                 .replace("{{last_products}}", format_last_products(last_products))
                 .replace("{{query}}", query))
        return [SystemMessage(content=self.system), HumanMessage(content=human)]

    def full_text(self, query: str = "", history: Sequence[Dict[str, str]] = (),
                  last_products: Sequence[str] = ()) -> str:
        msgs = self.messages(query, history, last_products)
        return msgs[0].content + "\n" + msgs[1].content


def split_template(text: str) -> tuple:
    """Cut before the `HISTORY:` label that introduces {{history}} (or before
    the {{history}} line itself when it has no label line)."""
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if _HISTORY_MARKER in ln), None)
    if idx is None:
        raise ValueError("prequal prompt has no {{history}} placeholder")
    if idx > 0 and lines[idx - 1].strip() == "HISTORY:":
        idx -= 1
    return "\n".join(lines[:idx]).rstrip(), "\n".join(lines[idx:]).strip()


@lru_cache(maxsize=None)
def get_template(key: str) -> PrequalTemplate:
    static, dynamic = split_template(load_prompt(PROMPT_FILES[key]))
    return PrequalTemplate(key, static, dynamic)
