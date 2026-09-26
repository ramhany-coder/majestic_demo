"""Builds the 10 prompts from prompts/filter_extractor/*.md and the catalog.

Every template's static part (instructions, allowed list, example) becomes the
SystemMessage; the trailing CONTEXT / Q lines become the HumanMessage. The
static prefix is therefore byte-identical across requests, which is what
provider prompt caching keys on, and the query always comes last.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agents.filter_extractor.catalog import Catalog, get_catalog
from agents.filter_extractor.schemas import CALLS_BY_KEY, CallSpec
from config import settings
from llm.prompt_loader import load_prompt

ALLOWED_SEP = "; "
_DYNAMIC_MARKERS = ("{{query}}", "{{context}}")
TRANSLATE_PROMPT_FILE = "filter_extractor/translate.md"


@dataclass(frozen=True)
class PromptTemplate:
    key: str
    system: str       # static, filled at startup
    human: str        # contains {{query}} (and {{context}} for the names call)

    def messages(self, query: str, context: Optional[Sequence[str]] = None) -> list:
        human = self.human.replace("{{context}}", format_context(context)).replace("{{query}}", query)
        return [SystemMessage(content=self.system), HumanMessage(content=human)]

    def full_text(self, query: str = "", context: Optional[Sequence[str]] = None) -> str:
        """System + human as one string (used for token budgeting)."""
        msgs = self.messages(query, context)
        return msgs[0].content + "\n" + msgs[1].content


def format_context(context: Optional[Sequence[str]]) -> str:
    names = [c for c in (context or []) if c]
    return "; ".join(names) if names else "none"


def split_template(text: str) -> tuple:
    lines = text.splitlines()
    cut = next((i for i, ln in enumerate(lines) if any(m in ln for m in _DYNAMIC_MARKERS)), len(lines))
    return "\n".join(lines[:cut]).rstrip(), "\n".join(lines[cut:]).strip()


def render_static(spec: CallSpec, text: str, catalog: Catalog, allowed_in_prompt: bool) -> str:
    out = []
    for line in text.splitlines():
        if "{{allowed}}" in line:
            if not allowed_in_prompt or not spec.enum_key:
                continue
            line = line.replace("{{allowed}}", ALLOWED_SEP.join(catalog.allowed[spec.enum_key]))
        line = line.replace("{{product_lines}}", catalog.product_lines_text())
        out.append(line)
    return "\n".join(out)


def build_template(spec: CallSpec, catalog: Catalog, allowed_in_prompt: bool = True) -> PromptTemplate:
    static, dynamic = split_template(load_prompt(spec.prompt_file))
    return PromptTemplate(spec.key, render_static(spec, static, catalog, allowed_in_prompt), dynamic)


@lru_cache(maxsize=None)
def get_template(key: str) -> PromptTemplate:
    return build_template(CALLS_BY_KEY[key], get_catalog(), settings.EXTRACTOR_ALLOWED_IN_PROMPT)


@lru_cache(maxsize=1)
def get_translate_template() -> PromptTemplate:
    static, dynamic = split_template(load_prompt(TRANSLATE_PROMPT_FILE))
    return PromptTemplate("translate", static, dynamic)


def count_tokens(text: str) -> int:
    """Offline estimate (tiktoken o200k_base). Real per-provider counts are
    reported by the eval script from the API's usage metadata."""
    return len(_encoder().encode(text))


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def all_templates() -> List[PromptTemplate]:
    return [get_template(k) for k in CALLS_BY_KEY]
