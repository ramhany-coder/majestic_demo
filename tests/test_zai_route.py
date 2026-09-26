"""Z.ai routes use JSON mode with the schema in the system prompt (no
json_schema response_format, no forced tool choice on that API). No network."""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

import llm.fallback as fallback_module
from llm.client import fallback_client
from llm.llm_models import client_llm

SCHEMA = {"title": "majestic_brand", "type": "object",
          "properties": {"brand": {"type": "array", "items": {"type": "string"}}}, "required": ["brand"]}


def test_zai_model_points_at_zai_endpoint():
    m = client_llm.get_model("zai", "glm-5.3-flash")
    assert m.model_name == "glm-5.3-flash"
    assert "z.ai" in str(m.openai_api_base) or "bigmodel" in str(m.openai_api_base)


def test_schema_goes_into_the_system_message():
    msgs = fallback_module.with_schema_prompt([SystemMessage(content="Extract brands."), HumanMessage(content="q")],
                                              SCHEMA)
    assert len(msgs) == 2 and msgs[0].content.startswith("Extract brands.")
    assert '"title":"majestic_brand"' in msgs[0].content


def test_zai_route_uses_json_mode(monkeypatch):
    seen = {}

    class Structured:
        async def ainvoke(self, messages):
            seen["messages"] = messages
            return {"parsed": {"brand": ["Cerave"]}, "raw": None, "parsing_error": None}

    class Model:
        def with_structured_output(self, schema, **kw):
            seen.update(kw)
            return Structured()

    monkeypatch.setattr(fallback_module.client_llm, "get_cached_model", lambda router, model, **kw: Model())
    msgs = [SystemMessage(content="Extract brands."), HumanMessage(content="cerave cleanser")]
    res = asyncio.run(fallback_client.aconstrained_invoke(msgs, ["zai:glm-5.3-flash"], SCHEMA))
    assert res.data == {"brand": ["Cerave"]} and res.route == "zai:glm-5.3-flash"
    assert seen["method"] == "json_mode" and "strict" not in seen
    assert "majestic_brand" in seen["messages"][0].content
