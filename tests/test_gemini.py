import asyncio
import base64
import io
import json
import os

import httpx
import pixeltable as pxt
import pixeltable.functions.gemini as pxt_gemini
import pytest
from google import genai
from google.genai import types
from PIL import Image

from pixelagent.gemini import Agent

requires_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "P", "CMYK"])
def test_gemini_image_request(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Image turns reach the SDK with valid PNG bytes and preserve transparency."""
    image = Image.new("RGBA", (8, 8), color=(255, 0, 0, 64)).convert(mode)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Red."}]}}],
        })

    client = genai.Client(
        api_key="test-placeholder",
        http_options=types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    monkeypatch.setattr(pxt_gemini, "_genai_client", lambda: client)
    try:
        agent = Agent(name="gemini_image_request", system_prompt="Be helpful.", reset=True)
        assert agent.chat("What color is this?", image=image) == "Red."
    finally:
        asyncio.run(client.aio.aclose())
        client.close()

    assert len(requests) == 1
    blob = requests[0]["contents"][0]["parts"][0]["inlineData"]
    assert blob["mimeType"] == "image/png"
    with Image.open(io.BytesIO(base64.urlsafe_b64decode(blob["data"]))) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == image.size
        assert decoded.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()
    assert agent.memory.count() == 2


@pxt.udf
def weather(city: str) -> str:
    """Returns the weather in a given city."""
    return f"The weather in {city} is sunny."


@pytest.mark.live
@requires_key
@pytest.mark.gemini
@pytest.mark.chat
def test_gemini_agent_chat():
    """Test basic chat functionality with Gemini agent."""
    agent = Agent(
        name="gemini_chat_assistant",
        system_prompt="You're a helpful assistant.",
        reset=True,
    )

    response = agent.chat("Hi, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

    response2 = agent.chat("What was my last question?")
    assert isinstance(response2, str)
    assert len(response2) > 0


@pytest.mark.live
@requires_key
@pytest.mark.gemini
@pytest.mark.tool_calling
def test_gemini_agent_tools():
    """Test tool calling functionality with Gemini agent."""
    agent = Agent(
        name="gemini_tools_assistant",
        system_prompt="You're my assistant.",
        tools=pxt.tools(weather),
        reset=True,
    )

    response = agent.tool_call("Get weather in San Francisco")
    assert isinstance(response, str)
    assert "San Francisco" in response or "sunny" in response
