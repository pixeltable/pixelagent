import base64
import io

import PIL
import pixeltable as pxt

MINIMAX_MODELS: dict[str, dict] = {
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "pricing_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
    },
    "MiniMax-M2.7": {
        "context_window": 204_800,
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        "input_modalities": ["text"],
        "thinking": ["always_on"],
    },
}

MINIMAX_ENDPOINTS: dict[str, dict[str, str]] = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
        "docs_root": "https://platform.minimax.io/docs",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
        "docs_root": "https://platform.minimaxi.com/docs",
    },
}


def resolve_openai_base_url(region: str, base_url: str | None = None) -> str:
    if base_url is not None:
        return base_url
    try:
        return MINIMAX_ENDPOINTS[region]["openai_base_url"]
    except KeyError as exc:
        supported = ", ".join(sorted(MINIMAX_ENDPOINTS))
        raise ValueError(f"Unsupported MiniMax region {region!r}; use one of: {supported}") from exc


def with_base_url(kwargs: dict | None, region: str, base_url: str | None) -> dict:
    resolved = dict(kwargs or {})
    resolved.setdefault("base_url", resolve_openai_base_url(region, base_url))
    return resolved


@pxt.udf
def create_messages(
    system_prompt: str,
    memory_context: list[dict],
    current_message: str,
    image: PIL.Image.Image | None = None,
) -> list[dict]:

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_context.copy())

    if not image:
        messages.append({"role": "user", "content": current_message})
        return messages

    bytes_arr = io.BytesIO()
    image.save(bytes_arr, format="jpeg")
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    b64_encoded_image = b64_bytes.decode("utf-8")

    content_blocks = [
        {"type": "text", "text": current_message},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_encoded_image}"},
        },
    ]

    messages.append({"role": "user", "content": content_blocks})
    return messages
