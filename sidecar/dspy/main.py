"""
DSPy Bridge Sidecar — FastAPI service that executes DSPy modules on behalf
of the Open AI Router's ``+dspy`` plugin.

Environment variables
---------------------
ROUTER_BASE_URL   Base URL of the Open AI Router (default http://localhost:3000)
DSPY_SIDECAR_PORT Port to listen on (default 8780)
DSPY_DEFAULT_LM   Fallback LM model name for the router (default gpt-4o-mini)

The sidecar configures ``dspy.LM`` with ``api_base`` pointing back to the
router so every LM call the DSPy module makes is routed through the same
pipeline (minus the ``+dspy`` suffix, which is stripped by the Go plugin).

Streaming
---------
When ``stream=True`` the sidecar uses ``dspy.streamify`` to stream progress
back as SSE events.  Each event is a JSON object with a ``type`` field:

  * ``chunk``      — incremental token for a signature field
  * ``status``     — status/progress message from DSPy internals
  * ``tool_call``  — tool invocation from ReAct
  * ``prediction`` — final prediction (all output fields)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from typing import Any

import dspy
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ─── Configuration ────────────────────────────────────────────────────────────

ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", "http://localhost:3000/inference/v1")
SIDECAR_PORT = int(os.getenv("DSPY_SIDECAR_PORT", "8780"))
DEFAULT_LM = os.getenv("DSPY_DEFAULT_LM", "gpt-4o-mini")

logger = logging.getLogger("dspy_sidecar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="DSPy Bridge Sidecar", version="0.1.0")


# ─── LM factory ──────────────────────────────────────────────────────────────

def build_lm(model: str, auth_token: str | None = None) -> dspy.LM:
    """Create a dspy.LM that calls back into the router."""
    api_base = ROUTER_BASE_URL
    api_key = auth_token or "sidecar-internal"
    return dspy.LM(
        model=f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        # Reasonable defaults; callers can override via DSPy settings if needed.
        temperature=0.7,
        max_tokens=4096,
    )


# ─── Module factory ──────────────────────────────────────────────────────────

def build_module(kind: str, signature: str, tools: list[dict] | None = None) -> dspy.Module:
    """Instantiate the appropriate DSPy module for *kind*."""
    if kind == "predict":
        return dspy.Predict(signature)
    elif kind == "cot":
        return dspy.ChainOfThought(signature)
    elif kind == "react":
        dspy_tools = _convert_tools(tools or [])
        return dspy.ReAct(signature, tools=dspy_tools)
    elif kind == "rlm":
        # RLM is only available in newer DSPy builds; fall back to CoT.
        if hasattr(dspy, "RLM"):
            return dspy.RLM(signature)
        logger.warning("dspy.RLM not available, falling back to ChainOfThought")
        return dspy.ChainOfThought(signature)
    else:
        raise ValueError(f"Unknown DSPy kind: {kind!r}")


def _convert_tools(tools: list[dict]) -> list:
    """Convert sidecar tool definitions to DSPy-compatible tool objects.

    DSPy ReAct expects tool callables (or dspy.Tool wrappers).  For now we
    create thin stubs that simply return a JSON placeholder — the actual
    tool execution is handled by the Go router's ToolPlugin.
    """
    dspy_tools = []
    for td in tools:
        name = td.get("name", "unknown")
        desc = td.get("description", "")
        schema = td.get("schema", {})

        # Create a stub function that DSPy can inspect.
        def _make_stub(n: str, d: str, s: dict):
            def stub(**kwargs: Any) -> str:
                return json.dumps({
                    "__tool_call__": True,
                    "name": n,
                    "args": kwargs,
                })
            stub.__name__ = n
            stub.__doc__ = d
            # Attach schema for DSPy introspection.
            stub.__json_schema__ = s
            return stub

        dspy_tools.append(_make_stub(name, desc, schema))
    return dspy_tools


# ─── History → DSPy messages ─────────────────────────────────────────────────

def build_history_value(raw_history: str | list) -> list[dict[str, str]]:
    """Parse the ``history`` input into the format DSPy expects.

    The Go plugin serialises history as a JSON array of ``{role, content}``
    dicts.  If it arrives as a string (JSON-encoded), decode it first.
    """
    if isinstance(raw_history, str):
        try:
            raw_history = json.loads(raw_history)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw_history, list):
        return []
    return [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in raw_history
        if isinstance(m, dict) and m.get("content")
    ]


# ─── Non-streaming invocation ────────────────────────────────────────────────

def invoke_sync(module: dspy.Module, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the DSPy module synchronously and return the prediction."""
    prediction = module(**inputs)

    # Extract all output fields.
    outputs: dict[str, str] = {}
    if hasattr(prediction, "items"):
        for k, v in prediction.items():
            outputs[k] = str(v) if v is not None else ""
    elif hasattr(prediction, "toDict"):
        outputs = {k: str(v) for k, v in prediction.toDict().items()}
    else:
        # Fallback: try common field names.
        for attr in ("answer", "reasoning", "rationale", "response"):
            val = getattr(prediction, attr, None)
            if val is not None:
                outputs[attr] = str(val)

    # Map DSPy's "rationale" to "reasoning" for the Go plugin's THINK block.
    if "rationale" in outputs and "reasoning" not in outputs:
        outputs["reasoning"] = outputs.pop("rationale")

    return {"outputs": outputs}


# ─── Streaming invocation ────────────────────────────────────────────────────

def _parse_output_fields(signature: str) -> list[str]:
    """Extract output field names from a DSPy signature string.

    ``"history, question -> answer"`` → ``["answer"]``
    ``"question -> answer, summary"`` → ``["answer", "summary"]``
    """
    parts = signature.split("->", 1)
    if len(parts) < 2:
        return ["answer"]
    fields = []
    for f in parts[1].split(","):
        f = f.strip()
        # Strip optional type annotations like "field: str"
        if ":" in f:
            f = f.split(":")[0].strip()
        if f:
            fields.append(f)
    return fields or ["answer"]


def _build_stream_listeners(
    module: dspy.Module,
    signature: str,
    kind: str,
) -> list:
    """Build ``StreamListener`` objects for the user-facing output fields.

    CoT and ReAct automatically add a ``reasoning`` field; we always listen
    for it so it can be routed to ``reasoning_content`` on the Go side.
    """
    from dspy.streaming import StreamListener

    output_fields = _parse_output_fields(signature)

    # Always include reasoning (CoT/ReAct add it, Predict doesn't but
    # having an extra listener for a missing field is harmless).
    listen_fields = list(dict.fromkeys(["reasoning"] + output_fields))

    listeners = []
    for field in listen_fields:
        listeners.append(StreamListener(signature_field_name=field))
    return listeners


async def invoke_stream(
    module: dspy.Module,
    inputs: dict[str, Any],
    signature: str,
    kind: str,
):
    """Yield SSE events from a streamified DSPy module."""
    try:
        listeners = _build_stream_listeners(module, signature, kind)
        stream_module = dspy.streamify(
            module,
            stream_listeners=listeners,
            include_final_prediction_in_output_stream=True,
        )
        stream = stream_module(**inputs)

        async for chunk in _iter_stream(stream):
            yield chunk

    except Exception as exc:
        logger.error("Streaming error: %s", traceback.format_exc())
        yield _sse_event({"type": "status", "message": f"error: {exc}"})


async def _iter_stream(stream):
    """Iterate a DSPy stream, yielding SSE event strings.

    With ``stream_listeners`` configured, dspy.streamify yields:
    - ``StreamResponse`` (dspy) — parsed field-level chunks with
      ``.signature_field_name`` and ``.chunk``.  This is the primary path.
    - ``StatusMessage`` — progress info (tool calls, etc.).
    - ``Prediction`` — the final assembled prediction.
    - ``ModelResponseStream`` (litellm) — raw LM deltas, only when listeners
      cannot capture (e.g. cache hits).  Treated as fallback.
    """
    try:
        # dspy.streamify returns an async generator.
        async for item in stream:
            type_name = type(item).__name__

            if type_name == "StatusMessage":
                yield _sse_event({"type": "status", "message": str(item)})

            elif type_name == "ModelResponseStream":
                # Raw litellm streaming chunk — extract delta content.
                choices = getattr(item, "choices", None)
                if choices and len(choices) > 0:
                    delta = getattr(choices[0], "delta", None)
                    if delta:
                        # Reasoning / thinking content
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            yield _sse_event({
                                "type": "chunk",
                                "field": "reasoning",
                                "text": str(reasoning),
                            })
                        # Regular text content
                        content = getattr(delta, "content", None)
                        if content:
                            yield _sse_event({
                                "type": "chunk",
                                "field": "answer",
                                "text": str(content),
                            })

            elif type_name == "StreamResponse":
                # Higher-level DSPy StreamResponse (when stream_listeners are active).
                # Has .signature_field_name and .chunk.
                field = getattr(item, "signature_field_name", "answer")
                # Normalise rationale → reasoning for the Go plugin's THINK block.
                if field == "rationale":
                    field = "reasoning"
                chunk = getattr(item, "chunk", "")
                if chunk:
                    yield _sse_event({
                        "type": "chunk",
                        "field": field,
                        "text": str(chunk),
                    })

            elif type_name == "Prediction":
                # Final prediction — emit all outputs.
                outputs: dict[str, str] = {}
                if hasattr(item, "items"):
                    for k, v in item.items():
                        outputs[k] = str(v) if v is not None else ""
                elif hasattr(item, "toDict"):
                    outputs = {k: str(v) for k, v in item.toDict().items()}

                if "rationale" in outputs and "reasoning" not in outputs:
                    outputs["reasoning"] = outputs.pop("rationale")

                yield _sse_event({"type": "prediction", "outputs": outputs})
            else:
                # Unknown chunk type; emit raw.
                yield _sse_event({"type": "status", "message": str(item)})

    except Exception as exc:
        logger.error("Stream iteration error: %s", traceback.format_exc())
        yield _sse_event({"type": "status", "message": f"error: {exc}"})

    # SSE terminator.
    yield "data: [DONE]\n\n"


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── FastAPI endpoints ───────────────────────────────────────────────────────

@app.post("/invoke")
async def invoke(request: Request):
    """Main endpoint called by the Go DSPy plugin."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    kind: str = body.get("kind", "cot")
    signature: str = body.get("signature", "question -> answer")
    raw_inputs: dict = body.get("inputs", {})
    tools: list = body.get("tools", [])
    model: str = body.get("model", DEFAULT_LM)
    stream: bool = body.get("stream", False)
    auth_token: str | None = body.get("auth_token") or request.headers.get("x-upstream-authorization", "").removeprefix("Bearer ").strip() or None

    logger.info("invoke kind=%s model=%s stream=%s sig=%s", kind, model, stream, signature)

    # Configure DSPy LM per-request using dspy.context (async-safe).
    lm = build_lm(model, auth_token)

    # Build module.
    try:
        module = build_module(kind, signature, tools)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Prepare inputs — deserialise history field if present.
    inputs = dict(raw_inputs)
    if "history" in inputs:
        inputs["history"] = build_history_value(inputs["history"])

    if stream:
        async def _stream_with_ctx():
            with dspy.context(lm=lm):
                async for chunk in invoke_stream(module, inputs, signature, kind):
                    yield chunk

        return StreamingResponse(
            _stream_with_ctx(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            def _sync_with_ctx():
                with dspy.context(lm=lm):
                    return invoke_sync(module, inputs)

            result = await asyncio.to_thread(_sync_with_ctx)
            return JSONResponse(result)
        except Exception as exc:
            logger.error("Invoke error: %s", traceback.format_exc())
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/health")
async def health():
    """Health check for the Go plugin to verify sidecar reachability."""
    return {"status": "ok", "dspy_version": getattr(dspy, "__version__", "unknown")}


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting DSPy sidecar on port %d, router at %s", SIDECAR_PORT, ROUTER_BASE_URL)
    uvicorn.run(app, host="0.0.0.0", port=SIDECAR_PORT, log_level="info")
