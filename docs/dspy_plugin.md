# DSPy Bridge Plugin — Architecture

## Overview

The `+dspy` plugin bridges the Open AI Router to [DSPy](https://dspy.ai), enabling
DSPy modules (ChainOfThought, ReAct, Predict, RLM) to run as transparent
inference plugins.

```
Client  ──POST /v1/chat/completions──►  Router  ──POST /invoke──►  Sidecar (Python)
                                         ▲                            │
                                         │   loopback LM calls        │
                                         └────────────────────────────┘
```

The sidecar configures `dspy.LM(api_base=router_url)` so every LM call DSPy makes
routes back through the router — **minus the `+dspy` suffix** — enabling full
provider fallback, logging, and all other plugins on the inner calls.

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as "Go Router"
    participant Plugin as "+dspy Plugin"
    participant Sidecar as "Python Sidecar"
    participant LM as "dspy.LM (loopback)"

    Client->>Router: POST /v1/chat/completions<br/>model=gpt-4o+dspy:cot
    Router->>Plugin: RecursiveHandler(params, prog)
    Note over Plugin: Extract inputs from AIL Program<br/>Build sidecar payload
    Plugin->>Sidecar: POST /invoke<br/>{kind, signature, inputs, model, stream}
    
    loop DSPy module execution
        Sidecar->>LM: dspy.LM("openai/gpt-4o", api_base=router)
        LM->>Router: POST /v1/chat/completions<br/>model=gpt-4o (no +dspy)
        Note over Router: Recursion guard skips dspy plugin
        Router-->>LM: LLM response
        LM-->>Sidecar: tokens
    end
    
    Sidecar-->>Plugin: Prediction {outputs, reasoning}
    Note over Plugin: Build AIL response program<br/>THINK_START/CHUNK/END for reasoning<br/>TXT_CHUNK for answer
    Plugin-->>Router: Write response
    Router-->>Client: ChatCompletions JSON
```

### Streaming Variant

```mermaid
sequenceDiagram
    participant Client
    participant Plugin as "+dspy Plugin"
    participant Sidecar as "Python Sidecar"

    Client->>Plugin: stream=true
    Plugin->>Sidecar: POST /invoke (stream=true, Accept: text/event-stream)
    
    loop SSE events
        Sidecar-->>Plugin: {type: "chunk", field: "reasoning", text: "..."}
        Note over Plugin: STREAM_THINK_DELTA → reasoning_content
        Plugin-->>Client: SSE delta (reasoning_content)
        
        Sidecar-->>Plugin: {type: "chunk", field: "answer", text: "..."}
        Note over Plugin: STREAM_DELTA → content
        Plugin-->>Client: SSE delta (content)
        
        Sidecar-->>Plugin: {type: "status", message: "..."}
        Plugin-->>Client: SSE comment
    end
    
    Sidecar-->>Plugin: {type: "prediction", outputs: {...}}
    Plugin-->>Client: data: [DONE]
```

## Plugin Syntax

```
model+dspy                                    → kind=cot, sig="history, question -> answer"
model+dspy:predict                            → Predict module
model+dspy:cot                                → ChainOfThought module
model+dspy:react                              → ReAct agent (with tool use)
model+dspy:rlm                                → Recursive Language Model
model+dspy:cot:context,%20question%20->%20answer  → custom signature (URL-encoded)
```

## Input Mapping

The Go plugin extracts inputs from the AIL program based on the signature's input fields:

| Signature Field | AIL Source |
|----------------|------------|
| `history` | All messages → `[{role, content}, ...]` (JSON) |
| `context` | `SystemPrompt()` |
| `question` | `LastUserMessage()` → `MessageText()` |
| (other) | Falls back to `LastUserMessage()` |

## Output Mapping

| Sidecar Output | AIL Opcode | OpenAI Field |
|---------------|-----------|-------------|
| `reasoning` / `rationale` | `THINK_START` + `THINK_CHUNK` + `THINK_END` | `reasoning_content` |
| `answer` (or other output fields) | `TXT_CHUNK` | `content` |
| tool calls (ReAct) | `CALL_START` + `CALL_NAME` + `CALL_ARGS` + `CALL_END` | `tool_calls` |

## Recursion Guard

The plugin sets a `dspyRecursionGuard{}` context key before the first call.
When the sidecar's LM calls loop back through the router, the guard prevents
the `+dspy` plugin from triggering again — the request falls through to
the normal inference pipeline.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DSPY_SIDECAR_URL` | `http://localhost:8780` | Sidecar base URL |
| `DSPY_TIMEOUT` | `120s` | HTTP timeout for sidecar calls |
| `ROUTER_BASE_URL` | `http://localhost:3000` | Router URL (sidecar config) |
| `DSPY_SIDECAR_PORT` | `8780` | Sidecar listen port |
| `DSPY_DEFAULT_LM` | `gpt-4o-mini` | Fallback model name |

## File Layout

```
src/plugins/dspy/
  dspy.go          Go RecursiveHandlerPlugin
  dspy_test.go     Unit tests (param parsing, signature parsing)
sidecar/
  dspy_sidecar.py  FastAPI sidecar (DSPy execution)
  requirements.txt Python dependencies
```

## Extensibility

The architecture is designed for future extension:

- **New kinds**: Add to `validKinds` map in Go + `build_module()` in Python
- **Optimized programs**: The sidecar's `/invoke` endpoint could accept an
  additional `program_path` field to load compiled DSPy programs
- **Custom tools**: ReAct tool stubs can be replaced with real implementations
  that call external APIs directly from the sidecar
