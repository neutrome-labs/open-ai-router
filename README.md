# Open AI Router (Gateway)

# Why another gateway

- Caddy-based -> fast, confugurable, reliable;
- Go-based -> exteremely fast, error-resilient;
- Built-in fallback over providers, eg. `if OpenAI returns 4xx/5xx, transparently retry with Openrouter`;
- [Plugins](#plugins) -> programable, including:
    - Fallback over models, eg. `if GPT-5 is not available, transparently retry with GPT-4.1`;
    - Autocompact (infinite context) -> summarize messages to fit into model limits;
    - Strip completed tools data to save tokens;

# Philosophy

# Features Map

Style                   | Server  | Client
------------------------|---------|--------
OpenAI Chat Completions | Full    | Full 
OpenAI Responses        | Planned | Full
Anthropic Messages      | Full    | Full
Google GenAI            | Full    | Full
Cloudflare Workers AI   | Planned | Planned
Cloudflare AI Gateway   | Planned | Planned
AIL (Raw)               | Full    | Full

# Plugins

### posthog

### models

### parallel

### select

### fuzz

### zip

### stools

### dspy

The `dspy` plugin delegates inference to a Python DSPy sidecar, enabling DSPy modules (ChainOfThought, ReAct, Predict, RLM) as transparent plugins.

**Syntax:** `model+dspy`, `model+dspy:kind`, or `model+dspy:kind:signature`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kind` | `cot` | DSPy module: `predict`, `cot`, `react`, `rlm` |
| `signature` | `history, question -> answer` | DSPy signature (URL-encoded if special chars) |

**Usage examples:**
```bash
# Default Chain of Thought
curl -X POST http://localhost:3000/v1/chat/completions \
  -d '{"model": "gpt-4o-mini+dspy", "messages": [{"role": "user", "content": "What is 2+2?"}]}'

# ReAct agent with tools
curl -X POST http://localhost:3000/v1/chat/completions \
  -d '{"model": "gpt-4o+dspy:react", "messages": [{"role": "user", "content": "Search for..."}], "tools": [...]}'

# Custom signature
curl -X POST http://localhost:3000/v1/chat/completions \
  -d '{"model": "gpt-4o+dspy:cot:context,%20question%20->%20answer", "messages": [...]}'
```

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `DSPY_SIDECAR_URL` | `http://localhost:8780` | Sidecar base URL |
| `DSPY_TIMEOUT` | `120s` | Request timeout |

**Setup:**
```bash
# Install and start the sidecar
make dspy-sidecar-install
make dspy-sidecar

# In another terminal, start the router
make run
```

See [docs/dspy_plugin.md](docs/dspy_plugin.md) for architecture details.

# ail

The `ail` directive exposes a raw AIL (AI Intermediate Language) endpoint. You can POST AIL programs directly — in binary (`.ail`) or text (`.ail.txt`) format — and receive inference responses in the same format.

**Caddyfile:**
```caddyfile
handle_path /v1/ail* {
    route {
        ail {
            router default
        }
    }
}
```

**Content negotiation:**

| Header         | Value                                      | Meaning                    |
|----------------|--------------------------------------------|----------------------------|
| Content-Type   | `application/x-ail`, `application/octet-stream` | Binary AIL input      |
| Content-Type   | `text/plain`, `text/x-ail`                 | Text (disassembly) input   |
| Accept         | `application/x-ail`                        | Binary AIL output          |
| Accept         | `text/plain`                               | Text (disassembly) output  |

If Content-Type is omitted, the handler auto-detects binary vs text by checking for the AIL magic bytes (`AIL\x00`). If Accept is omitted, the output format mirrors the input.

**Examples:**

```bash
# Send a binary .ail sample, get binary response
curl -X POST http://localhost:9111/v1/ail \
  -H "Content-Type: application/x-ail" \
  --data-binary @samples/request.ail \
  -o response.ail

# Send a text .ail.txt sample, get text response
curl -X POST http://localhost:9111/v1/ail \
  -H "Content-Type: text/plain" \
  -H "Accept: text/plain" \
  --data-binary @samples/request.ail.txt

# Send binary, get text disassembly back
curl -X POST http://localhost:9111/v1/ail \
  -H "Content-Type: application/x-ail" \
  -H "Accept: text/plain" \
  --data-binary @samples/request.ail
```
