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
