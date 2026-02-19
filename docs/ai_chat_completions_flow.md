# Chat Completions Data Flow

This document describes how HTTP requests and responses (headers and bodies) flow through the `ai_chat_completions` and `ail` Caddy modules.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Client
        REQ[HTTP Request]
        RES[HTTP Response]
    end
    
    subgraph Caddy["Caddy Server"]
        subgraph Module["ChatCompletionsModule"]
            SERVE[ServeHTTP]
            HANDLE[handleRequest]
            CHAT[serveChatCompletions]
            STREAM[serveChatCompletionsStream]
        end
        
        subgraph Services
            ROUTER[RouterModule]
            AUTH[AuthService]
            PROVIDER[ProviderService]
        end
        
        subgraph Plugins["Plugin Chain"]
            BEFORE[RunBefore]
            AFTER[RunAfter]
            CHUNK[RunAfterChunk]
            STREAMEND[RunStreamEnd]
            ERROR[RunError]
            RECURSIVE[RunRecursiveHandlers]
        end
        
        subgraph Drivers["Driver Layer"]
            CMD[InferenceCommand]
            CONV[StyleConverter]
        end
    end
    
    subgraph Provider["AI Provider"]
        API[Provider API]
    end
    
    REQ --> SERVE
    SERVE --> AUTH
    SERVE --> ROUTER
    SERVE --> RECURSIVE
    RECURSIVE --> HANDLE
    
    HANDLE --> BEFORE
    BEFORE --> CHAT
    BEFORE --> STREAM
    
    CHAT --> CONV
    CHAT --> CMD
    CHAT --> AFTER
    
    STREAM --> CONV
    STREAM --> CMD
    STREAM --> CHUNK
    STREAM --> STREAMEND
    
    CMD --> API
    API --> CMD
    
    AFTER --> RES
    CHUNK --> RES
    STREAMEND --> RES
    ERROR --> RES
```

## Detailed Request Flow

### 1. Entry Point - ServeHTTP

```mermaid
sequenceDiagram
    participant Client
    participant ServeHTTP as ServeHTTP
    participant Router as RouterModule
    participant Auth as AuthService
    participant Plugins as PluginChain
    participant HandleReq as handleRequest
    
    Client->>ServeHTTP: HTTP POST /chat/completions
    Note over ServeHTTP: Read request body (io.ReadAll)
    Note over ServeHTTP: ParsePartialJSON(body) -> PartialJSON
    Note over ServeHTTP: TryGetFromPartialJSON[string](pj, "model")
    Note over ServeHTTP: TryGetFromPartialJSON[bool](pj, "stream")
    
    ServeHTTP->>Router: GetRouter(routerName)
    Router-->>ServeHTTP: RouterModule
    
    ServeHTTP->>Auth: CollectIncomingAuth(request)
    Note over Auth: Extract auth from headers<br/>Set context values (user_id, key_id)
    Auth-->>ServeHTTP: Modified request with context
    
    ServeHTTP->>Plugins: TryResolvePlugins(url, model)
    Note over Plugins: Parse URL path plugins<br/>Parse model suffix plugins<br/>Add head/tail plugins
    Plugins-->>ServeHTTP: PluginChain
    
    Note over ServeHTTP: Generate trace_id (UUID)<br/>Add to request context
    
    ServeHTTP->>Plugins: RunRecursiveHandlers()
    alt Plugin handles request
        Note over Plugins: e.g. +dspy (DSPy bridge),<br/>tool_plugin (tool dispatch)
        Plugins-->>Client: Plugin-generated response
    else No recursive handler
        ServeHTTP->>HandleReq: handleRequest()
        HandleReq-->>Client: Provider response
    end
```

### 2. Request Body Processing (PartialJSON)

The system uses `styles.PartialJSON` (a `map[string]json.RawMessage`) to enable lazy parsing - the body is parsed once at the top level, but nested fields like `messages` are only parsed when actually needed.

```mermaid
flowchart LR
    subgraph "Request Body Flow"
        direction TB
        RAW["Raw HTTP Body<br/>[]byte"]
        PARSE["ParsePartialJSON<br/>styles.PartialJSON"]
        EXTRACT["TryGetFromPartialJSON<br/>stream, model"]
        CLONE["Clone PartialJSON<br/>pj.Clone()"]
        BEFORE["RunBefore Plugins<br/>Modify PartialJSON"]
        CONVERT["Style Converter<br/>PartialJSON transform"]
        PROVIDER["Provider Request<br/>PartialJSON"]
    end
    
    RAW --> PARSE
    PARSE --> EXTRACT
    EXTRACT --> CLONE
    CLONE --> BEFORE
    BEFORE --> CONVERT
    CONVERT --> PROVIDER
```

**Key Benefits of PartialJSON:**
- Parse once at entry, avoid repeated `json.Unmarshal` calls
- Access top-level fields (`model`, `stream`) without parsing nested content
- Plugins can modify fields via `pj.Set(key, value)` without full round-trip
- Clone is cheap - just copies map pointers to `json.RawMessage` slices

### 3. Provider Resolution

```mermaid
flowchart TB
    subgraph "Provider Resolution"
        MODEL["Input Model String<br/>e.g. 'provider/gpt-4+plugin:arg'"]
        
        subgraph Parse["Parse Model"]
            STRIP["Strip Plugin Suffix<br/>'gpt-4'"]
            PREFIX["Check Provider Prefix<br/>'provider/gpt-4' -> 'gpt-4'"]
        end
        
        subgraph Lookup["Provider Lookup"]
            EXPLICIT[Explicit Provider<br/>from prefix]
            DEFAULT[Default Provider<br/>for model]
            ORDER[Providers Order<br/>fallback list]
        end
        
        PROVIDERS[Ordered Provider List]
    end
    
    MODEL --> STRIP
    STRIP --> PREFIX
    PREFIX --> EXPLICIT
    EXPLICIT --> DEFAULT
    DEFAULT --> ORDER
    EXPLICIT --> PROVIDERS
    DEFAULT --> PROVIDERS
    ORDER --> PROVIDERS
```

## Non-Streaming Flow

```mermaid
sequenceDiagram
    participant Handle as handleRequest
    participant Plugins as PluginChain
    participant Serve as serveChatCompletions
    participant Converter as StyleConverter
    participant Driver as ChatCompletions
    participant Provider as AI Provider
    participant Writer as ResponseWriter
    
    Handle->>Handle: ResolveProvidersOrderAndModel()
    
    loop For each provider
        Handle->>Plugins: RunBefore(provider, reqJson PartialJSON)
        Note over Plugins: Logger, Models, Custom plugins
        Plugins-->>Handle: Modified PartialJSON
        
        Handle->>Serve: serveChatCompletions()
        
        Serve->>Converter: ConvertRequest(reqJson, ChatCompletions, providerStyle)
        Note over Converter: Passthrough if same style<br/>Transform PartialJSON if different
        Converter-->>Serve: Provider-format PartialJSON
        
        Serve->>Driver: DoInference(provider, reqJson, request)
        
        Driver->>Driver: createRequest()
        Note over Driver: reqJson.Marshal() -> []byte<br/>Clone headers<br/>Set Content-Type: application/json<br/>Set Authorization header
        
        Driver->>Provider: HTTP POST /chat/completions
        Provider-->>Driver: HTTP Response + Body
        
        alt Success (200 OK)
            Note over Driver: ParsePartialJSON(respBody)
            Driver-->>Serve: (response, resJson PartialJSON, nil)
            
            Serve->>Converter: ConvertResponse(resJson, providerStyle, ChatCompletions)
            Converter-->>Serve: ChatCompletions-format PartialJSON
            
            Serve->>Plugins: RunAfter(provider, reqJson, resJson)
            Plugins-->>Serve: Modified PartialJSON
            
            Note over Serve: resJson.Marshal() -> []byte
            Serve->>Writer: Write JSON response
            Note over Writer: Content-Type: application/json
            
        else Error
            Driver-->>Serve: (response, nil, error)
            Serve->>Plugins: RunError(provider, reqJson, error)
            Note over Handle: Try next provider
        end
    end
    
    Handle->>Writer: Set X-Real-Provider-Id header
    Handle->>Writer: Set X-Real-Model-Id header
    Handle->>Writer: Set X-Plugins-Executed header
```

## Streaming Flow

```mermaid
sequenceDiagram
    participant Handle as handleRequest
    participant Plugins as PluginChain
    participant Stream as serveChatCompletionsStream
    participant SSEWriter as SSE Writer
    participant Converter as StyleConverter
    participant Driver as ChatCompletions
    participant Provider as AI Provider
    participant SSEReader as SSE Reader
    
    Handle->>Stream: serveChatCompletionsStream()
    
    Stream->>SSEWriter: NewWriter(responseWriter)
    Note over SSEWriter: Set headers:<br/>Content-Type: text/event-stream<br/>Cache-Control: no-cache<br/>Connection: keep-alive<br/>X-Accel-Buffering: no
    
    Stream->>SSEWriter: WriteHeartbeat("ok")
    Note over SSEWriter: ":ok\n\n"
    
    Stream->>Converter: ConvertRequest(reqJson, ChatCompletions, providerStyle)
    Converter-->>Stream: Provider-format PartialJSON
    
    Stream->>Driver: DoInferenceStream(provider, reqJson, request)
    
    Driver->>Driver: createRequest()
    Note over Driver: reqJson.Marshal() -> []byte
    Driver->>Provider: HTTP POST /chat/completions (stream=true)
    
    Provider-->>Driver: HTTP Response (streaming)
    
    Driver->>Driver: Create chunks channel
    
    Note over Driver: Goroutine starts
    
    Driver->>SSEReader: NewDefaultReader(response.Body)
    
    loop For each SSE event
        SSEReader->>Driver: Event{Data: []byte, Done, Error}
        
        alt Done signal
            Driver->>Driver: Close channel
        else Error
            Driver->>Stream: InferenceStreamChunk{RuntimeError}
        else Data
            Note over Driver: ParsePartialJSON(event.Data)
            Driver->>Stream: InferenceStreamChunk{Data: PartialJSON}
        end
    end
    
    loop For each chunk from channel
        alt RuntimeError
            Stream->>SSEWriter: WriteError(message)
            Stream->>Plugins: RunError()
        else Data chunk
            Stream->>Converter: ConvertResponseChunk(chunkJson, providerStyle, ChatCompletions)
            Converter-->>Stream: ChatCompletions-format PartialJSON
            
            Stream->>Plugins: RunAfterChunk(chunkJson)
            Plugins-->>Stream: Modified PartialJSON
            
            Note over Stream: chunkJson.Marshal() -> []byte
            Stream->>SSEWriter: WriteRaw(chunkData)
            Note over SSEWriter: "data: {...}\n\n"
        end
    end
    
    Stream->>Plugins: RunStreamEnd(lastChunk PartialJSON)
    Stream->>SSEWriter: WriteDone()
    Note over SSEWriter: "data: [DONE]\n\n"
```

## Header Flow Details

### Incoming Request Headers

```mermaid
flowchart LR
    subgraph "Incoming Headers"
        H1[Authorization<br/>Bearer token]
        H2[Content-Type<br/>application/json]
        H3["Accept<br/>*/*"]
        H4[Accept-Encoding<br/>gzip, deflate]
    end
    
    subgraph "Processing"
        AUTH[AuthService.CollectIncomingAuth]
        CLONE[Header.Clone]
    end
    
    subgraph "Outgoing to Provider"
        O1[Authorization<br/>Provider API key]
        O2[Content-Type<br/>application/json]
        O3["Accept-Encoding<br/>REMOVED"]
    end
    
    H1 --> AUTH
    AUTH --> |Extract user context| CLONE
    H2 --> CLONE
    H3 --> CLONE
    H4 --> CLONE
    CLONE --> |Set new auth| O1
    CLONE --> O2
    CLONE --> |Delete| O3
```

### Response Headers

```mermaid
flowchart TB
    subgraph "Non-Streaming Response"
        NS1["Content-Type: application/json"]
        NS2["X-Real-Provider-Id: provider-name"]
        NS3["X-Real-Model-Id: model-name"]
        NS4["X-Plugins-Executed: plugin1,plugin2"]
    end
    
    subgraph "Streaming Response"
        S1["Content-Type: text/event-stream"]
        S2["Cache-Control: no-cache, no-transform"]
        S3["Connection: keep-alive"]
        S4["X-Accel-Buffering: no"]
        S5["X-Real-Provider-Id: provider-name"]
        S6["X-Real-Model-Id: model-name"]
        S7["X-Plugins-Executed: plugin1,plugin2"]
    end
```

## Plugin Chain Execution Order

```mermaid
flowchart TB
    subgraph "Plugin Resolution"
        HEAD["Head Plugins<br/>models, parallel"]
        PATH["Path Plugins<br/>/plugin1:arg/plugin2"]
        MODEL["Model Plugins<br/>model+plugin:arg"]
        TAIL["Tail Plugins<br/>posthog"]
    end
    
    subgraph "Execution Phases"
        RECURSIVE[RecursiveHandler<br/>Can intercept entire flow]
        BEFORE[Before<br/>Modify request]
        INFERENCE[Inference<br/>Provider call]
        AFTER_NS[After<br/>Non-streaming response]
        AFTER_CHUNK[AfterChunk<br/>Each stream chunk]
        STREAM_END[StreamEnd<br/>Stream completion]
        ERROR[OnError<br/>Error handling]
    end
    
    HEAD --> PATH --> MODEL --> TAIL
    
    TAIL --> RECURSIVE
    RECURSIVE --> BEFORE
    BEFORE --> INFERENCE
    INFERENCE --> |Non-streaming| AFTER_NS
    INFERENCE --> |Streaming| AFTER_CHUNK
    AFTER_CHUNK --> STREAM_END
    INFERENCE --> |Error| ERROR
```

## Style Conversion (with PartialJSON)

```mermaid
flowchart LR
    subgraph "Input Style"
        OPENAI_CHAT["ChatCompletions<br/>/chat/completions"]
    end
    
    subgraph "Provider Styles"
        OPENAI["StyleChatCompletions<br/>Passthrough PartialJSON"]
        RESPONSES["StyleResponses<br/>Transform PartialJSON"]
    end
    
    subgraph "Conversion (services.DefaultConverter)"
        REQ_CONV["ConvertRequest<br/>(reqJson, from, to)"]
        RES_CONV["ConvertResponse<br/>(resJson, from, to)"]
        CHUNK_CONV["ConvertResponseChunk<br/>(chunkJson, from, to)"]
    end
    
    OPENAI_CHAT --> |PartialJSON| REQ_CONV
    REQ_CONV --> |Passthrough| OPENAI
    REQ_CONV --> |Transform| RESPONSES
    
    OPENAI --> |PartialJSON| RES_CONV
    RESPONSES --> |PartialJSON| RES_CONV
    RES_CONV --> OPENAI_CHAT
    
    OPENAI --> |PartialJSON| CHUNK_CONV
    RESPONSES --> |PartialJSON| CHUNK_CONV
    CHUNK_CONV --> OPENAI_CHAT
```

## Context Values

```mermaid
flowchart TB
    subgraph "Request Context"
        TRACE[trace_id<br/>UUID per request]
        USER[user_id<br/>From auth]
        KEY[key_id<br/>From auth]
    end
    
    subgraph "Set By"
        MODULE[ChatCompletionsModule<br/>Sets trace_id]
        AUTH[AuthService<br/>Sets user_id, key_id]
    end
    
    subgraph "Used By"
        PLUGINS[Plugins<br/>Logging, analytics]
        DRIVERS[Drivers<br/>Request context]
    end
    
    MODULE --> TRACE
    AUTH --> USER
    AUTH --> KEY
    
    TRACE --> PLUGINS
    USER --> PLUGINS
    KEY --> PLUGINS
    TRACE --> DRIVERS
```

## Error Handling Flow

```mermaid
flowchart TB
    subgraph "Error Sources"
        PARSE[Request Parse Error]
        AUTH_ERR[Auth Error]
        PLUGIN_ERR[Plugin Error]
        CONV_ERR[Conversion Error]
        PROVIDER_ERR[Provider Error]
        STREAM_ERR[Stream Runtime Error]
    end
    
    subgraph "Handling"
        HTTP_ERR[http.Error<br/>Return to client]
        NEXT_PROVIDER[Try Next Provider]
        PLUGIN_NOTIFY[RunError Plugins]
        SSE_ERR[SSE WriteError]
    end
    
    PARSE --> HTTP_ERR
    AUTH_ERR --> HTTP_ERR
    PLUGIN_ERR --> NEXT_PROVIDER
    CONV_ERR --> HTTP_ERR
    PROVIDER_ERR --> PLUGIN_NOTIFY --> NEXT_PROVIDER
    STREAM_ERR --> PLUGIN_NOTIFY --> SSE_ERR
```

## Data Types Summary

| Stage | Type | Description |
|-------|------|-------------|
| HTTP Request Body | `[]byte` | Raw JSON bytes from client |
| Parsed Request | `styles.PartialJSON` | `map[string]json.RawMessage` - lazy parsed |
| Field Access | `TryGetFromPartialJSON[T]` | Type-safe extraction (e.g., `model`, `stream`) |
| Plugin Processing | `styles.PartialJSON` | Modifiable via `Set()`, clonable via `Clone()` |
| Style Conversion | `styles.PartialJSON` | Transformed between formats |
| Provider Request | `styles.PartialJSON` | Serialized to `[]byte` at driver boundary |
| Provider Response | `styles.PartialJSON` | Parsed from provider response |
| Stream Chunk | `drivers.InferenceStreamChunk` | `{Data: styles.PartialJSON, RuntimeError: error}` |
| SSE Event | `sse.Event` | `{Data: []byte, Done: bool, Error: error}` |
| HTTP Response | `[]byte` / SSE stream | Final client response (serialized from PartialJSON) |

### PartialJSON Type Details

```go
// styles/partial_json.go
type PartialJSON map[string]json.RawMessage

// Core operations
ParsePartialJSON(data []byte) (PartialJSON, error)     // Parse raw bytes
TryGetFromPartialJSON[T](pj, key) T                    // Type-safe field access
pj.Set(key, value) error                               // Modify field
pj.Clone() PartialJSON                                 // Shallow clone
pj.Marshal() ([]byte, error)                           // Serialize back to bytes
```

### Driver Interface (with PartialJSON)

```go
// drivers/interfaces.go
type InferenceCommand interface {
    DoInference(p *ProviderService, reqJson styles.PartialJSON, r *http.Request) (*http.Response, styles.PartialJSON, error)
    DoInferenceStream(p *ProviderService, reqJson styles.PartialJSON, r *http.Request) (*http.Response, chan InferenceStreamChunk, error)
}

type InferenceStreamChunk struct {
    Data         styles.PartialJSON
    RuntimeError error
}
```

### Plugin Interfaces (with PartialJSON)

```go
// plugin/interfaces.go
type BeforePlugin interface {
    Before(params string, p *ProviderService, r *http.Request, reqJson styles.PartialJSON) (styles.PartialJSON, error)
}

type AfterPlugin interface {
    After(params string, p *ProviderService, r *http.Request, reqJson styles.PartialJSON, res *http.Response, resJson styles.PartialJSON) (styles.PartialJSON, error)
}

type StreamChunkPlugin interface {
    AfterChunk(params string, p *ProviderService, r *http.Request, reqJson styles.PartialJSON, res *http.Response, chunk styles.PartialJSON) (styles.PartialJSON, error)
}
```

## AIL Handler (`ail` directive)

The `ail` handler provides a raw AIL endpoint that bypasses JSON serialization entirely.
Clients can POST AIL programs in binary or text format and receive inference responses in the same format.

### AIL Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant AIL as AILModule
    participant R as RouterModule
    participant P as PluginChain
    participant D as InferenceCommand
    participant API as Provider API

    C->>AIL: POST /v1/ail (binary or text AIL)
    AIL->>AIL: Detect format (magic bytes or Content-Type)
    alt Binary AIL
        AIL->>AIL: ail.Decode(body)
    else Text AIL
        AIL->>AIL: ail.Asm(body)
    end
    AIL->>R: GetRouter(name)
    AIL->>R: CollectIncomingAuth
    AIL->>P: Resolve plugins and model rewrites
    AIL->>R: ResolveProvidersOrderAndModel
    loop Each provider
        AIL->>P: RunBefore
        alt Non-streaming
            AIL->>D: DoInference
            D->>API: HTTP request
            API-->>D: JSON response
            D-->>AIL: response *ail.Program
            AIL->>P: RunAfter
        else Streaming (collected)
            AIL->>D: DoInferenceStream
            D->>API: HTTP request (SSE)
            loop Each chunk
                API-->>D: SSE chunk
                D-->>AIL: chunk *ail.Program
                AIL->>P: RunAfterChunk
                AIL->>AIL: StreamAssembler.Push
            end
            AIL->>P: RunStreamEnd
            AIL->>AIL: StreamAssembler.Program()
        end
    end
    alt Binary output (Accept: application/x-ail)
        AIL->>AIL: prog.Encode()
        AIL-->>C: binary AIL response
    else Text output (Accept: text/plain)
        AIL->>AIL: prog.Disasm()
        AIL-->>C: text AIL response
    end
```

### Content Negotiation

| Header       | Value                                            | Effect               |
|--------------|--------------------------------------------------|-----------------------|
| Content-Type | `application/x-ail`, `application/octet-stream`  | Binary AIL input      |
| Content-Type | `text/plain`, `text/x-ail`                       | Text AIL input        |
| Content-Type | _(absent)_                                       | Auto-detect via magic |
| Accept       | `application/x-ail`                              | Binary AIL output     |
| Accept       | `text/plain`                                     | Text AIL output       |
| Accept       | _(absent)_                                       | Mirror input format   |
