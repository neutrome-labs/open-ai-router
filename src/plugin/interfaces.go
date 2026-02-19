// Package plugin provides the AIL-based plugin system for AI routing.
// Plugins operate on *ail.Program, the universal intermediate representation.
package plugin

import (
	"net/http"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// Logger for plugin chain - can be set by modules
var Logger *zap.Logger = zap.NewNop()

// Context keys
type contextKey string

const (
	traceIDKey contextKey = "trace_id"
	userIDKey  contextKey = "user_id"
	keyIDKey   contextKey = "key_id"
)

// ContextTraceID returns the trace ID context key
func ContextTraceID() contextKey { return traceIDKey }

// ContextUserID returns the user ID context key
func ContextUserID() contextKey { return userIDKey }

// ContextKeyID returns the key ID context key
func ContextKeyID() contextKey { return keyIDKey }

// Plugin is the base interface for all plugins
type Plugin interface {
	// Name returns the plugin's identifier
	Name() string
}

// ModelRewritePlugin can rewrite the model name before plugin resolution.
// Used by virtual providers for model aliasing. Runs in a loop until the
// model stabilises, so chained virtual→virtual mappings work naturally.
type ModelRewritePlugin interface {
	Plugin
	// RewriteModel returns the rewritten model and true if it matched,
	// or the original model and false if it didn't.
	RewriteModel(model string) (rewritten string, matched bool)
}

// RequestInitPlugin is called once per request after the initial AIL program
// is parsed and plugin resolution is complete, but before provider iteration.
// Ideal for sampling and observability hooks that need the pre-plugin state.
type RequestInitPlugin interface {
	Plugin
	// OnRequestInit receives the original parsed program before any
	// provider-specific before-plugins mutate it.
	OnRequestInit(r *http.Request, prog *ail.Program)
}

// BeforePlugin processes requests before sending to provider.
// Operates on the AIL program representation.
type BeforePlugin interface {
	Plugin
	// Before is called before the request is sent to the provider.
	// Returns the (possibly modified) program.
	Before(params string, p *services.ProviderService, r *http.Request, prog *ail.Program) (*ail.Program, error)
}

// AfterPlugin processes non-streaming responses.
type AfterPlugin interface {
	Plugin
	// After is called after receiving a complete (non-streaming) response.
	After(params string, p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, resProg *ail.Program) (*ail.Program, error)
}

// StreamChunkPlugin processes individual streaming chunks.
type StreamChunkPlugin interface {
	Plugin
	// AfterChunk is called for each streaming chunk (as an AIL program fragment).
	AfterChunk(params string, p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, chunk *ail.Program) (*ail.Program, error)
}

// StreamEndPlugin handles stream completion.
type StreamEndPlugin interface {
	Plugin
	// StreamEnd is called when the stream completes.
	StreamEnd(params string, p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, lastChunk *ail.Program) error
}

// ErrorPlugin handles errors from provider calls.
type ErrorPlugin interface {
	Plugin
	// OnError is called when a provider call fails.
	OnError(params string, p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, providerErr error) error
}

// HandlerInvoker allows plugins to invoke the outer handler recursively.
// Used by plugins like fallback (retry with different providers) and parallel (fan-out).
type HandlerInvoker interface {
	// InvokeHandler invokes the outer handler with the given AIL program.
	InvokeHandler(prog *ail.Program, w http.ResponseWriter, r *http.Request) error

	// InvokeHandlerCapture invokes the handler and captures the response as an AIL program.
	InvokeHandlerCapture(prog *ail.Program, r *http.Request) (*ail.Program, error)

	// InvokeHandlerCaptureStream invokes the handler with streaming enabled,
	// captures all SSE chunks internally, and returns the assembled AIL program.
	// Used by ToolPlugin for intermediate tool-dispatch rounds on streaming
	// requests: the response is buffered so tool calls can be detected and
	// handled, without streaming partial results to the real client.
	InvokeHandlerCaptureStream(prog *ail.Program, r *http.Request) (*ail.Program, error)

	// ParseCapturedResponse parses raw bytes from a ResponseCaptureWriter
	// into an AIL program. Inspects the Content-Type header to determine
	// the wire format (SSE text/event-stream vs non-streaming JSON/AIL).
	// Used by ToolPlugin to parse captured responses regardless of the
	// underlying module format.
	ParseCapturedResponse(capture *services.ResponseCaptureWriter) (*ail.Program, error)
}

// ResponseParser converts raw captured response bytes into an AIL program.
// Injected into CaddyModuleInvoker so the invoker is not coupled to any
// particular wire format (ChatCompletions JSON, AIL binary, etc.).
type ResponseParser interface {
	ParseResponse(data []byte) (*ail.Program, error)
}

// StreamResponseParser converts captured SSE response bytes into an AIL program.
// Used by InvokeHandlerCaptureStream to reassemble a streamed response.
type StreamResponseParser interface {
	ParseStreamResponse(data []byte) (*ail.Program, error)
}

// RecursiveHandlerPlugin can intercept the request flow and invoke the handler recursively.
// Used for plugins that need to make multiple calls (fallback, parallel, etc.).
type RecursiveHandlerPlugin interface {
	Plugin
	// RecursiveHandler is called before normal provider iteration.
	// If handled is true, the plugin has handled the request.
	RecursiveHandler(
		params string,
		invoker HandlerInvoker,
		prog *ail.Program,
		w http.ResponseWriter,
		r *http.Request,
	) (handled bool, err error)
}

// ProviderLister returns all provisioned provider services.
// Set by the router module during Provision so that plugins
// like fuzz can discover providers without importing modules.
var ProviderLister func() []*services.ProviderService

// PluginInstance represents a plugin with its parameters.
type PluginInstance struct {
	Plugin Plugin
	Params string
}
