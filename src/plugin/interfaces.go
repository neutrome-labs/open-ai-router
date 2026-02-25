// Package plugin provides the AIL-based plugin system for AI routing.
// Plugins operate on *ail.Program, the universal intermediate representation.
package plugin

import (
	"bytes"
	"context"
	"net/http"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"go.uber.org/zap"
)

// ─── Inference function types ───────────────────────────────────────────────

// InferFunc runs a single inference round through the provider pipeline.
// Plugins call it with a ResponseCaptureWriter to capture intermediate
// results, or the real http.ResponseWriter to stream directly to the client.
type InferFunc func(prog *ail.Program, w http.ResponseWriter, r *http.Request) error

// InferenceContext bundles inference capabilities for recursive handler plugins.
// Created by the endpoint module and passed to RecursiveHandler — replaces
// the old HandlerInvoker interface with simple function pointers + convenience methods.
type InferenceContext struct {
	// Infer runs inference through the current plugin chain
	// (Before → provider → After). Does NOT re-enter recursive handlers
	// or re-run RequestPreamble. Use for tool dispatch loops where the
	// model and plugin chain don't change between rounds.
	Infer InferFunc

	// InferFresh re-enters the full handler (ServeHTTP) with fresh plugin
	// resolution for a different model. Recursive handlers are automatically
	// bypassed on re-entry via a framework-level context guard.
	// Use when the model changes (e.g., sub-agent stripping its suffix).
	InferFresh InferFunc

	// ParseCapture parses raw captured response bytes into an AIL program.
	// Handles both SSE (text/event-stream) and non-streaming formats by
	// inspecting the Content-Type header from the capture.
	ParseCapture func(capture *services.ResponseCaptureWriter) (*ail.Program, error)
}

// Capture runs inference, captures the raw response, and parses it to AIL.
// Returns both the parsed program and the raw capture (for replay).
func (ic *InferenceContext) Capture(prog *ail.Program, r *http.Request) (*ail.Program, *services.ResponseCaptureWriter, error) {
	cap := &services.ResponseCaptureWriter{}
	if err := ic.Infer(prog, cap, r); err != nil {
		return nil, cap, err
	}
	parsed, err := ic.ParseCapture(cap)
	return parsed, cap, err
}

// CaptureFresh runs inference through the full handler (fresh plugin resolution)
// and captures + parses the response. Use when the model changed.
func (ic *InferenceContext) CaptureFresh(prog *ail.Program, r *http.Request) (*ail.Program, *services.ResponseCaptureWriter, error) {
	cap := &services.ResponseCaptureWriter{}
	if err := ic.InferFresh(prog, cap, r); err != nil {
		return nil, cap, err
	}
	parsed, err := ic.ParseCapture(cap)
	return parsed, cap, err
}

// ReplayCapture writes a previously captured response (headers + body)
// to the real client writer.
func ReplayCapture(capture *services.ResponseCaptureWriter, w http.ResponseWriter) {
	for k, vs := range capture.Headers {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.Write(capture.Response)
}

// ParseCapturedResponse parses raw captured response bytes into an AIL program.
// Auto-detects SSE (text/event-stream) vs non-streaming by inspecting the
// Content-Type header. For SSE, each event is parsed with the StreamChunkParser
// and then reassembled into a full-message program (STREAM_* → MSG/CALL/TXT).
// Used by endpoint modules to build the ParseCapture closure for InferenceContext.
func ParseCapturedResponse(capture *services.ResponseCaptureWriter, respParser ResponseParser, streamParser ail.StreamChunkParser) (*ail.Program, error) {
	if len(capture.Response) == 0 {
		return ail.NewProgram(), nil
	}
	ct := ""
	if capture.Headers != nil {
		ct = capture.Headers.Get("Content-Type")
	}
	if strings.HasPrefix(ct, "text/event-stream") {
		return parseSSECapture(capture.Response, streamParser)
	}
	return respParser.ParseResponse(capture.Response)
}

// parseSSECapture reads captured SSE bytes, parses each chunk with StreamChunkParser,
// and reassembles the accumulated streaming opcodes into a full-message program.
func parseSSECapture(data []byte, parser ail.StreamChunkParser) (*ail.Program, error) {
	reader := sse.NewDefaultReader(bytes.NewReader(data))
	events := reader.ReadEvents()

	result := ail.NewProgram()
	for ev := range events {
		if ev.Done || ev.Error != nil {
			break
		}
		if len(ev.Data) == 0 {
			continue
		}
		chunk, err := parser.ParseStreamChunk(ev.Data)
		if err != nil {
			// Skip unparseable chunks (e.g. heartbeats, metadata)
			continue
		}
		result = result.Append(chunk)
	}
	// Convert streaming opcodes (STREAM_DELTA, STREAM_TOOL_DELTA, etc.)
	// into full message opcodes (TXT_CHUNK, CALL_START/CALL_END, etc.)
	// so that ToolCalls() and Messages() work correctly on the result.
	return ail.ReassembleStream(result), nil
}

// ─── recursionBypassKey ─────────────────────────────────────────────────────

// recursionBypassKey is a framework-level context key that tells
// RunRecursiveHandlers to skip all recursive handler plugins.
// Set by InferFresh so that re-entering ServeHTTP for a different
// model doesn't trigger recursive handlers again.
type recursionBypassKey struct{}

// WithRecursionBypass returns a context that causes RunRecursiveHandlers
// to return handled=false immediately. Used by InferFresh.
func WithRecursionBypass(ctx context.Context) context.Context {
	return context.WithValue(ctx, recursionBypassKey{}, true)
}

// HasRecursionBypass checks whether the recursion bypass is set.
func HasRecursionBypass(ctx context.Context) bool {
	_, ok := ctx.Value(recursionBypassKey{}).(bool)
	return ok
}

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

// ResponseParser converts raw captured response bytes into an AIL program.
// Used by InferenceContext.ParseCapture to decode wire-format responses
// (ChatCompletions JSON, AIL binary, etc.) back into AIL programs.
type ResponseParser interface {
	ParseResponse(data []byte) (*ail.Program, error)
}

// RecursiveHandlerPlugin can intercept the request flow and run inference
// rounds via the InferenceContext. Used for plugins that need multiple
// inference calls (tool dispatch, fallback, parallel, etc.).
type RecursiveHandlerPlugin interface {
	Plugin
	// RecursiveHandler is called before normal provider iteration.
	// If handled is true, the plugin has written the response.
	RecursiveHandler(
		params string,
		ic *InferenceContext,
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

// ─── Client style context ───────────────────────────────────────────────────

// clientStyleCtxKey is the context key for the client-facing API style.
type clientStyleCtxKey struct{}

// ContextClientStyleKey returns the context key for the client-facing API style.
// Endpoint modules store their ail.Style in request context under this key so
// recursive plugins (DSPy, tools, etc.) can emit responses in the correct format.
func ContextClientStyleKey() any { return clientStyleCtxKey{} }

// ClientStyleFromContext extracts the client API style from request context.
// Returns StyleChatCompletions as the default if not set.
func ClientStyleFromContext(ctx context.Context) ail.Style {
	if v, ok := ctx.Value(clientStyleCtxKey{}).(ail.Style); ok {
		return v
	}
	return ail.StyleChatCompletions
}
