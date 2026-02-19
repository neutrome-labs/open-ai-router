package plugin

import (
	"bytes"
	"net/http"
	"strings"

	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"github.com/neutrome-labs/open-ai-router/src/sse"
)

// CaddyModuleInvoker invokes Caddy HTTP modules as plugins.
// Uses context to pass AIL programs without re-serialization.
// The ResponseParser is injected at construction so the invoker is
// decoupled from any particular wire format.
type CaddyModuleInvoker struct {
	module caddyhttp.MiddlewareHandler
	parser ResponseParser
}

func NewCaddyModuleInvoker(module caddyhttp.MiddlewareHandler, parser ResponseParser) *CaddyModuleInvoker {
	return &CaddyModuleInvoker{
		module: module,
		parser: parser,
	}
}

// InvokeHandler invokes the handler with an AIL program stored in request context.
func (inv *CaddyModuleInvoker) InvokeHandler(prog *ail.Program, w http.ResponseWriter, r *http.Request) error {
	// Store the program in context so the handler can pick it up without re-parsing
	newR := r.Clone(r.Context())
	newR = newR.WithContext(ail.ContextWithProgram(newR.Context(), prog))
	return inv.module.ServeHTTP(w, newR, nil)
}

// InvokeHandlerCapture invokes the handler and captures the response as an AIL program.
// The response is captured via ResponseCaptureWriter and parsed back into AIL
// using the injected ResponseParser. Used by tool dispatch and fan-out plugins.
func (inv *CaddyModuleInvoker) InvokeHandlerCapture(prog *ail.Program, r *http.Request) (*ail.Program, error) {
	capture := &services.ResponseCaptureWriter{}
	if err := inv.InvokeHandler(prog, capture, r); err != nil {
		return nil, err
	}

	if len(capture.Response) == 0 {
		return ail.NewProgram(), nil
	}

	return inv.parser.ParseResponse(capture.Response)
}

// InvokeHandlerCaptureStream invokes the handler with streaming enabled,
// captures all SSE events internally, and returns the assembled AIL program.
//
// The captured SSE body is parsed event-by-event: each "data:" line that
// isn't the [DONE] sentinel is fed through the injected ResponseParser and
// appended to the result. This makes intermediate tool-dispatch rounds
// possible on streaming requests without leaking partial output to the client.
func (inv *CaddyModuleInvoker) InvokeHandlerCaptureStream(prog *ail.Program, r *http.Request) (*ail.Program, error) {
	capture := &services.ResponseCaptureWriter{}
	if err := inv.InvokeHandler(prog, capture, r); err != nil {
		return nil, err
	}

	if len(capture.Response) == 0 {
		return ail.NewProgram(), nil
	}

	// If the response is SSE (text/event-stream), parse each chunk.
	ct := ""
	if capture.Headers != nil {
		ct = capture.Headers.Get("Content-Type")
	}
	if strings.HasPrefix(ct, "text/event-stream") {
		return inv.parseSSECapture(capture.Response)
	}

	// Non-SSE fallback: the handler may have decided to respond without streaming.
	return inv.parser.ParseResponse(capture.Response)
}

// parseSSECapture reads captured SSE bytes and reassembles all chunk programs.
func (inv *CaddyModuleInvoker) parseSSECapture(data []byte) (*ail.Program, error) {
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
		chunk, err := inv.parser.ParseResponse(ev.Data)
		if err != nil {
			// Skip unparseable chunks (e.g. heartbeats, metadata)
			continue
		}
		result = result.Append(chunk)
	}
	return result, nil
}

// ParseCapturedResponse parses raw captured response bytes into an AIL program.
// Handles both streaming (SSE) and non-streaming formats by inspecting the
// Content-Type header from the capture.
func (inv *CaddyModuleInvoker) ParseCapturedResponse(capture *services.ResponseCaptureWriter) (*ail.Program, error) {
	if len(capture.Response) == 0 {
		return ail.NewProgram(), nil
	}
	ct := ""
	if capture.Headers != nil {
		ct = capture.Headers.Get("Content-Type")
	}
	if strings.HasPrefix(ct, "text/event-stream") {
		return inv.parseSSECapture(capture.Response)
	}
	return inv.parser.ParseResponse(capture.Response)
}

// Compile-time check.
var _ HandlerInvoker = (*CaddyModuleInvoker)(nil)
