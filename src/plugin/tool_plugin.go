package plugin

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// ─── Interface for concrete tool implementations ─────────────────────────────

// ToolHandler defines the methods a concrete on-router tool must provide.
// The ToolPlugin base handles all orchestration (def injection, call dispatch,
// inference re-invocation) by composing existing plugin interfaces.
type ToolHandler interface {
	// ToolName returns the unique tool name for registration.
	ToolName() string

	// ToolDefs returns the AIL instructions for this tool's definitions.
	// Called during the Before phase to inject definitions into the program.
	ToolDefs(params string) []ail.Instruction

	// HandleToolCall executes a call to this tool and returns the result.
	// Return ("", false, nil) if the call wasn't actually for this tool.
	HandleToolCall(params string, callID string, args json.RawMessage, ctx *ToolCallContext) (result string, handled bool, err error)
}

// ToolCallContext carries request-scoped state available to tool handlers.
type ToolCallContext struct {
	// TraceID is the unique trace identifier for the current request.
	TraceID string

	// RequestProg is the original (pre-tool-injection) request AIL program.
	RequestProg *ail.Program
}

// ─── ToolPlugin: composable base ─────────────────────────────────────────────

// ToolPlugin is a reusable base struct that turns any ToolHandler into a
// full plugin by composing existing plugin interfaces:
//
//   - BeforePlugin: injects tool definitions into the AIL program.
//   - RecursiveHandlerPlugin: after inference, checks for tool calls
//     matching this handler; if found, executes them locally, appends
//     synthetic tool-result messages, and re-invokes inference.
//
// Concrete tools embed ToolPlugin and provide a ToolHandler.
// No new chain logic is needed — the chain remains a dumb dispatcher.
type ToolPlugin struct {
	Handler ToolHandler

	// MaxRounds limits the tool-call dispatch loop (default 10).
	MaxRounds int
}

// NewToolPlugin creates a ToolPlugin wrapping the given handler.
func NewToolPlugin(h ToolHandler) *ToolPlugin {
	return &ToolPlugin{Handler: h, MaxRounds: 10}
}

// Name returns the tool handler's name — satisfies Plugin interface.
func (tp *ToolPlugin) Name() string {
	return tp.Handler.ToolName()
}

// Before injects tool definitions — satisfies BeforePlugin.
func (tp *ToolPlugin) Before(params string, _ *services.ProviderService, _ *http.Request, prog *ail.Program) (*ail.Program, error) {
	defs := tp.Handler.ToolDefs(params)
	if len(defs) == 0 {
		return prog, nil
	}
	return injectDefs(prog, defs), nil
}

// toolRecursionGuard is a context key to prevent re-entrant RecursiveHandler calls.
// When ToolPlugin calls InvokeHandler internally, the inner ServeHTTP would
// call RunRecursiveHandlers again. The guard makes the plugin return
// handled=false on re-entry so the inner pipeline runs normally.
type toolRecursionGuard struct{}

// RecursiveHandler intercepts the request flow to handle local tool calls —
// satisfies RecursiveHandlerPlugin.
//
// Captures the provider response (SSE for streaming, JSON/AIL for non-streaming),
// detects tool calls that match this handler's registered tools, dispatches them
// locally, appends the results, and re-invokes inference — repeating up to MaxRounds.
//
// Only in-router tools (those from ToolDefs) are intercepted. Client-provided
// tools pass through transparently — the captured response is replayed to the
// client and the client SDK handles those tool calls normally.
//
// For streaming requests, intermediate rounds (with tool calls) are buffered
// internally and never streamed to the client. The final round (no more in-router
// tool calls) is replayed as-is — the client receives the complete SSE stream
// of the final response.
func (tp *ToolPlugin) RecursiveHandler(
	params string,
	invoker HandlerInvoker,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) (bool, error) {
	// Recursion guard: skip if we're already inside a ToolPlugin invocation.
	// The inner pipeline will proceed with normal (non-recursive) flow,
	// which includes BeforePlugin (tool def injection) → inference → AfterPlugin.
	if r.Context().Value(toolRecursionGuard{}) != nil {
		return false, nil
	}

	maxRounds := tp.MaxRounds
	if maxRounds <= 0 {
		maxRounds = 10
	}

	// Extract trace ID for the tool call context.
	traceID := ""
	if v := r.Context().Value(ContextTraceID()); v != nil {
		traceID, _ = v.(string)
	}

	ctx := &ToolCallContext{
		TraceID:     traceID,
		RequestProg: prog,
	}

	// Set recursion guard so inner InvokeHandler calls don't re-enter.
	guardR := r.WithContext(context.WithValue(r.Context(), toolRecursionGuard{}, true))

	// First round: invoke the normal pipeline and capture the raw response.
	// For streaming requests this captures the full SSE byte stream;
	// for non-streaming it captures the JSON/AIL response body.
	capture := &services.ResponseCaptureWriter{}
	if err := invoker.InvokeHandler(prog, capture, guardR); err != nil {
		// Pipeline failed — don't handle, let the caller deal with it.
		return false, nil
	}

	// Parse captured response into AIL (format-agnostic via invoker).
	resProg, err := invoker.ParseCapturedResponse(capture)
	if err != nil {
		// Can't parse — replay raw response as-is.
		replayCapture(capture, w)
		return true, nil
	}

	// Check if the response has any calls to our tools.
	resultInsts, nHandled := tp.dispatchCalls(params, resProg, ctx)
	if nHandled == 0 {
		// No tool calls for us — replay the captured response.
		// Client-provided tool calls (if any) pass through to the client.
		replayCapture(capture, w)
		return true, nil
	}

	Logger.Debug("ToolPlugin handling tool calls",
		zap.String("tool", tp.Handler.ToolName()),
		zap.Bool("streaming", prog.IsStreaming()),
		zap.Int("tools_dispatched", nHandled))

	// Tool-call dispatch loop.
	currentProg := prog.Clone()
	// Append the assistant's response messages.
	for _, msg := range resProg.Messages() {
		currentProg = currentProg.Append(resProg.ExtractMessage(msg))
	}
	// Append tool results.
	currentProg.Code = append(currentProg.Code, resultInsts...)

	for round := 1; round < maxRounds; round++ {
		Logger.Debug("ToolPlugin re-invoking inference",
			zap.String("tool", tp.Handler.ToolName()),
			zap.Int("round", round))

		capture = &services.ResponseCaptureWriter{}
		if err := invoker.InvokeHandler(currentProg, capture, guardR); err != nil {
			return true, err
		}

		resProg, err = invoker.ParseCapturedResponse(capture)
		if err != nil {
			replayCapture(capture, w)
			return true, nil
		}

		resultInsts, nHandled = tp.dispatchCalls(params, resProg, ctx)
		if nHandled == 0 {
			// Model finished — replay final response to client.
			// For streaming: the captured SSE bytes are replayed, producing
			// a valid SSE stream (delayed first byte, but complete).
			replayCapture(capture, w)
			return true, nil
		}

		// Append assistant response + tool results for next round.
		for _, msg := range resProg.Messages() {
			currentProg = currentProg.Append(resProg.ExtractMessage(msg))
		}
		currentProg.Code = append(currentProg.Code, resultInsts...)
	}

	// Max rounds exhausted — replay the last response as-is.
	Logger.Warn("ToolPlugin max rounds exhausted",
		zap.String("tool", tp.Handler.ToolName()),
		zap.Int("max_rounds", maxRounds))
	replayCapture(capture, w)
	return true, nil
}

// replayCapture writes a captured response (headers + body) to the real writer.
func replayCapture(capture *services.ResponseCaptureWriter, w http.ResponseWriter) {
	for k, vs := range capture.Headers {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.Write(capture.Response)
}

// dispatchCalls checks a response program for tool calls matching our handler
// and returns synthetic tool-result instructions.
func (tp *ToolPlugin) dispatchCalls(
	params string,
	resProg *ail.Program,
	ctx *ToolCallContext,
) (results []ail.Instruction, handled int) {
	// Build the set of function names this handler provides,
	// extracted from the tool definitions (DEF_NAME instructions).
	funcNames := make(map[string]bool)
	for _, inst := range tp.Handler.ToolDefs(params) {
		if inst.Op == ail.DEF_NAME {
			funcNames[inst.Str] = true
		}
	}

	for _, call := range resProg.ToolCalls() {
		if !funcNames[call.Name] {
			continue
		}

		// Extract args JSON.
		var args json.RawMessage
		for i := call.Start; i <= call.End && i < len(resProg.Code); i++ {
			if resProg.Code[i].Op == ail.CALL_ARGS {
				args = resProg.Code[i].JSON
				break
			}
		}

		Logger.Debug("ToolPlugin dispatching call",
			zap.String("tool", call.Name),
			zap.String("call_id", call.CallID))

		result, wasHandled, err := tp.Handler.HandleToolCall(params, call.CallID, args, ctx)
		if err != nil {
			Logger.Error("ToolPlugin handler error",
				zap.String("tool", call.Name),
				zap.Error(err))
			result = "error: " + err.Error()
		}
		if !wasHandled && err == nil {
			continue
		}

		handled++
		results = append(results,
			ail.Instruction{Op: ail.MSG_START},
			ail.Instruction{Op: ail.ROLE_TOOL},
			ail.Instruction{Op: ail.RESULT_START, Str: call.CallID},
			ail.Instruction{Op: ail.RESULT_DATA, Str: result},
			ail.Instruction{Op: ail.RESULT_END},
			ail.Instruction{Op: ail.MSG_END},
		)
	}

	return results, handled
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// BuildToolDef is a convenience helper that builds a complete
// DEF_START..DEF_END instruction sequence from structured data.
func BuildToolDef(name, description string, schema json.RawMessage) []ail.Instruction {
	insts := []ail.Instruction{
		{Op: ail.DEF_START},
		{Op: ail.DEF_NAME, Str: name},
		{Op: ail.DEF_DESC, Str: description},
	}
	if len(schema) > 0 {
		insts = append(insts, ail.Instruction{Op: ail.DEF_SCHEMA, JSON: schema})
	}
	insts = append(insts, ail.Instruction{Op: ail.DEF_END})
	return insts
}

// injectDefs inserts tool definition instructions into the program.
// Placed after the last existing DEF_END, or before the first message,
// or appended if neither exists.
func injectDefs(prog *ail.Program, defs []ail.Instruction) *ail.Program {
	// After last existing DEF_END.
	for i := len(prog.Code) - 1; i >= 0; i-- {
		if prog.Code[i].Op == ail.DEF_END {
			return prog.InsertAfter(i, defs...)
		}
	}
	// Before first message.
	msgs := prog.Messages()
	if len(msgs) > 0 {
		return prog.InsertBefore(msgs[0].Start, defs...)
	}
	// Append.
	result := prog.Clone()
	result.Code = append(result.Code, defs...)
	return result
}

// Compile-time interface checks.
var (
	_ BeforePlugin           = (*ToolPlugin)(nil)
	_ RecursiveHandlerPlugin = (*ToolPlugin)(nil)
)
