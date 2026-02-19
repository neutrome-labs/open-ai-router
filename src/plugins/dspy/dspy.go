// Package dspy provides the DSPy bridge plugin.
//
// It delegates inference to a Python DSPy sidecar process, enabling
// DSPy modules (ChainOfThought, ReAct, Predict, RLM, …) to be used
// as transparent plugins in the Open AI Router pipeline.
//
// Syntax (model suffix):
//
//	+dspy                                              → kind=cot, sig="history, question -> answer"
//	+dspy:cot                                          → explicit CoT
//	+dspy:react                                        → ReAct agent (tool use)
//	+dspy:predict                                      → bare Predict
//	+dspy:rlm                                          → Recursive Language Model
//	+dspy:cot:context,%20question%20->%20answer         → custom signature (URL-encoded)
//
// The sidecar must be running and reachable at DSPY_SIDECAR_URL (default
// http://localhost:8780).  It receives LM-callback credentials so its
// own dspy.LM calls route back through the router.
package dspy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"go.uber.org/zap"
)

// ─── Defaults ────────────────────────────────────────────────────────────────

const (
	defaultKind      = "cot"
	defaultSignature = "history, question -> answer"
	defaultTimeout   = 5 * time.Minute
)

// validKinds is the set of DSPy module kinds the sidecar understands.
var validKinds = map[string]bool{
	"predict": true,
	"cot":     true,
	"react":   true,
	"rlm":     true,
}

// ─── DSPy Plugin ─────────────────────────────────────────────────────────────

// DSPy is a RecursiveHandlerPlugin that bridges the Go router to a
// Python DSPy sidecar process.
type DSPy struct{}

func (d *DSPy) Name() string { return "dspy" }

// dspyRecursionGuard prevents re-entrant calls when the sidecar calls
// back into the router.
type dspyRecursionGuard struct{}

// ─── RecursiveHandler ────────────────────────────────────────────────────────

func (d *DSPy) RecursiveHandler(
	params string,
	invoker plugin.HandlerInvoker,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) (bool, error) {
	// Recursion guard: if we're already inside a DSPy callback, let the
	// normal pipeline handle this request.
	if r.Context().Value(dspyRecursionGuard{}) != nil {
		return false, nil
	}

	kind, signature := parseParams(params)
	if !validKinds[kind] {
		plugin.Logger.Error("dspy: unknown kind", zap.String("kind", kind))
		http.Error(w, fmt.Sprintf("dspy: unknown kind %q", kind), http.StatusBadRequest)
		return true, nil
	}

	// Build the sidecar request payload.
	payload, err := buildSidecarPayload(kind, signature, prog)
	if err != nil {
		plugin.Logger.Error("dspy: failed to build payload", zap.Error(err))
		http.Error(w, "dspy: "+err.Error(), http.StatusInternalServerError)
		return true, nil
	}

	// Forward auth from the original request so the sidecar's LM calls
	// are attributed to the same user.
	authHeader := r.Header.Get("Authorization")

	sidecarURL := getSidecarURL()
	timeout := getTimeout()

	if prog.IsStreaming() {
		err = d.handleStreaming(sidecarURL, timeout, payload, authHeader, w)
	} else {
		err = d.handleNonStreaming(sidecarURL, timeout, payload, authHeader, w)
	}
	if err != nil {
		plugin.Logger.Error("dspy: sidecar call failed", zap.Error(err))
		// If headers haven't been written yet, return a proper HTTP error.
		// If streaming already started, the SSE error was already emitted
		// inside handleStreaming, so just return the Go error for logging.
		return true, fmt.Errorf("dspy: sidecar error: %w", err)
	}
	return true, nil
}

// ─── Non-streaming path ─────────────────────────────────────────────────────

func (d *DSPy) handleNonStreaming(
	sidecarURL string,
	timeout time.Duration,
	payload *sidecarRequest,
	authHeader string,
	w http.ResponseWriter,
) error {
	payload.Stream = false

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", sidecarURL+"/invoke", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if authHeader != "" {
		req.Header.Set("X-Upstream-Authorization", authHeader)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("sidecar POST: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("sidecar returned %d: %s", resp.StatusCode, string(respBody))
	}

	var sResp sidecarResponse
	if err := json.NewDecoder(resp.Body).Decode(&sResp); err != nil {
		return fmt.Errorf("decode sidecar response: %w", err)
	}

	// Build an AIL response program from the sidecar prediction.
	resProg := buildResponseProgram(payload.Model, payload.Signature, &sResp)

	// Emit as ChatCompletions JSON.
	emitter := &ail.ChatCompletionsEmitter{}
	resData, err := emitter.EmitResponse(resProg)
	if err != nil {
		return fmt.Errorf("emit response: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-DSPy-Kind", payload.Kind)
	_, err = w.Write(resData)
	return err
}

// ─── Streaming path ──────────────────────────────────────────────────────────

func (d *DSPy) handleStreaming(
	sidecarURL string,
	timeout time.Duration,
	payload *sidecarRequest,
	authHeader string,
	w http.ResponseWriter,
) error {
	payload.Stream = true

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", sidecarURL+"/invoke", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if authHeader != "" {
		req.Header.Set("X-Upstream-Authorization", authHeader)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("sidecar POST: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("sidecar returned %d: %s", resp.StatusCode, string(respBody))
	}

	sseWriter := sse.NewWriter(w)
	if err := sseWriter.WriteHeartbeat("ok"); err != nil {
		return err
	}

	w.Header().Set("X-DSPy-Kind", payload.Kind)

	reader := sse.NewDefaultReader(resp.Body)
	events := reader.ReadEvents()

	emitter := &ail.ChatCompletionsEmitter{}
	chunkIndex := 0
	var streamErr error

	for ev := range events {
		if ev.Done {
			break
		}
		if ev.Error != nil {
			_ = sseWriter.WriteError(ev.Error.Error())
			streamErr = ev.Error
			break
		}
		if len(ev.Data) == 0 {
			continue
		}

		var sEvent sidecarStreamEvent
		if err := json.Unmarshal(ev.Data, &sEvent); err != nil {
			plugin.Logger.Debug("dspy: skipping unparseable SSE event", zap.Error(err))
			continue
		}

		switch sEvent.Type {
		case "chunk":
			chunkProg := buildStreamChunk(payload.Model, sEvent.Field, sEvent.Text, chunkIndex == 0)
			chunkData, err := emitter.EmitStreamChunk(chunkProg)
			if err != nil {
				plugin.Logger.Debug("dspy: emit stream chunk error", zap.Error(err))
				continue
			}
			if err := sseWriter.WriteRaw(chunkData); err != nil {
				return err
			}
			chunkIndex++

		case "status":
			// Emit as SSE comment so standard clients ignore it but
			// aware clients can show progress.
			if _, err := w.Write([]byte(":status " + sEvent.Message + "\n\n")); err != nil {
				return err
			}
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

		case "tool_call":
			// Tool call from ReAct — emit as a tool_calls delta.
			chunkProg := buildStreamToolCall(payload.Model, &sEvent)
			chunkData, err := emitter.EmitStreamChunk(chunkProg)
			if err != nil {
				continue
			}
			if err := sseWriter.WriteRaw(chunkData); err != nil {
				return err
			}

		case "prediction":
			// If no chunks were streamed yet (dspy.streamify may skip
			// incremental deltas and emit only a final Prediction),
			// emit the prediction content as stream chunks so the
			// client receives actual data.
			if chunkIndex == 0 && sEvent.Outputs != nil {
				_, outputFields := parseSignatureFields(payload.Signature)

				// Emit reasoning as a thinking delta if present.
				if reasoning, ok := sEvent.Outputs["reasoning"]; ok && reasoning != "" {
					chunkProg := buildStreamChunk(payload.Model, "reasoning", reasoning, chunkIndex == 0)
					if chunkData, err := emitter.EmitStreamChunk(chunkProg); err == nil {
						_ = sseWriter.WriteRaw(chunkData)
						chunkIndex++
					}
				}

				// Emit each output field (except reasoning, already handled).
				for _, field := range outputFields {
					if field == "reasoning" {
						continue
					}
					text, ok := sEvent.Outputs[field]
					if !ok || text == "" {
						continue
					}
					chunkProg := buildStreamChunk(payload.Model, field, text, chunkIndex == 0)
					if chunkData, err := emitter.EmitStreamChunk(chunkProg); err == nil {
						_ = sseWriter.WriteRaw(chunkData)
						chunkIndex++
					}
				}
			}

		case "error":
			// Sidecar reported an error during streaming.
			errMsg := sEvent.Message
			if errMsg == "" {
				errMsg = "unknown sidecar error"
			}
			_ = sseWriter.WriteError(errMsg)
			streamErr = fmt.Errorf("sidecar stream error: %s", errMsg)
		}
	}

	_ = sseWriter.WriteDone()
	return streamErr
}

// ─── Params parsing ──────────────────────────────────────────────────────────

// parseParams splits "kind:signature" where both are optional.
// The signature may be URL-encoded.
func parseParams(params string) (kind, signature string) {
	kind = defaultKind
	signature = defaultSignature

	if params == "" {
		return
	}

	// Split on first and second ':'
	parts := strings.SplitN(params, ":", 2)
	if parts[0] != "" {
		kind = parts[0]
	}

	if len(parts) > 1 && parts[1] != "" {
		sig, err := url.QueryUnescape(parts[1])
		if err != nil {
			sig = parts[1]
		}
		signature = sig
	}

	return
}

// stripDspySuffix removes "+dspy" and any trailing ":params" from a model
// name so the sidecar's loopback calls don't re-trigger the plugin.
// e.g. "openai/gpt-4.1-mini+dspy:cot" → "openai/gpt-4.1-mini"
func stripDspySuffix(model string) string {
	if idx := strings.Index(model, "+dspy"); idx >= 0 {
		return model[:idx]
	}
	return model
}

// ─── Sidecar payload builders ────────────────────────────────────────────────

type sidecarRequest struct {
	Kind      string            `json:"kind"`
	Signature string            `json:"signature"`
	Inputs    map[string]string `json:"inputs"`
	Tools     []sidecarToolDef  `json:"tools,omitempty"`
	Model     string            `json:"model"`
	Stream    bool              `json:"stream"`
	AuthToken string            `json:"auth_token,omitempty"`
}

type sidecarToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type sidecarResponse struct {
	Outputs   map[string]string `json:"outputs"`
	ToolCalls []sidecarToolCall `json:"tool_calls,omitempty"`
}

type sidecarToolCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type sidecarStreamEvent struct {
	Type    string `json:"type"`              // "chunk", "status", "tool_call", "prediction"
	Field   string `json:"field,omitempty"`   // for "chunk": signature field name
	Text    string `json:"text,omitempty"`    // for "chunk": token content
	Message string `json:"message,omitempty"` // for "status"

	// For "tool_call"
	CallID   string          `json:"call_id,omitempty"`
	ToolName string          `json:"tool_name,omitempty"`
	ToolArgs json.RawMessage `json:"tool_args,omitempty"`

	// For "prediction"
	Outputs map[string]string `json:"outputs,omitempty"`
}

// buildSidecarPayload extracts inputs from the AIL program for the sidecar.
func buildSidecarPayload(kind, signature string, prog *ail.Program) (*sidecarRequest, error) {
	inputFields, _ := parseSignatureFields(signature)

	inputs := make(map[string]string)

	// Build structured history from the conversation (all messages except config).
	history := buildHistory(prog)

	// Map AIL data to signature input fields.
	for _, field := range inputFields {
		switch field {
		case "history":
			histJSON, err := json.Marshal(history)
			if err != nil {
				return nil, fmt.Errorf("marshal history: %w", err)
			}
			inputs[field] = string(histJSON)
		case "context":
			inputs[field] = prog.SystemPrompt()
		case "question":
			if lastUser, ok := prog.LastUserMessage(); ok {
				inputs[field] = prog.MessageText(lastUser)
			}
		default:
			// For unknown fields, try mapping the last user message.
			if lastUser, ok := prog.LastUserMessage(); ok {
				inputs[field] = prog.MessageText(lastUser)
			}
		}
	}

	// Extract model name and strip the +dspy... suffix (the plugin
	// parser resolves plugins but does not mutate the model string).
	model := stripDspySuffix(prog.GetModel())

	// Extract tool definitions for ReAct.
	var tools []sidecarToolDef
	if kind == "react" {
		for _, td := range prog.ToolDefs() {
			def := sidecarToolDef{Name: td.Name}
			for i := td.Start; i <= td.End && i < len(prog.Code); i++ {
				switch prog.Code[i].Op {
				case ail.DEF_DESC:
					def.Description = prog.Code[i].Str
				case ail.DEF_SCHEMA:
					def.Schema = prog.Code[i].JSON
				}
			}
			tools = append(tools, def)
		}
	}

	return &sidecarRequest{
		Kind:      kind,
		Signature: signature,
		Inputs:    inputs,
		Tools:     tools,
		Model:     model,
	}, nil
}

// ─── History building ────────────────────────────────────────────────────────

// historyMessage mirrors the dspy.History format: role + content.
type historyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// buildHistory extracts conversation history from the AIL program,
// returning a list of {role, content} dicts suitable for dspy.History.
func buildHistory(prog *ail.Program) []historyMessage {
	var history []historyMessage
	for _, msg := range prog.Messages() {
		var role string
		switch msg.Role {
		case ail.ROLE_SYS:
			role = "system"
		case ail.ROLE_USR:
			role = "user"
		case ail.ROLE_AST:
			role = "assistant"
		case ail.ROLE_TOOL:
			role = "tool"
		default:
			continue
		}
		text := prog.MessageText(msg)
		if text == "" {
			continue
		}
		history = append(history, historyMessage{Role: role, Content: text})
	}
	return history
}

// ─── Signature parsing ───────────────────────────────────────────────────────

// parseSignatureFields splits "a, b -> c, d" into input fields and output fields.
func parseSignatureFields(sig string) (inputs []string, outputs []string) {
	parts := strings.SplitN(sig, "->", 2)
	if len(parts) < 2 {
		return []string{"question"}, []string{"answer"}
	}

	// Parse input fields (left of ->)
	for _, f := range strings.Split(parts[0], ",") {
		f = strings.TrimSpace(f)
		// Strip optional type annotations like "field: str"
		if colonIdx := strings.Index(f, ":"); colonIdx > 0 {
			f = strings.TrimSpace(f[:colonIdx])
		}
		if f != "" {
			inputs = append(inputs, f)
		}
	}

	// Parse output fields (right of ->)
	for _, f := range strings.Split(parts[1], ",") {
		f = strings.TrimSpace(f)
		if colonIdx := strings.Index(f, ":"); colonIdx > 0 {
			f = strings.TrimSpace(f[:colonIdx])
		}
		if f != "" {
			outputs = append(outputs, f)
		}
	}

	if len(inputs) == 0 {
		inputs = []string{"question"}
	}
	if len(outputs) == 0 {
		outputs = []string{"answer"}
	}

	return
}

// ─── Response building ───────────────────────────────────────────────────────

// buildResponseProgram converts a sidecar prediction into an AIL response program.
func buildResponseProgram(model, signature string, resp *sidecarResponse) *ail.Program {
	_, outputFields := parseSignatureFields(signature)

	prog := ail.NewProgram()
	prog.EmitString(ail.RESP_ID, "dspy-"+fmt.Sprintf("%d", time.Now().UnixNano()))
	prog.EmitString(ail.RESP_MODEL, model)

	prog.Emit(ail.MSG_START)
	prog.Emit(ail.ROLE_AST)

	// If the sidecar returned a "reasoning" field, emit it as a thinking block.
	if reasoning, ok := resp.Outputs["reasoning"]; ok && reasoning != "" {
		prog.Emit(ail.THINK_START)
		prog.EmitString(ail.THINK_CHUNK, reasoning)
		prog.Emit(ail.THINK_END)
	}

	// Emit the primary output field(s) as text content.
	var textParts []string
	for _, field := range outputFields {
		if field == "reasoning" {
			continue // already handled as thinking block
		}
		if val, ok := resp.Outputs[field]; ok {
			textParts = append(textParts, val)
		}
	}
	if len(textParts) > 0 {
		prog.EmitString(ail.TXT_CHUNK, strings.Join(textParts, "\n"))
	}

	// Emit tool calls if present (ReAct returning pending client-side tools).
	if len(resp.ToolCalls) > 0 {
		for _, tc := range resp.ToolCalls {
			prog.EmitString(ail.CALL_START, tc.ID)
			prog.EmitString(ail.CALL_NAME, tc.Name)
			if len(tc.Args) > 0 {
				prog.EmitJSON(ail.CALL_ARGS, tc.Args)
			}
			prog.Emit(ail.CALL_END)
		}
		prog.EmitString(ail.RESP_DONE, "tool_calls")
	} else {
		prog.EmitString(ail.RESP_DONE, "stop")
	}

	prog.Emit(ail.MSG_END)
	return prog
}

// buildStreamChunk creates an AIL stream chunk for a text token.
func buildStreamChunk(model, field, text string, isFirst bool) *ail.Program {
	prog := ail.NewProgram()
	prog.EmitString(ail.RESP_MODEL, model)

	if isFirst {
		prog.Emit(ail.STREAM_START)
	}

	if field == "reasoning" {
		prog.EmitString(ail.STREAM_THINK_DELTA, text)
	} else {
		prog.EmitString(ail.STREAM_DELTA, text)
	}

	return prog
}

// buildStreamToolCall creates an AIL stream chunk for a tool call delta.
func buildStreamToolCall(model string, ev *sidecarStreamEvent) *ail.Program {
	prog := ail.NewProgram()
	prog.EmitString(ail.RESP_MODEL, model)

	toolDelta := map[string]any{
		"index": 0,
		"id":    ev.CallID,
		"name":  ev.ToolName,
	}
	if len(ev.ToolArgs) > 0 {
		toolDelta["arguments"] = string(ev.ToolArgs)
	}
	deltaJSON, _ := json.Marshal(toolDelta)
	prog.EmitJSON(ail.STREAM_TOOL_DELTA, deltaJSON)

	return prog
}

// ─── Config helpers ──────────────────────────────────────────────────────────

func getSidecarURL() string {
	if u := os.Getenv("DSPY_SIDECAR_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	return "http://localhost:8780"
}

func getTimeout() time.Duration {
	if t := os.Getenv("DSPY_TIMEOUT"); t != "" {
		if d, err := time.ParseDuration(t); err == nil {
			return d
		}
	}
	return defaultTimeout
}

// ─── Compile-time checks ─────────────────────────────────────────────────────

var _ plugin.RecursiveHandlerPlugin = (*DSPy)(nil)
