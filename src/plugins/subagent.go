package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"go.uber.org/zap"
)

// SubAgent is a tool plugin that injects a "run_in_subagent" tool definition
// into the request. When the LLM calls this tool, SubAgent spawns a full
// sub-inference call (via InferenceContext) with the given task, captures
// the response, and returns a textual summary back to the calling model.
//
// This lets the model branch out heavy or specialised work to a separate
// inference round — effectively creating an agent→sub-agent pattern at the
// router level, transparent to the client.
//
// Syntax (model suffix):
//
//	+subagent                           → defaults (same model, max 3 depth)
//	+subagent:max_depth=5               → allow up to 5 nested levels
//
// The sub-inference reuses the same provider pipeline but strips the
// "+subagent" suffix from the model so the inner call doesn't inject
// the tool again (preventing infinite recursion). An additional depth
// counter enforced via AIL metadata guards against runaway nesting.
type SubAgent struct {
	plugin.ToolPlugin
}

// NewSubAgent creates a SubAgent plugin wired to its ToolPlugin base.
func NewSubAgent() *SubAgent {
	s := &SubAgent{}
	s.ToolPlugin = *plugin.NewToolPlugin(s)
	return s
}

const (
	subagentToolName = "run_in_subagent"
	defaultMaxDepth  = 3
)

var subagentToolSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"task": {
			"type": "string",
			"description": "A detailed description of the task for the sub-agent to perform. The sub-agent will process this task in a separate inference call and return a summarised result."
		},
		"system_prompt": {
			"type": "string",
			"description": "Optional system prompt to guide the sub-agent's behaviour. If omitted, the sub-agent receives only the task as a user message."
		}
	},
	"required": ["task"],
	"additionalProperties": false
}`)

// ─── ToolHandler interface ───────────────────────────────────────────────────

// ToolName satisfies plugin.ToolHandler.
func (s *SubAgent) ToolName() string { return "subagent" }

// ToolDefs returns the run_in_subagent function definition.
func (s *SubAgent) ToolDefs(_ string) []ail.Instruction {
	return plugin.BuildToolDef(
		subagentToolName,
		"Branch out a heavy or specialised task to a separate LLM sub-agent. "+
			"The sub-agent runs a full inference call with the provided task and "+
			"returns a textual summary of its response. Use this when a task is "+
			"complex enough to benefit from dedicated reasoning in isolation.",
		subagentToolSchema,
	)
}

// HandleToolCall executes run_in_subagent by spawning a sub-inference call.
func (s *SubAgent) HandleToolCall(
	params string,
	callID string,
	args json.RawMessage,
	ctx *plugin.ToolCallContext,
) (string, bool, error) {
	var input struct {
		Task         string `json:"task"`
		SystemPrompt string `json:"system_prompt"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return "invalid arguments: " + err.Error(), true, nil
	}
	if input.Task == "" {
		return "task is required", true, nil
	}

	if ctx == nil || ctx.Infer == nil || ctx.Request == nil {
		return "subagent: inference context not available", true, nil
	}

	// ── Depth guard ──────────────────────────────────────────────────────
	maxDepth := parseMaxDepth(params)
	currentDepth := getDepth(ctx.Request)
	if currentDepth >= maxDepth {
		plugin.Logger.Warn("subagent: max depth reached",
			zap.Int("current", currentDepth),
			zap.Int("max", maxDepth))
		return fmt.Sprintf("Max sub-agent nesting depth (%d) reached. Please handle this task directly.", maxDepth), true, nil
	}

	// ── Build a fresh AIL program for the sub-inference ──────────────────
	subProg := ail.NewProgram()

	// Use the same model but strip +subagent to prevent recursion.
	model := stripSubagentSuffix(ctx.RequestProg.GetModel())
	subProg.EmitString(ail.SET_MODEL, model)

	// System prompt (optional).
	if input.SystemPrompt != "" {
		subProg.Emit(ail.MSG_START)
		subProg.Emit(ail.ROLE_SYS)
		subProg.EmitString(ail.TXT_CHUNK, input.SystemPrompt)
		subProg.Emit(ail.MSG_END)
	}

	// User message with the task.
	subProg.Emit(ail.MSG_START)
	subProg.Emit(ail.ROLE_USR)
	subProg.EmitString(ail.TXT_CHUNK, input.Task)
	subProg.Emit(ail.MSG_END)

	// Non-streaming for the sub-call (we need the full response to summarise).
	subProg.EmitString(ail.SET_STREAM, "false")

	plugin.Logger.Debug("subagent: dispatching sub-inference",
		zap.String("call_id", callID),
		zap.String("model", model),
		zap.Int("depth", currentDepth+1),
		zap.Int("task_len", len(input.Task)))

	// ── Invoke sub-inference ─────────────────────────────────────────────
	// Clone the request and bump the depth counter so nested subagent
	// calls are tracked. Use CaptureFresh since the model changed
	// (stripped +subagent suffix → needs fresh plugin resolution).
	subReq := ctx.Request.Clone(withDepth(ctx.Request.Context(), currentDepth+1))

	resProg, _, err := ctx.Infer.CaptureFresh(subProg, subReq)
	if err != nil {
		plugin.Logger.Error("subagent: sub-inference failed",
			zap.String("call_id", callID),
			zap.Error(err))
		return "sub-agent inference failed: " + err.Error(), true, nil
	}

	// ── Extract text from response ───────────────────────────────────────
	text := extractResponseText(resProg)
	if text == "" {
		return "(sub-agent returned an empty response)", true, nil
	}

	plugin.Logger.Debug("subagent: sub-inference complete",
		zap.String("call_id", callID),
		zap.Int("response_len", len(text)))

	return text, true, nil
}

// ─── Depth tracking via context ──────────────────────────────────────────────

type subagentDepthKey struct{}

func getDepth(r *http.Request) int {
	if v, ok := r.Context().Value(subagentDepthKey{}).(int); ok {
		return v
	}
	return 0
}

func withDepth(parent context.Context, depth int) context.Context {
	return context.WithValue(parent, subagentDepthKey{}, depth)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// stripSubagentSuffix removes "+subagent" (and any params) from the model
// so the sub-inference won't re-inject the tool.
func stripSubagentSuffix(model string) string {
	// Handle "+subagent:params" and "+subagent" at any position in the
	// plugin suffix chain. E.g. "openai/gpt-4+kvtools+subagent:3" →
	// "openai/gpt-4+kvtools".
	parts := strings.Split(model, "+")
	var kept []string
	for _, part := range parts {
		name := part
		if idx := strings.IndexByte(part, ':'); idx >= 0 {
			name = part[:idx]
		}
		if name != "subagent" {
			kept = append(kept, part)
		}
	}
	return strings.Join(kept, "+")
}

// parseMaxDepth extracts the max_depth param. Defaults to defaultMaxDepth.
func parseMaxDepth(params string) int {
	if params == "" {
		return defaultMaxDepth
	}
	// Support "max_depth=N" or bare "N".
	s := params
	if strings.HasPrefix(s, "max_depth=") {
		s = s[len("max_depth="):]
	}
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err == nil && n > 0 {
		return n
	}
	return defaultMaxDepth
}

// extractResponseText extracts all assistant text content from a response
// AIL program. Concatenates thinking blocks and text chunks.
func extractResponseText(prog *ail.Program) string {
	if prog == nil {
		return ""
	}
	var sb strings.Builder
	for _, msg := range prog.Messages() {
		if msg.Role != ail.ROLE_AST {
			continue
		}
		text := prog.MessageText(msg)
		if text != "" {
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(text)
		}
	}
	return sb.String()
}

// Compile-time checks.
var (
	_ plugin.BeforePlugin           = (*SubAgent)(nil)
	_ plugin.RecursiveHandlerPlugin = (*SubAgent)(nil)
)
