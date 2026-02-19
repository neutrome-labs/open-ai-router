package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"github.com/neutrome-labs/open-ai-router/src/services/kv"
)

// KvTools strips completed tool-call interactions from the conversation
// (like stools) but caches each tool result into a KV store before
// removing it. It injects a synthetic "get_tool_result" tool definition
// so the AI model can recall any previous tool result on demand.
//
// Architecture:
//   - Embeds plugin.ToolPlugin which provides BeforePlugin (def injection)
//     and RecursiveHandlerPlugin (tool-call dispatch loop) automatically.
//   - Additionally implements BeforePlugin itself for the cache-and-strip
//     logic that runs before tool defs are injected.
//
// Syntax:
//
//	kvtools         → defaults (memory backend, 30m TTL)
//	kvtools:redis   → use redis backend
type KvTools struct {
	plugin.ToolPlugin // BeforePlugin (def injection) + RecursiveHandlerPlugin (dispatch loop)
	store             kv.Store
}

// NewKvTools creates a KvTools plugin wired to its ToolPlugin base.
func NewKvTools() *KvTools {
	k := &KvTools{}
	k.ToolPlugin = *plugin.NewToolPlugin(k)
	return k
}

const kvToolName = "get_tool_result"

var kvToolSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"tool_call_id": {
			"type": "string",
			"description": "The ID of a previous tool call whose result you want to retrieve."
		}
	},
	"required": ["tool_call_id"]
}`)

// ─── ToolHandler interface ───────────────────────────────────────────────────

// ToolName satisfies plugin.ToolHandler — also used as Plugin.Name().
func (k *KvTools) ToolName() string { return "kvtools" }

// ToolDefs returns the get_tool_result function definition —
// satisfies plugin.ToolHandler. The ToolPlugin base calls this in its
// own BeforePlugin.Before to inject the def.
func (k *KvTools) ToolDefs(_ string) []ail.Instruction {
	return plugin.BuildToolDef(
		kvToolName,
		"Retrieve the result of a previous tool call by its ID. Use this when you need data from a tool call that was made earlier in the conversation but whose result is no longer in context.",
		kvToolSchema,
	)
}

// HandleToolCall serves get_tool_result by looking up the call ID in KV —
// satisfies plugin.ToolHandler.
func (k *KvTools) HandleToolCall(params string, callID string, args json.RawMessage, ctx *plugin.ToolCallContext) (string, bool, error) {
	var input struct {
		ToolCallID string `json:"tool_call_id"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return "invalid arguments: " + err.Error(), true, nil
	}
	if input.ToolCallID == "" {
		return "tool_call_id is required", true, nil
	}

	store := k.ensureStore(params)
	traceID := ""
	if ctx != nil {
		traceID = ctx.TraceID
	}
	val, err := store.Get(context.Background(), kvKey(traceID, input.ToolCallID))
	if err != nil {
		return "tool result not found for call_id: " + input.ToolCallID, true, nil
	}
	return val, true, nil
}

// ─── BeforePlugin: cache & strip ─────────────────────────────────────────────
//
// KvTools overrides the ToolPlugin's Before to add cache-and-strip logic
// BEFORE the tool defs are injected. The call chain is:
//   - chain.RunBefore → KvTools.Before (cache & strip + delegate to ToolPlugin.Before)

func (k *KvTools) Before(params string, p *services.ProviderService, r *http.Request, prog *ail.Program) (*ail.Program, error) {
	// First: cache older tool results and strip them.
	prog, err := k.cacheAndStrip(params, r, prog)
	if err != nil {
		return nil, err
	}
	// Then: delegate to ToolPlugin.Before to inject tool defs.
	return k.ToolPlugin.Before(params, p, r, prog)
}

// cacheAndStrip caches tool results from completed interactions, strips
// them from the conversation, and prepends a note about cached call IDs.
func (k *KvTools) cacheAndStrip(params string, r *http.Request, prog *ail.Program) (*ail.Program, error) {
	store := k.ensureStore(params)
	msgs := prog.Messages()

	// Identify tool interactions: assistant msg with calls + following tool results.
	type interaction struct {
		assistIdx int
		endIdx    int
	}
	var interactions []interaction
	for i := 0; i < len(msgs); i++ {
		if msgs[i].Role == ail.ROLE_AST && spanHasCalls(prog, msgs[i]) {
			ti := interaction{assistIdx: i, endIdx: i}
			for j := i + 1; j < len(msgs) && msgs[j].Role == ail.ROLE_TOOL; j++ {
				ti.endIdx = j
			}
			interactions = append(interactions, ti)
			i = ti.endIdx
		}
	}

	if len(interactions) <= 1 {
		return prog, nil
	}

	// Extract trace ID for KV key scoping.
	traceID := ""
	if v := r.Context().Value(plugin.ContextTraceID()); v != nil {
		traceID, _ = v.(string)
	}

	toCache := interactions[:len(interactions)-1]

	// Cache all tool results from older interactions.
	for _, ti := range toCache {
		for j := ti.assistIdx + 1; j <= ti.endIdx; j++ {
			if msgs[j].Role != ail.ROLE_TOOL {
				continue
			}
			results := prog.ToolResults()
			for _, res := range results {
				if res.Start >= msgs[j].Start && res.End <= msgs[j].End {
					for idx := res.Start; idx <= res.End && idx < len(prog.Code); idx++ {
						if prog.Code[idx].Op == ail.RESULT_DATA {
							_ = store.Set(
								context.Background(),
								kvKey(traceID, res.CallID),
								prog.Code[idx].Str,
								30*time.Minute,
							)
							break
						}
					}
				}
			}
		}
	}

	// Collect cached call IDs for the note.
	var cachedIDs []string
	for _, ti := range toCache {
		calls := prog.ToolCalls()
		for _, c := range calls {
			if c.Start >= msgs[ti.assistIdx].Start && c.End <= msgs[ti.assistIdx].End {
				cachedIDs = append(cachedIDs, c.CallID)
			}
		}
	}

	// Strip older interactions.
	var toRemove []ail.MessageSpan
	for _, ti := range toCache {
		for j := ti.assistIdx; j <= ti.endIdx; j++ {
			toRemove = append(toRemove, msgs[j])
		}
	}
	result := prog.RemoveMessages(toRemove...)

	// Prepend context about available recalls.
	if len(cachedIDs) > 0 {
		note := "Previous tool call results have been cached and removed from context to save tokens. " +
			"You can retrieve any of them using get_tool_result with these call IDs: " +
			strings.Join(cachedIDs, ", ")
		result = result.PrependSystemPrompt(note)
	}

	return result, nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

func (k *KvTools) ensureStore(params string) kv.Store {
	if k.store != nil {
		return k.store
	}
	backend := "memory"
	dsn := ""
	if params != "" {
		parts := strings.SplitN(params, "=", 2)
		backend = parts[0]
		if len(parts) == 2 {
			dsn = parts[1]
		}
	}
	s, err := kv.Open(backend, dsn)
	if err != nil {
		s, _ = kv.Open("memory", "")
	}
	k.store = s
	return k.store
}

func kvKey(traceID, callID string) string {
	if traceID != "" {
		return "kvtools:" + traceID + ":" + callID
	}
	return "kvtools::" + callID
}

// Compile-time checks: KvTools satisfies BeforePlugin (overriding the embedded one)
// and inherits RecursiveHandlerPlugin from ToolPlugin.
var (
	_ plugin.BeforePlugin           = (*KvTools)(nil)
	_ plugin.RecursiveHandlerPlugin = (*KvTools)(nil)
)
