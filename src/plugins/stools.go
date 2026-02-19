package plugins

import (
	"net/http"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

// StripTools removes completed tool-call interactions from the message
// history, keeping only the last interaction sequence. For earlier
// sequences the assistant message text (if any) is preserved but the
// tool_calls array and corresponding tool-result messages are dropped.
//
// This reduces token usage when the model has already summarised the
// tool output in a subsequent text reply.
type StripTools struct{}

func (f *StripTools) Name() string { return "stools" }

// toolInteraction records a group: one assistant msg with tool calls,
// followed by consecutive tool-result messages.
type toolInteraction struct {
	assistIdx int // index into msgs slice
	endIdx    int // index of last tool-result msg in msgs slice (inclusive)
}

// spanHasCalls reports whether any instruction in [span.Start, span.End]
// is a CALL_START opcode.
func spanHasCalls(prog *ail.Program, span ail.MessageSpan) bool {
	for i := span.Start; i <= span.End && i < len(prog.Code); i++ {
		if prog.Code[i].Op == ail.CALL_START {
			return true
		}
	}
	return false
}

func (f *StripTools) Before(_ string, _ *services.ProviderService, _ *http.Request, prog *ail.Program) (*ail.Program, error) {
	msgs := prog.Messages()

	// Group into tool interactions: assistant msg with calls + following tool results.
	var interactions []toolInteraction
	for i := 0; i < len(msgs); i++ {
		if msgs[i].Role == ail.ROLE_AST && spanHasCalls(prog, msgs[i]) {
			ti := toolInteraction{assistIdx: i, endIdx: i}
			for j := i + 1; j < len(msgs) && msgs[j].Role == ail.ROLE_TOOL; j++ {
				ti.endIdx = j
			}
			interactions = append(interactions, ti)
			i = ti.endIdx
		}
	}

	// 0-1 interactions → nothing to strip.
	if len(interactions) <= 1 {
		return prog, nil
	}

	// Determine what to do with each message in older interactions.
	type action struct {
		drop     bool
		textOnly bool // emit MSG_START + ROLE + text chunks + MSG_END only
	}
	msgAction := make(map[int]action) // keyed by message index in msgs

	for _, ti := range interactions[:len(interactions)-1] {
		// The assistant message: keep as text-only if it has content.
		if prog.MessageText(msgs[ti.assistIdx]) != "" {
			msgAction[ti.assistIdx] = action{drop: true, textOnly: true}
		} else {
			msgAction[ti.assistIdx] = action{drop: true}
		}
		// Tool-result messages: fully drop.
		for j := ti.assistIdx + 1; j <= ti.endIdx; j++ {
			msgAction[j] = action{drop: true}
		}
	}

	// Rebuild the program using message spans from ailmanip.
	out := ail.NewProgram()
	out.Buffers = prog.Buffers

	// Index message spans by their start instruction for the rebuild loop.
	msgByStart := make(map[int]int, len(msgs)) // instruction index → msgs slice index
	for idx, m := range msgs {
		msgByStart[m.Start] = idx
	}

	i := 0
	for i < len(prog.Code) {
		msgIdx, isMsgStart := msgByStart[i]
		if !isMsgStart {
			out.Code = append(out.Code, prog.Code[i])
			i++
			continue
		}

		m := msgs[msgIdx]
		act, hasAction := msgAction[msgIdx]

		if !hasAction {
			// Keep the entire message as-is.
			for j := m.Start; j <= m.End; j++ {
				out.Code = append(out.Code, prog.Code[j])
			}
		} else if act.textOnly {
			// Re-emit as text-only (strip CALL_START..CALL_END blocks).
			out.Code = append(out.Code, ail.Instruction{Op: ail.MSG_START})
			inCall := false
			for j := m.Start + 1; j < m.End; j++ {
				op := prog.Code[j].Op
				if op == ail.CALL_START {
					inCall = true
					continue
				}
				if op == ail.CALL_END {
					inCall = false
					continue
				}
				if inCall {
					continue
				}
				out.Code = append(out.Code, prog.Code[j])
			}
			out.Code = append(out.Code, ail.Instruction{Op: ail.MSG_END})
		}
		// else: fully dropped, emit nothing.

		i = m.End + 1
	}

	return out, nil
}

var _ plugin.BeforePlugin = (*StripTools)(nil)
