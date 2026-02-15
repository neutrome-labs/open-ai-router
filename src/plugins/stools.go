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

// ailMsg records the instruction-index span and metadata of a single
// message (MSG_START..MSG_END) inside the AIL program.
type ailMsg struct {
	start    int
	end      int // inclusive: index of MSG_END
	role     ail.Opcode
	hasCalls bool
	hasText  bool
}

// toolInteraction records a group: one assistant msg with tool calls,
// followed by consecutive tool-result messages.
type toolInteraction struct {
	assistIdx int // index into msgs slice
	endIdx    int // index of last tool-result msg in msgs slice (inclusive)
}

func (f *StripTools) Before(_ string, _ *services.ProviderService, _ *http.Request, prog *ail.Program) (*ail.Program, error) {
	// 1. Parse all messages into spans.
	var msgs []ailMsg
	cur := ailMsg{start: -1}
	for i, inst := range prog.Code {
		switch inst.Op {
		case ail.MSG_START:
			cur = ailMsg{start: i}
		case ail.ROLE_SYS, ail.ROLE_USR, ail.ROLE_AST, ail.ROLE_TOOL:
			cur.role = inst.Op
		case ail.CALL_START:
			cur.hasCalls = true
		case ail.TXT_CHUNK:
			cur.hasText = true
		case ail.MSG_END:
			cur.end = i
			msgs = append(msgs, cur)
			cur = ailMsg{start: -1}
		}
	}

	// 2. Group into tool interactions.
	var interactions []toolInteraction
	for i := 0; i < len(msgs); i++ {
		if msgs[i].role == ail.ROLE_AST && msgs[i].hasCalls {
			ti := toolInteraction{assistIdx: i, endIdx: i}
			for j := i + 1; j < len(msgs) && msgs[j].role == ail.ROLE_TOOL; j++ {
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

	// 3. Build a set of instruction-index ranges to drop.
	//    For stripped assistant msgs with text, we'll re-emit text-only.
	type action struct {
		drop     bool
		textOnly bool // emit MSG_START + ROLE + text chunks + MSG_END only
	}
	msgAction := make(map[int]action) // keyed by message index in msgs

	for _, ti := range interactions[:len(interactions)-1] {
		// The assistant message: keep as text-only if it has content.
		if msgs[ti.assistIdx].hasText {
			msgAction[ti.assistIdx] = action{drop: true, textOnly: true}
		} else {
			msgAction[ti.assistIdx] = action{drop: true}
		}
		// Tool-result messages: fully drop.
		for j := ti.assistIdx + 1; j <= ti.endIdx; j++ {
			msgAction[j] = action{drop: true}
		}
	}

	// 4. Rebuild the program.
	out := ail.NewProgram()
	out.Buffers = prog.Buffers

	msgIdx := 0
	i := 0
	for i < len(prog.Code) {
		inst := prog.Code[i]

		if inst.Op == ail.MSG_START && msgIdx < len(msgs) {
			act, hasAction := msgAction[msgIdx]
			m := msgs[msgIdx]
			msgIdx++

			if !hasAction {
				// Keep the entire message as-is.
				for j := m.start; j <= m.end; j++ {
					out.Code = append(out.Code, prog.Code[j])
				}
				i = m.end + 1
				continue
			}

			if act.textOnly {
				// Re-emit as text-only (strip CALL_START..CALL_END blocks).
				out.Code = append(out.Code, ail.Instruction{Op: ail.MSG_START})
				inCall := false
				for j := m.start + 1; j < m.end; j++ {
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

			i = m.end + 1
			continue
		}

		out.Code = append(out.Code, inst)
		i++
	}

	return out, nil
}

var _ plugin.BeforePlugin = (*StripTools)(nil)
