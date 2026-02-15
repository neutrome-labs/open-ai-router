package plugins

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

// SlidingWindow keeps a fixed-size window of messages.
//
// Syntax:
//
//	slwin          → keep 1 from start, 10 from end (defaults)
//	slwin:15       → keep 1 from start, 15 from end
//	slwin:15:3     → keep 3 from start, 15 from end
//
// Messages outside the window are removed entirely. Non-message
// instructions (SET_MODEL, tool definitions, etc.) are always preserved.
type SlidingWindow struct{}

func (f *SlidingWindow) Name() string { return "slwin" }

func (f *SlidingWindow) Before(params string, _ *services.ProviderService, _ *http.Request, prog *ail.Program) (*ail.Program, error) {
	keepEnd, keepStart := 10, 1
	if params != "" {
		parts := strings.SplitN(params, ":", 2)
		if v, err := strconv.Atoi(parts[0]); err == nil && v > 0 {
			keepEnd = v
		}
		if len(parts) == 2 {
			if v, err := strconv.Atoi(parts[1]); err == nil && v >= 0 {
				keepStart = v
			}
		}
	}

	// 1. Find all message spans (MSG_START..MSG_END).
	type msgSpan struct {
		start int
		end   int // inclusive
	}
	var msgs []msgSpan
	for i := 0; i < len(prog.Code); i++ {
		if prog.Code[i].Op == ail.MSG_START {
			for j := i; j < len(prog.Code); j++ {
				if prog.Code[j].Op == ail.MSG_END {
					msgs = append(msgs, msgSpan{start: i, end: j})
					i = j
					break
				}
			}
		}
	}

	total := len(msgs)
	if total <= keepStart+keepEnd {
		// Everything fits in the window — nothing to drop.
		return prog, nil
	}

	// 2. Determine which messages to keep.
	keepSet := make(map[int]bool, keepStart+keepEnd)
	for i := 0; i < keepStart && i < total; i++ {
		keepSet[i] = true
	}
	for i := total - keepEnd; i < total; i++ {
		if i >= 0 {
			keepSet[i] = true
		}
	}

	// 3. Build a set of instruction indices to drop.
	drop := make(map[int]bool)
	for mi, m := range msgs {
		if !keepSet[mi] {
			for idx := m.start; idx <= m.end; idx++ {
				drop[idx] = true
			}
		}
	}

	// 4. Rebuild the program, copying only non-dropped instructions.
	out := ail.NewProgram()
	out.Buffers = prog.Buffers
	for i, inst := range prog.Code {
		if !drop[i] {
			out.Code = append(out.Code, inst)
		}
	}

	return out, nil
}

var _ plugin.BeforePlugin = (*SlidingWindow)(nil)
