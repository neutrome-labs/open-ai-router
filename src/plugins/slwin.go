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

	msgs := prog.Messages()
	total := len(msgs)
	if total <= keepStart+keepEnd {
		return prog, nil
	}

	// Collect message spans that fall outside both windows.
	keepSet := make(map[int]bool, keepStart+keepEnd)
	for i := 0; i < keepStart && i < total; i++ {
		keepSet[i] = true
	}
	for i := total - keepEnd; i < total; i++ {
		if i >= 0 {
			keepSet[i] = true
		}
	}

	var toRemove []ail.MessageSpan
	for i, m := range msgs {
		if !keepSet[i] {
			toRemove = append(toRemove, m)
		}
	}

	return prog.RemoveMessages(toRemove...), nil
}

var _ plugin.BeforePlugin = (*SlidingWindow)(nil)
