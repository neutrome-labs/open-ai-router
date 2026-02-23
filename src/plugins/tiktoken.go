package plugins

import (
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
	tiktoken "github.com/pkoukk/tiktoken-go"
	"go.uber.org/zap"
)

// Tiktoken counts tokens on request arrival (OnRequestInit) and after
// before-plugins have run (Before). It logs the counts and the effective
// compression percentage, and emits a SET_META "x-token-diff" entry into
// the AIL program so the server module can surface it as a response header.
//
// It runs as a TailPlugin alongside the Sampler so it sees the final
// upstream-prepared program in Before.
type Tiktoken struct {
	// counts maps traceID → incoming token count for the current request.
	counts sync.Map
}

func NewTiktoken() *Tiktoken { return &Tiktoken{} }

func (t *Tiktoken) Name() string { return "tiktoken" }

// OnRequestInit counts tokens in the original parsed program.
func (t *Tiktoken) OnRequestInit(r *http.Request, prog *ail.Program) {
	traceID, _ := r.Context().Value(plugin.ContextTraceID()).(string)
	if traceID == "" {
		return
	}

	model := prog.GetModel()
	count := countTokens(model, prog)
	t.counts.Store(traceID, count)

	Logger.Debug("TIKTOKEN: incoming",
		zap.String("trace_id", traceID),
		zap.String("model", model),
		zap.Int("tokens", count),
	)
}

// Before counts tokens in the upstream-prepared program, computes the
// compression ratio vs. the incoming count, logs it, and emits the diff
// as SET_META "x-token-diff" into the program.
func (t *Tiktoken) Before(_ string, _ *services.ProviderService, r *http.Request, prog *ail.Program) (*ail.Program, error) {
	traceID, _ := r.Context().Value(plugin.ContextTraceID()).(string)
	incomingVal, ok := t.counts.Load(traceID)
	if !ok {
		return prog, nil
	}
	incoming := incomingVal.(int)
	defer t.counts.Delete(traceID)

	model := prog.GetModel()
	upstream := countTokens(model, prog)

	var pct float64
	if incoming > 0 {
		pct = float64(incoming-upstream) / float64(incoming) * 100
	}

	diff := fmt.Sprintf("%d -> %d = %.1f%%", incoming, upstream, pct)

	Logger.Info("TIKTOKEN: diff",
		zap.String("trace_id", traceID),
		zap.String("model", model),
		zap.Int("incoming", incoming),
		zap.Int("upstream", upstream),
		zap.String("compression", fmt.Sprintf("%.1f%%", pct)),
	)

	// Emit into the program so the server module can read it via Config().
	result := prog.Clone()
	result.EmitKeyVal(ail.SET_META, "x-token-diff", diff)
	return result, nil
}

// ─── Token counting ──────────────────────────────────────────────────────────

// countTokens extracts all textual content from the AIL program and
// returns the total token count using tiktoken for the given model.
func countTokens(model string, prog *ail.Program) int {
	// Collect all text that would be sent to the model.
	var sb strings.Builder

	for _, msg := range prog.Messages() {
		// Message text content.
		if text := prog.MessageText(msg); text != "" {
			sb.WriteString(text)
			sb.WriteByte('\n')
		}

		// Thinking/reasoning content within the message.
		for i := msg.Start; i <= msg.End && i < len(prog.Code); i++ {
			if prog.Code[i].Op == ail.THINK_CHUNK {
				sb.WriteString(prog.Code[i].Str)
				sb.WriteByte('\n')
			}
		}
	}

	// Tool definitions (name + description + schema text).
	for _, td := range prog.ToolDefs() {
		sb.WriteString(td.Name)
		sb.WriteByte('\n')
		for i := td.Start; i <= td.End && i < len(prog.Code); i++ {
			switch prog.Code[i].Op {
			case ail.DEF_DESC:
				sb.WriteString(prog.Code[i].Str)
				sb.WriteByte('\n')
			case ail.DEF_SCHEMA:
				sb.Write(prog.Code[i].JSON)
				sb.WriteByte('\n')
			}
		}
	}

	// Tool call arguments.
	for _, tc := range prog.ToolCalls() {
		sb.WriteString(tc.Name)
		sb.WriteByte('\n')
		for i := tc.Start; i <= tc.End && i < len(prog.Code); i++ {
			if prog.Code[i].Op == ail.CALL_ARGS {
				sb.Write(prog.Code[i].JSON)
				sb.WriteByte('\n')
			}
		}
	}

	// Tool result data.
	for _, res := range prog.ToolResults() {
		for i := res.Start; i <= res.End && i < len(prog.Code); i++ {
			if prog.Code[i].Op == ail.RESULT_DATA {
				sb.WriteString(prog.Code[i].Str)
				sb.WriteByte('\n')
			}
		}
	}

	// System prompt (already included via Messages, but Config metadata
	// like SET_META values could carry text too — skip for now as those
	// are not sent to the model as tokens).

	text := sb.String()
	if text == "" {
		return 0
	}

	enc := getEncoding(model)
	tokens := enc.Encode(text, nil, nil)
	return len(tokens)
}

// ─── Encoding cache ──────────────────────────────────────────────────────────

var (
	encodingCache sync.Map // model → *tiktoken.Tiktoken
	fallbackEnc   *tiktoken.Tiktoken
	fallbackOnce  sync.Once
)

// getEncoding returns a tiktoken encoder for the model, falling back to
// cl100k_base (GPT-4/GPT-3.5 family) for unknown models.
func getEncoding(model string) *tiktoken.Tiktoken {
	// Strip provider prefix (e.g. "openai/gpt-4o" → "gpt-4o").
	if i := strings.Index(model, "/"); i >= 0 {
		model = model[i+1:]
	}
	// Strip plugin suffixes (e.g. "gpt-4o+slwin+kvtools" → "gpt-4o").
	if i := strings.IndexByte(model, '+'); i >= 0 {
		model = model[:i]
	}

	if cached, ok := encodingCache.Load(model); ok {
		return cached.(*tiktoken.Tiktoken)
	}

	enc, err := tiktoken.EncodingForModel(model)
	if err == nil {
		encodingCache.Store(model, enc)
		return enc
	}

	// Fallback for non-OpenAI models.
	fallbackOnce.Do(func() {
		fallbackEnc, _ = tiktoken.GetEncoding("cl100k_base")
	})
	return fallbackEnc
}

// Compile-time checks.
var (
	_ plugin.RequestInitPlugin = (*Tiktoken)(nil)
	_ plugin.BeforePlugin      = (*Tiktoken)(nil)
)
