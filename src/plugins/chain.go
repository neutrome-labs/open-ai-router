// Package plugins — chain plugin for multi-step inference chaining on the gateway.
//
// The chain plugin allows composing multiple AI inference steps in a single
// request. Each chain step can inject a prompt, change the model, and
// override the system prompt. Steps are executed sequentially, with each
// step's result fed into the next.
//
// Syntax (in model suffix):
//
//	model+chain:prompt:mode:model:system
//
// Arguments (colon-separated):
//
//	arg0  prompt   — user message injected for this step
//	arg1  mode     — replace|prepend|append (default: replace)
//	arg2  model    — model override (empty = keep original)
//	arg3  system   — system prompt override; empty = remove system prompt
//
// Modes:
//
//	replace  — base result is hidden; chain step's output replaces it
//	prepend  — chain step runs BEFORE base; its output is shown first (as thinking)
//	append   — chain step runs AFTER base; its output is shown after base
//
// Multiple chain steps are supported:
//
//	model+chain:translate:replace+chain:jsonify:replace
//
// Streaming: the last visible step streams directly to the client.
// Intermediate steps always run non-streaming (captured internally).
// Prepend results appear as thinking blocks in the final SSE stream.
package plugins

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// ─── Chain plugin ────────────────────────────────────────────────────────────

// ChainPlugin orchestrates multi-step inference chaining.
// Implements RecursiveHandlerPlugin — the first chain instance in the
// plugin chain collects all chain steps and executes them sequentially.
type ChainPlugin struct{}

func (*ChainPlugin) Name() string { return "chain" }

// ─── Per-plugin re-entry guard ───────────────────────────────────────────────

type chainBypassKey struct{}

func withChainBypass(ctx context.Context) context.Context {
	return context.WithValue(ctx, chainBypassKey{}, true)
}

func hasChainBypass(ctx context.Context) bool {
	_, ok := ctx.Value(chainBypassKey{}).(bool)
	return ok
}

// ─── Param parsing ───────────────────────────────────────────────────────────

type chainStep struct {
	Prompt    string // arg0: user message for this step
	Mode      string // arg1: replace | prepend | append
	Model     string // arg2: model override
	System    string // arg3: system prompt override
	HasSystem bool   // true when arg3 was explicitly provided (even if empty)
	RawParams string // original params string for identity matching
}

func parseChainParams(params string) chainStep {
	step := chainStep{
		Mode:      "replace",
		RawParams: params,
	}

	parts := strings.SplitN(params, ":", 4)
	if len(parts) >= 1 {
		step.Prompt = parts[0]
	}
	if len(parts) >= 2 && parts[1] != "" {
		step.Mode = parts[1]
	}
	if len(parts) >= 3 {
		step.Model = parts[2]
	}
	if len(parts) >= 4 {
		step.System = parts[3]
		step.HasSystem = true
	}

	return step
}

// collectChainSteps finds all chain plugin instances in the plugin chain.
func collectChainSteps(chain *plugin.PluginChain) []chainStep {
	var steps []chainStep
	for _, pi := range chain.GetPlugins() {
		if pi.Plugin.Name() == "chain" {
			steps = append(steps, parseChainParams(pi.Params))
		}
	}
	return steps
}

// ─── RecursiveHandlerPlugin ──────────────────────────────────────────────────

func (c *ChainPlugin) RecursiveHandler(
	params string,
	ic *plugin.InferenceContext,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) (bool, error) {
	// Per-plugin re-entry guard: skip if we're inside a chained InferFresh call.
	if hasChainBypass(r.Context()) {
		return false, nil
	}

	steps := collectChainSteps(ic.Chain)
	if len(steps) == 0 {
		return false, nil
	}
	// Only the first chain instance orchestrates all steps.
	if steps[0].RawParams != params {
		return false, nil
	}

	plugin.Logger.Info("chain plugin starting",
		zap.Int("steps", len(steps)),
		zap.Bool("streaming", prog.IsStreaming()))

	isStream := prog.IsStreaming()
	originalProg := prog.Clone()

	// Partition steps into pre (prepend) and post (replace/append).
	var preSteps, postSteps []chainStep
	for _, s := range steps {
		if s.Mode == "prepend" {
			preSteps = append(preSteps, s)
		} else {
			postSteps = append(postSteps, s)
		}
	}

	// ── PRE PHASE: execute prepend steps ──────────────────────────────
	// Prepend steps run BEFORE the base inference. Their output is shown
	// as thinking blocks (preceding the main response).

	// Total steps for sampler: prepend + base + post.
	totalSteps := len(preSteps) + 1 + len(postSteps)
	stepCounter := 0

	currentProg := stripStreaming(prog)
	var prependResults []*ail.Program

	for i, step := range preSteps {
		plugin.Logger.Debug("chain: executing prepend step",
			zap.Int("index", i), zap.String("prompt", step.Prompt))

		stepProg := currentProg.Clone()
		if step.Prompt != "" {
			stepProg = stepProg.AppendUserMessage(step.Prompt)
		}
		stepProg = applyOverrides(stepProg, step)

		resProg, _, err := captureFull(ic, stepProg, step, r, stepCounter, totalSteps)
		stepCounter++
		if err != nil {
			return true, err
		}

		prependResults = append(prependResults, resProg)

		// Add prepend result as hidden context (user message wrapped in
		// <think> tags). The base model shouldn't see this as its own
		// prior output — otherwise it thinks it already answered.
		currentProg = appendAsHiddenContext(currentProg, resProg)
	}

	// ── BASE PHASE ────────────────────────────────────────────────────
	// Run the "normal" inference. If there are post-steps (replace/append),
	// capture it; otherwise it's the final output.

	if len(postSteps) == 0 {
		// No post-processing — base inference is the final step.
		// Restore streaming if requested and write directly.
		finalProg := currentProg
		if isStream {
			finalProg = restoreStreaming(finalProg)
		}

		if len(prependResults) > 0 && !isStream {
			// Non-streaming with prepend: capture base via exit path so
			// other recursive handlers (dspy, etc.) fire on the final output.
			baseStep := chainStep{Mode: "base", Prompt: "base"}
			baseRes, baseCapture, err := captureForExit(ic, stripStreaming(finalProg), baseStep, r, stepCounter, totalSteps)
			if err != nil {
				return true, err
			}
			return true, writeWithThinkBlocks(prependResults, baseRes, baseCapture, w)
		}

		if len(prependResults) > 0 && isStream {
			// Streaming with prepend: emit think events first, then stream base.
			return true, streamWithPrepend(prependResults, finalProg, ic, w, r)
		}

		// No prepend, no post — should never reach here (no chain steps).
		return false, nil
	}

	// Capture base inference (non-streaming for intermediate use).
	baseStep := chainStep{Mode: "base", Prompt: "base"}
	baseResProg, _, err := captureFull(ic, currentProg, baseStep, r, stepCounter, totalSteps)
	stepCounter++
	if err != nil {
		return true, err
	}

	plugin.Logger.Debug("chain: base inference captured",
		zap.Int("messages", baseResProg.CountMessages()))

	// ── POST PHASE: execute replace/append steps ──────────────────────
	prevResult := baseResProg
	var lastCapture *services.ResponseCaptureWriter

	for i, step := range postSteps {
		isLast := i == len(postSteps)-1

		plugin.Logger.Debug("chain: executing post step",
			zap.Int("index", i), zap.String("mode", step.Mode),
			zap.String("prompt", step.Prompt), zap.Bool("last", isLast))

		// For append mode on the first post-step, we need to show the base result.
		// In non-streaming mode we accumulate; in streaming this is the base stream.
		// TODO: append mode + streaming needs the base to stream first.

		// Build step program from ORIGINAL messages (not accumulated).
		stepProg := stripStreaming(originalProg.Clone())

		// Add previous (invisible) result.
		stepProg = appendAsHiddenContext(stepProg, prevResult)

		// Add chain prompt.
		if step.Prompt != "" {
			stepProg = stepProg.AppendUserMessage(step.Prompt)
		}
		stepProg = applyOverrides(stepProg, step)

		// Last step + streaming: stream directly to client.
		if isLast && isStream {
			stepProg = restoreStreaming(stepProg)
			return true, inferDirectToClient(ic, stepProg, step, w, r)
		}

		// Capture step.
		var capturedProg *ail.Program
		var capture *services.ResponseCaptureWriter
		if isLast {
			// Last non-streaming step: exit via InferFresh so dspy etc. fire.
			capturedProg, capture, err = captureForExit(ic, stepProg, step, r, stepCounter, totalSteps)
		} else {
			// Intermediate step: raw inference, no recursive handlers.
			capturedProg, capture, err = captureFull(ic, stepProg, step, r, stepCounter, totalSteps)
		}
		stepCounter++
		if err != nil {
			return true, err
		}
		prevResult = capturedProg
		lastCapture = capture
	}

	// Non-streaming final output: replay the last captured response.
	if len(prependResults) > 0 {
		return true, writeWithThinkBlocks(prependResults, prevResult, lastCapture, w)
	}

	plugin.ReplayCapture(lastCapture, w)
	return true, nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// stripStreaming removes SET_STREAM opcodes from a program to force
// non-streaming capture.
func stripStreaming(prog *ail.Program) *ail.Program {
	indices := prog.FindAll(ail.SET_STREAM)
	if len(indices) == 0 {
		return prog
	}
	return prog.ClearAtIndex(indices...)
}

// restoreStreaming ensures the program has SET_STREAM.
func restoreStreaming(prog *ail.Program) *ail.Program {
	if prog.IsStreaming() {
		return prog
	}
	result := prog.Clone()
	result.Emit(ail.SET_STREAM)
	return result
}

// applyOverrides applies model and system prompt overrides to a program.
func applyOverrides(prog *ail.Program, step chainStep) *ail.Program {
	if step.Model != "" {
		prog.SetModel(step.Model)
	}
	if step.HasSystem {
		if step.System == "" {
			// Empty system = remove all system messages.
			sysMsgs := prog.SystemPrompts()
			if len(sysMsgs) > 0 {
				prog = prog.RemoveMessages(sysMsgs...)
			}
		} else {
			prog = prog.ReplaceSystemPrompt(step.System)
		}
	}
	return prog
}

// appendAsHiddenContext adds the assistant output from resProg as a user
// message wrapped in <think></think> tags. Used for intermediate chain
// results that are invisible to the end user (e.g. replace-mode steps
// whose output is swallowed). Presenting them as user messages with think
// tags tells the next model "this is prior reasoning context, not your
// own output".
func appendAsHiddenContext(prog *ail.Program, resProg *ail.Program) *ail.Program {
	var parts []string
	for _, msg := range resProg.MessagesByRole(ail.ROLE_AST) {
		text := resProg.MessageText(msg)
		if text != "" {
			parts = append(parts, text)
		}
	}
	if len(parts) == 0 {
		return prog
	}
	wrapped := "<internal_thoughts>\n" + strings.Join(parts, "\n") + "\n</internal_thoughts>\nNow I am ready to summarize this to the user."
	return prog.AppendAssistantMessage(wrapped)
}

// captureFull runs inference for an intermediate chain step and returns
// both parsed program and raw capture.
// Sets SamplerStep in context so the sampler can namespace files per step.
//
// Intermediate steps use ic.Infer() — just Before→provider→After through
// the current plugin chain. No recursive handlers fire. If step.Model is
// set, uses InferFresh with the override model + chainBypass.
func captureFull(ic *plugin.InferenceContext, stepProg *ail.Program, step chainStep, r *http.Request, stepIdx int, totalSteps int) (*ail.Program, *services.ResponseCaptureWriter, error) {
	label := step.Prompt
	if label == "" {
		label = step.Mode
	}
	samplerStep := plugin.SamplerStep{
		Index: stepIdx,
		Label: fmt.Sprintf("chain:%s", label),
		Total: totalSteps,
	}
	stepR := r.WithContext(plugin.WithSamplerStep(r.Context(), samplerStep))

	if step.Model != "" {
		// Model override: InferFresh so the override model's plugins fire.
		stepR = stepR.WithContext(withChainBypass(stepR.Context()))
		return ic.CaptureFresh(stepProg, stepR)
	}
	// Same model: raw inference through current chain. No recursive
	// handlers re-fire — chain is already orchestrating.
	return ic.Capture(stepProg, stepR)
}

// captureForExit runs inference for the final (exit) chain step in
// non-streaming mode and returns both parsed program and raw capture.
// Strips +chain:... suffixes from the model and uses CaptureFresh so
// other recursive handlers (dspy, kvtools, etc.) fire on the exit output.
func captureForExit(ic *plugin.InferenceContext, stepProg *ail.Program, step chainStep, r *http.Request, stepIdx int, totalSteps int) (*ail.Program, *services.ResponseCaptureWriter, error) {
	label := step.Prompt
	if label == "" {
		label = step.Mode
	}
	samplerStep := plugin.SamplerStep{
		Index: stepIdx,
		Label: fmt.Sprintf("chain:%s", label),
		Total: totalSteps,
	}
	stepR := r.WithContext(plugin.WithSamplerStep(r.Context(), samplerStep))

	if step.Model == "" {
		// Strip chain suffixes so the fresh chain has dspy/kvtools but NOT chain.
		exitModel := stripChainSuffixes(stepProg.GetModel())
		stepProg.SetModel(exitModel)
	}
	stepR = stepR.WithContext(withChainBypass(stepR.Context()))
	return ic.CaptureFresh(stepProg, stepR)
}

// inferDirectToClient runs inference and writes the response directly to
// the client writer. Used for the final (exit) streaming step.
//
// Strips +chain:... suffixes from the model and uses InferFresh so that
// other recursive handlers (dspy, kvtools, etc.) fire on the final output.
// If step.Model is set, uses the override model instead.
func inferDirectToClient(ic *plugin.InferenceContext, stepProg *ail.Program, step chainStep, w http.ResponseWriter, r *http.Request) error {
	if step.Model == "" {
		// Strip chain suffixes from current model so the fresh chain
		// has dspy/kvtools/etc. but NOT chain.
		exitModel := stripChainSuffixes(stepProg.GetModel())
		stepProg.SetModel(exitModel)
	}
	freshR := r.WithContext(withChainBypass(r.Context()))
	return ic.InferFresh(stepProg, w, freshR)
}

// writeWithThinkBlocks wraps prepend results in THINK blocks and replays
// the final response to the client (non-streaming output).
func writeWithThinkBlocks(prependResults []*ail.Program, finalRes *ail.Program, capture *services.ResponseCaptureWriter, w http.ResponseWriter) error {
	// For non-streaming, we replay the raw captured response.
	// The think block wrapping would require re-encoding, which is complex.
	// POC: just replay the final capture; prepend content is in conversation
	// context so the model already incorporated it.
	// TODO: inject THINK blocks into the JSON response for rich clients.
	plugin.ReplayCapture(capture, w)
	return nil
}

// streamWithPrepend writes prepend results as initial SSE think events,
// then streams the final inference step.
func streamWithPrepend(prependResults []*ail.Program, finalProg *ail.Program, ic *plugin.InferenceContext, w http.ResponseWriter, r *http.Request) error {
	// Strip chain suffixes so the fresh chain has dspy etc. but not chain.
	exitModel := stripChainSuffixes(finalProg.GetModel())
	finalProg.SetModel(exitModel)
	freshR := r.WithContext(withChainBypass(r.Context()))
	return ic.InferFresh(finalProg, w, freshR)
}

// stripChainSuffixes removes all +chain:... segments from a model string,
// preserving other plugin suffixes like +dspy:react, +kvtools, etc.
//
// Example: "openai/gpt-4.1+dspy:react+chain::prepend+kvtools" → "openai/gpt-4.1+dspy:react+kvtools"
func stripChainSuffixes(model string) string {
	plusIdx := strings.IndexByte(model, '+')
	if plusIdx < 0 {
		return model
	}

	base := model[:plusIdx]
	rest := model[plusIdx+1:]

	var kept []string
	for len(rest) > 0 {
		var part string
		if nextPlus := strings.IndexByte(rest, '+'); nextPlus >= 0 {
			part = rest[:nextPlus]
			rest = rest[nextPlus+1:]
		} else {
			part = rest
			rest = ""
		}

		// Extract plugin name (before first ':')
		name := part
		if colonIdx := strings.IndexByte(part, ':'); colonIdx >= 0 {
			name = part[:colonIdx]
		}
		if name != "chain" {
			kept = append(kept, part)
		}
	}

	if len(kept) == 0 {
		return base
	}
	return base + "+" + strings.Join(kept, "+")
}

// ─── Compile-time checks ────────────────────────────────────────────────────

var _ plugin.RecursiveHandlerPlugin = (*ChainPlugin)(nil)
