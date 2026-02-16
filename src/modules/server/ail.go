package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/google/uuid"
	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/modules"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"go.uber.org/zap"
)

// AILModule handles raw AIL (AI Intermediate Language) requests over HTTP.
//
// Accepts AIL programs in binary or text (disassembly) format and returns
// the inference response as an AIL program in the same format.
//
// Content negotiation:
//   - Content-Type: application/x-ail  → binary AIL input
//   - Content-Type: text/plain         → text (disassembly) AIL input
//   - Accept: application/x-ail        → binary AIL output (default)
//   - Accept: text/plain               → text (disassembly) AIL output
//
// If Content-Type is absent or unrecognized, the handler auto-detects:
// binary if the body starts with the AIL magic bytes ("AIL\x00"), text otherwise.
type AILModule struct {
	RouterName string `json:"router,omitempty"`
	logger     *zap.Logger
}

func ParseAILModule(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var m AILModule
	for h.Next() {
		for h.NextBlock(0) {
			switch h.Val() {
			case "router":
				if !h.NextArg() {
					return nil, h.ArgErr()
				}
				m.RouterName = h.Val()
			default:
				return nil, h.Errf("unrecognized ail option '%s'", h.Val())
			}
		}
	}
	return &m, nil
}

func (*AILModule) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.ail",
		New: func() caddy.Module { return new(AILModule) },
	}
}

func (m *AILModule) Provision(ctx caddy.Context) error {
	m.logger = ctx.Logger(m)
	return nil
}

// ailMagic is the 4-byte header of binary AIL files.
var ailMagic = []byte("AIL\x00")

func (m *AILModule) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	m.logger.Debug("AIL request received", zap.String("method", r.Method))

	body, err := io.ReadAll(r.Body)
	if err != nil {
		m.logger.Error("failed to read request body", zap.Error(err))
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return nil
	}

	if len(body) == 0 {
		http.Error(w, "empty request body", http.StatusBadRequest)
		return nil
	}

	// Determine input format
	inputBinary := m.isInputBinary(r, body)

	// Parse the AIL program
	var prog *ail.Program
	if inputBinary {
		prog, err = ail.Decode(bytes.NewReader(body))
		if err != nil {
			m.logger.Error("failed to decode binary AIL", zap.Error(err))
			http.Error(w, "invalid binary AIL: "+err.Error(), http.StatusBadRequest)
			return nil
		}
	} else {
		prog, err = ail.Asm(string(body))
		if err != nil {
			m.logger.Error("failed to assemble text AIL", zap.Error(err))
			http.Error(w, "invalid AIL text: "+err.Error(), http.StatusBadRequest)
			return nil
		}
	}

	m.logger.Debug("AIL program parsed",
		zap.String("model", prog.GetModel()),
		zap.Bool("streaming", prog.IsStreaming()),
		zap.Int("instructions", prog.Len()),
		zap.Bool("input_binary", inputBinary))

	// Determine output format from Accept header (default: same as input)
	wantBinaryOutput := m.wantBinaryOutput(r, inputBinary)

	// Sample AIL to disk when SAMPLE_AIL is set
	if hash := trySampleAIL(body, prog, m.logger); hash != "" {
		r = r.WithContext(context.WithValue(r.Context(), ctxKeySampleHash, hash))
	}

	// Route through the provider pipeline
	resProg, err := m.handleAILRequest(prog, w, r)
	if err != nil {
		m.logger.Error("AIL request handling failed", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return nil
	}

	if resProg == nil {
		http.Error(w, "no response from providers", http.StatusBadGateway)
		return nil
	}

	// Sample response AIL
	if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
		trySampleAILResponse(hash, resProg, m.logger)
	}

	// Encode the response
	if wantBinaryOutput {
		w.Header().Set("Content-Type", "application/x-ail")
		var buf bytes.Buffer
		if err := resProg.Encode(&buf); err != nil {
			m.logger.Error("failed to encode binary AIL response", zap.Error(err))
			http.Error(w, "response encoding error", http.StatusInternalServerError)
			return nil
		}
		_, err = w.Write(buf.Bytes())
	} else {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, err = io.WriteString(w, resProg.Disasm())
	}

	if err != nil {
		m.logger.Error("failed to write AIL response", zap.Error(err))
	}
	return nil
}

// isInputBinary determines whether the request body is binary AIL or text.
func (m *AILModule) isInputBinary(r *http.Request, body []byte) bool {
	ct := r.Header.Get("Content-Type")
	switch {
	case strings.HasPrefix(ct, "application/x-ail"), strings.HasPrefix(ct, "application/octet-stream"):
		return true
	case strings.HasPrefix(ct, "text/plain"), strings.HasPrefix(ct, "text/x-ail"):
		return false
	default:
		// Auto-detect: binary AIL starts with magic bytes
		return len(body) >= 4 && bytes.Equal(body[:4], ailMagic)
	}
}

// wantBinaryOutput determines the desired output format from the Accept header.
func (m *AILModule) wantBinaryOutput(r *http.Request, inputBinary bool) bool {
	accept := r.Header.Get("Accept")
	switch {
	case strings.Contains(accept, "application/x-ail"), strings.Contains(accept, "application/octet-stream"):
		return true
	case strings.Contains(accept, "text/plain"), strings.Contains(accept, "text/x-ail"):
		return false
	default:
		// Default: mirror the input format
		return inputBinary
	}
}

// handleAILRequest runs the AIL program through the provider pipeline and
// returns the response as an AIL program.
//
// If the request has SET_STREAM, streaming is performed internally and all
// chunks are assembled into a complete response program.
func (m *AILModule) handleAILRequest(
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) (*ail.Program, error) {
	router, ok := modules.GetRouter(m.RouterName)
	if !ok {
		m.logger.Error("Router not found", zap.String("name", m.RouterName))
		return nil, fmt.Errorf("router %q not found", m.RouterName)
	}

	// Collect incoming auth
	r, err := router.Impl.Auth.CollectIncomingAuth(r)
	if err != nil {
		m.logger.Error("failed to collect incoming auth", zap.Error(err))
		return nil, fmt.Errorf("authentication error: %w", err)
	}

	// Resolve virtual model aliases
	model := prog.GetModel()
	var chain *plugin.PluginChain
	const maxRewriteDepth = 10
	for i := 0; i < maxRewriteDepth; i++ {
		chain = plugin.TryResolvePlugins(*r.URL, model)
		if rewritten := chain.RunModelRewrite(model); rewritten != model {
			m.logger.Debug("Virtual model resolved",
				zap.String("from", model),
				zap.String("to", rewritten))
			model = rewritten
			continue
		}
		break
	}
	prog.SetModel(model)

	traceId := uuid.New().String()
	r = r.WithContext(context.WithValue(r.Context(), plugin.ContextTraceID(), traceId))

	// Resolve providers
	providers, resolvedModel := router.ResolveProvidersOrderAndModel(prog.GetModel())

	m.logger.Debug("AIL resolved providers",
		zap.String("model", resolvedModel),
		zap.Strings("providers", providers))

	var lastErr error
	for _, name := range providers {
		p, ok := router.ProviderConfigs[name]
		if !ok {
			m.logger.Error("provider not found", zap.String("name", name))
			continue
		}

		cmd, ok := p.Impl.Commands["inference"].(drivers.InferenceCommand)
		if !ok {
			continue
		}

		providerProg := prog.Clone()
		providerProg.SetModel(resolvedModel)

		// Run before plugins
		processedProg, err := chain.RunBefore(&p.Impl, r, providerProg)
		if err != nil {
			m.logger.Error("plugin before hook error", zap.String("provider", name), zap.Error(err))
			lastErr = err
			continue
		}
		providerProg = processedProg

		// Sample the upstream-prepared AIL (after model resolve + plugins)
		if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
			trySampleAILUpstream(hash, providerProg, m.logger)
		}

		// Set response headers
		w.Header().Set("X-Real-Provider-Id", name)
		w.Header().Set("X-Real-Model-Id", resolvedModel)

		var resProg *ail.Program

		if providerProg.IsStreaming() {
			resProg, err = m.doStreamingInference(p, cmd, chain, providerProg, r)
		} else {
			resProg, err = m.doNonStreamingInference(p, cmd, chain, providerProg, r)
		}

		if err != nil {
			lastErr = err
			continue
		}

		return resProg, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, nil
}

// doNonStreamingInference performs non-streaming inference and returns the response AIL program.
func (m *AILModule) doNonStreamingInference(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	r *http.Request,
) (*ail.Program, error) {
	res, resProg, err := cmd.DoInference(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference error", zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, res, err)
		return nil, err
	}

	resProg, err = chain.RunAfter(&p.Impl, r, prog, res, resProg)
	if err != nil {
		m.logger.Error("plugin after hook error", zap.Error(err))
		return nil, err
	}

	return resProg, nil
}

// doStreamingInference performs streaming inference, collecting all chunks
// into a complete response AIL program via StreamAssembler.
func (m *AILModule) doStreamingInference(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	r *http.Request,
) (*ail.Program, error) {
	hres, stream, err := cmd.DoInferenceStream(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference stream error", zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, hres, err)
		return nil, err
	}

	asm := ail.NewStreamAssembler()
	var lastChunk *ail.Program

	for chunk := range stream {
		if chunk.RuntimeError != nil {
			_ = chain.RunError(&p.Impl, r, prog, hres, chunk.RuntimeError)
			return nil, chunk.RuntimeError
		}

		chunkProg := chunk.Data

		chunkProg, err = chain.RunAfterChunk(&p.Impl, r, prog, hres, chunkProg)
		if err != nil {
			m.logger.Error("plugin after chunk error", zap.Error(err))
			continue
		}

		if chunkProg != nil {
			lastChunk = chunkProg
			asm.Push(chunkProg)
		}
	}

	// Run stream end plugins
	_ = chain.RunStreamEnd(&p.Impl, r, prog, hres, lastChunk)

	return asm.Program(), nil
}

var (
	_ caddy.Provisioner           = (*AILModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*AILModule)(nil)
)
