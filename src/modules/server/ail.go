package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"strings"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/google/uuid"
	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/drivers/openai"
	"github.com/neutrome-labs/open-ai-router/src/drivers/virtual"
	"github.com/neutrome-labs/open-ai-router/src/modules"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"go.uber.org/zap"
)

// ailOutputCtxKey carries the desired output encoding (binary vs text)
// through the request context so ServeNonStreaming/ServeStreaming can read it.
type ailOutputCtxKey struct{}

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
// Streaming: when the program contains SET_STREAM, the response is pushed
// incrementally as SSE events (text/event-stream). Each data event carries
// one AIL chunk program in text disasm (Accept: text/plain) or base64-encoded
// binary (Accept: application/x-ail). The stream ends with [DONE].
//
// Recursive handlers: plugins that implement RecursiveHandlerPlugin (e.g.
// ToolPlugin for on-router tool dispatch) are fully supported, including
// streaming requests via the hybrid buffer-then-stream approach.
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

	// Provision package-level loggers so that plugins, drivers, and virtual
	// providers log correctly when only the AIL endpoint is used.
	plugin.Logger = m.logger.Named("plugin")
	openai.Logger = m.logger.Named("openai")
	virtual.Logger = m.logger.Named("virtual")

	return nil
}

// ailMagic is the 4-byte header of binary AIL files.
var ailMagic = []byte("AIL\x00")

func (m *AILModule) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	m.logger.Debug("AIL request received", zap.String("method", r.Method))

	var prog *ail.Program
	var wantBinaryOutput bool

	// Check if an AIL program is already in context (recursive call from plugin).
	if ctxProg, ok := ail.ProgramFromContext(r.Context()); ok {
		prog = ctxProg
		m.logger.Debug("Using AIL program from context (recursive call)")
		// Use text output for internal recursive calls — simpler to parse back,
		// no base64 overhead, and the response stays in-process anyway.
		wantBinaryOutput = false
	} else {
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

		// Determine input format.
		inputBinary := m.isInputBinary(r, body)

		// Parse the AIL program.
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

		// Determine output format from Accept header (default: same as input).
		wantBinaryOutput = m.wantBinaryOutput(r, inputBinary)

		// Sample AIL to disk when SAMPLE_AIL is set.
		if hash := trySampleAIL(body, prog, m.logger); hash != "" {
			r = r.WithContext(context.WithValue(r.Context(), ctxKeySampleHash, hash))
		}
	}

	// Store output format in context for InferenceHandler methods.
	r = r.WithContext(context.WithValue(r.Context(), ailOutputCtxKey{}, wantBinaryOutput))

	router, ok := modules.GetRouter(m.RouterName)
	if !ok {
		m.logger.Error("Router not found", zap.String("name", m.RouterName))
		http.Error(w, "Router not found", http.StatusInternalServerError)
		return nil
	}

	// Shared preamble: auth, model rewrite, plugin resolution.
	chain, r, err := RequestPreamble(router, prog, r, m.logger)
	if err != nil {
		http.Error(w, "authentication error", http.StatusUnauthorized)
		return nil
	}

	traceId := uuid.New().String()
	r = r.WithContext(context.WithValue(r.Context(), plugin.ContextTraceID(), traceId))

	// Recursive handler plugins (tool dispatch, fallback, parallel, etc.).
	invoker := plugin.NewCaddyModuleInvoker(m, &ailResponseParser{})
	handled, err := chain.RunRecursiveHandlers(invoker, prog, w, r)
	if handled {
		if err != nil {
			m.logger.Error("recursive handler plugin failed", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return nil
	}

	// Normal flow — shared provider iteration pipeline.
	if err := RunInferencePipeline(router, chain, prog, w, r, m, m.logger); err != nil {
		m.logger.Error("AIL request handling failed", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return nil
	}

	return nil
}

// ─── InferenceHandler implementation ─────────────────────────────────────────

// ServeNonStreaming implements InferenceHandler for AIL.
func (m *AILModule) ServeNonStreaming(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	res, resProg, err := cmd.DoInference(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference error", zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, res, err)
		return err
	}

	resProg, err = chain.RunAfter(&p.Impl, r, prog, res, resProg)
	if err != nil {
		m.logger.Error("plugin after hook error", zap.Error(err))
		return err
	}

	// Sample response AIL.
	if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
		trySampleAILResponse(hash, resProg, m.logger)
	}

	// Encode and write the response.
	wantBinary, _ := r.Context().Value(ailOutputCtxKey{}).(bool)
	return m.writeAILResponse(w, resProg, wantBinary)
}

// ServeStreaming implements InferenceHandler for AIL.
// Pushes AIL chunk programs to the client incrementally via SSE.
func (m *AILModule) ServeStreaming(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	sseWriter := sse.NewWriter(w)

	if err := sseWriter.WriteHeartbeat("ok"); err != nil {
		return err
	}

	hres, stream, err := cmd.DoInferenceStream(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference stream error (start)",
			zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, hres, err)
		_ = sseWriter.WriteError("start failed")
		_ = sseWriter.WriteDone()
		return err
	}

	wantBinary, _ := r.Context().Value(ailOutputCtxKey{}).(bool)

	chunks := make([]*ail.Program, 0, 10)
	var lastChunk *ail.Program

	for chunk := range stream {
		if chunk.RuntimeError != nil {
			_ = sseWriter.WriteError(chunk.RuntimeError.Error())
			_ = chain.RunError(&p.Impl, r, prog, hres, chunk.RuntimeError)
			return nil
		}

		chunkProg := chunk.Data

		// Run after-chunk plugins (may modify the AIL program).
		chunkProg, err = chain.RunAfterChunk(&p.Impl, r, prog, hres, chunkProg)
		if err != nil {
			m.logger.Error("plugin after chunk error", zap.Error(err))
			continue
		}

		if chunkProg != nil {
			lastChunk = chunkProg
			chunks = append(chunks, chunkProg)

			// Encode the chunk and push via SSE.
			chunkData, encErr := m.encodeAILChunk(chunkProg, wantBinary)
			if encErr != nil {
				m.logger.Error("chunk encode error", zap.Error(encErr))
				continue
			}
			if err := sseWriter.WriteRaw(chunkData); err != nil {
				m.logger.Error("stream write error", zap.Error(err))
				return err
			}
		}
	}

	_ = chain.RunStreamEnd(&p.Impl, r, prog, hres, lastChunk)

	// Sample the assembled complete response (all chunks).
	assembled := ail.NewProgram()
	for _, c := range chunks {
		if c != nil {
			assembled = assembled.Append(c)
		}
	}
	if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
		trySampleAILResponse(hash, assembled, m.logger)
	}

	_ = sseWriter.WriteDone()
	return nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// writeAILResponse encodes an AIL program and writes it to the response writer.
func (m *AILModule) writeAILResponse(w http.ResponseWriter, prog *ail.Program, wantBinary bool) error {
	if wantBinary {
		w.Header().Set("Content-Type", "application/x-ail")
		var buf bytes.Buffer
		if err := prog.Encode(&buf); err != nil {
			m.logger.Error("failed to encode binary AIL response", zap.Error(err))
			return err
		}
		_, err := w.Write(buf.Bytes())
		return err
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, err := io.WriteString(w, prog.Disasm())
	return err
}

// encodeAILChunk encodes a single AIL chunk program for SSE delivery.
// Text mode: returns the disasm directly.
// Binary mode: base64-encodes the binary AIL (SSE is text-based).
func (m *AILModule) encodeAILChunk(prog *ail.Program, wantBinary bool) ([]byte, error) {
	if !wantBinary {
		return []byte(prog.Disasm()), nil
	}
	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		return nil, err
	}
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	return []byte(encoded), nil
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

// ─── AIL Response Parser ─────────────────────────────────────────────────────

// ailResponseParser implements plugin.ResponseParser for the AIL wire format.
// Auto-detects binary AIL (magic header) vs text (disassembly) and parses accordingly.
// Used by CaddyModuleInvoker when the AIL module is invoked recursively.
type ailResponseParser struct{}

func (p *ailResponseParser) ParseResponse(data []byte) (*ail.Program, error) {
	// Binary AIL starts with the magic header.
	if len(data) >= 4 && bytes.Equal(data[:4], ailMagic) {
		return ail.Decode(bytes.NewReader(data))
	}
	// Text (disassembly) format.
	return ail.Asm(string(data))
}

var (
	_ caddy.Provisioner           = (*AILModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*AILModule)(nil)
	_ InferenceHandler            = (*AILModule)(nil)
	_ plugin.ResponseParser       = (*ailResponseParser)(nil)
)
