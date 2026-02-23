package server

import (
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/google/uuid"
	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/drivers/virtual"
	"github.com/neutrome-labs/open-ai-router/src/modules"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"github.com/neutrome-labs/open-ai-router/src/styles"
	"go.uber.org/zap"
)

// InferenceSseModule is a generic endpoint handler for any AIL-supported API style.
// It parses inbound requests using the configured client style, runs the
// shared plugin/inference pipeline over AIL programs, and emits responses
// back in the same client format.
//
// Adding a new client-facing API (e.g. Anthropic Messages) requires only
// a Caddyfile route pointing to ai_inference_sse with the desired style — no new Go code.
//
// Caddyfile:
//
//	ai_inference_sse {
//	    router <name>
//	    style  <style>   # chat-completions | openai-responses | anthropic-messages | ...
//	}
type InferenceSseModule struct {
	RouterName string `json:"router,omitempty"`
	StyleName  string `json:"style,omitempty"`

	// Resolved at provision time from StyleName.
	clientStyle ail.Style
	reqParser   ail.Parser
	respEmitter ail.ResponseEmitter
	respParser  ail.ResponseParser // used by CaddyModuleInvoker for capture parsing
	logger      *zap.Logger
}

// ─── Caddyfile parsing ──────────────────────────────────────────────────────

func parseInferenceSseModuleHelper(h httpcaddyfile.Helper, defaultStyle string) (*InferenceSseModule, error) {
	m := &InferenceSseModule{StyleName: defaultStyle}
	for h.Next() {
		for h.NextBlock(0) {
			switch h.Val() {
			case "router":
				if !h.NextArg() {
					return nil, h.ArgErr()
				}
				m.RouterName = h.Val()
			case "style":
				if !h.NextArg() {
					return nil, h.ArgErr()
				}
				m.StyleName = h.Val()
			default:
				return nil, h.Errf("unrecognized ai_inference_sse option '%s'", h.Val())
			}
		}
	}
	return m, nil
}

// ParseInferenceSseModule handles the ai_inference_sse { ... } directive.
func ParseInferenceSseModule(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	return parseInferenceSseModuleHelper(h, "chat-completions")
}

// ─── Caddy module lifecycle ─────────────────────────────────────────────────

func (*InferenceSseModule) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.ai_inference_sse",
		New: func() caddy.Module { return new(InferenceSseModule) },
	}
}

func (m *InferenceSseModule) Provision(ctx caddy.Context) error {
	m.logger = ctx.Logger(m)

	// Provision package-level loggers.
	plugin.Logger = m.logger.Named("plugin")
	drivers.Logger = m.logger.Named("drivers")
	virtual.Logger = m.logger.Named("virtual")

	if m.StyleName == "" {
		m.StyleName = "chat-completions"
	}

	s, err := styles.ParseStyle(m.StyleName)
	if err != nil {
		return fmt.Errorf("ai_inference_sse: invalid style %q: %w", m.StyleName, err)
	}
	m.clientStyle = s

	m.reqParser, err = ail.GetParser(s)
	if err != nil {
		return fmt.Errorf("ai_inference_sse: no request parser for style %s: %w", s, err)
	}
	m.respEmitter, err = ail.GetResponseEmitter(s)
	if err != nil {
		return fmt.Errorf("ai_inference_sse: no response emitter for style %s: %w", s, err)
	}
	m.respParser, err = ail.GetResponseParser(s)
	if err != nil {
		return fmt.Errorf("ai_inference_sse: no response parser for style %s: %w", s, err)
	}

	return nil
}

// ─── ServeHTTP ──────────────────────────────────────────────────────────────

func (m *InferenceSseModule) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	// Check if an AIL program is already in context (recursive call from plugin).
	var prog *ail.Program
	if ctxProg, ok := ail.ProgramFromContext(r.Context()); ok {
		prog = ctxProg
		m.logger.Debug("Using AIL program from context (recursive call)")
	} else {
		reqBody, err := io.ReadAll(r.Body)
		if err != nil {
			m.logger.Error("failed to read request body", zap.Error(err))
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return nil
		}

		prog, err = m.reqParser.ParseRequest(reqBody)
		if err != nil {
			m.logger.Error("failed to parse request",
				zap.String("style", string(m.clientStyle)), zap.Error(err))
			http.Error(w, "invalid request", http.StatusBadRequest)
			return nil
		}
	}

	m.logger.Debug("Request parsed",
		zap.String("style", string(m.clientStyle)),
		zap.String("model", prog.GetModel()),
		zap.Bool("streaming", prog.IsStreaming()))

	router, ok := modules.GetRouter(m.RouterName)
	if !ok {
		m.logger.Error("Router not found", zap.String("name", m.RouterName))
		http.Error(w, "Router not found", http.StatusInternalServerError)
		return nil
	}

	chain, r, err := RequestPreamble(router, prog, r, m.logger)
	if err != nil {
		http.Error(w, "authentication error", http.StatusUnauthorized)
		return nil
	}

	traceID := uuid.New().String()
	ctx := r.Context()
	ctx = context.WithValue(ctx, plugin.ContextTraceID(), traceID)
	ctx = context.WithValue(ctx, plugin.ContextClientStyleKey(), m.clientStyle)
	r = r.WithContext(ctx)

	// Notify plugins of the initial parsed request (e.g., sampler).
	chain.RunRequestInit(r, prog)

	// The invoker captures responses in the client format and parses them back to AIL.
	invoker := plugin.NewCaddyModuleInvoker(m, m.respParser)

	// Check if any recursive handler plugin wants to handle this request.
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
		m.logger.Error("request handling failed", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return nil
	}

	return nil
}

// ─── InferenceHandler implementation ────────────────────────────────────────

// ServeNonStreaming implements InferenceHandler.
func (m *InferenceSseModule) ServeNonStreaming(
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
		http.Error(w, "Plugin error", http.StatusInternalServerError)
		return nil
	}

	resData, err := m.respEmitter.EmitResponse(resProg)
	if err != nil {
		m.logger.Error("Failed to emit response", zap.Error(err))
		http.Error(w, "Response emission error", http.StatusInternalServerError)
		return nil
	}

	w.Header().Set("Content-Type", "application/json")
	_, err = w.Write(resData)
	return err
}

// ServeStreaming implements InferenceHandler.
func (m *InferenceSseModule) ServeStreaming(
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

	// StreamConverter handles cross-style chunk conversion (provider → client).
	conv, err := ail.NewStreamConverter(p.Impl.Style, m.clientStyle)
	if err != nil {
		m.logger.Error("failed to create stream converter", zap.Error(err))
		return err
	}

	hres, stream, err := cmd.DoInferenceStream(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference stream error", zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, hres, err)
		_ = sseWriter.WriteError("start failed")
		_ = sseWriter.WriteDone()
		return err
	}

	chunks := make([]*ail.Program, 0, 10)

	for chunk := range stream {
		if chunk.RuntimeError != nil {
			_ = sseWriter.WriteError(chunk.RuntimeError.Error())
			_ = chain.RunError(&p.Impl, r, prog, hres, chunk.RuntimeError)
			return nil
		}

		chunkProg := chunk.Data

		chunkProg, err = chain.RunAfterChunk(&p.Impl, r, prog, hres, chunkProg)
		if err != nil {
			m.logger.Error("plugin after chunk error", zap.Error(err))
			continue
		}

		if chunkProg != nil {
			chunks = append(chunks, chunkProg)

			outputs, convErr := conv.PushProgram(chunkProg)
			if convErr != nil {
				m.logger.Error("stream convert error", zap.Error(convErr))
				continue
			}

			for _, out := range outputs {
				if err := sseWriter.WriteRaw(out); err != nil {
					m.logger.Error("stream write error", zap.Error(err))
					return err
				}
			}
		}
	}

	// Flush buffered data (e.g. pending tool calls).
	if final, flushErr := conv.Flush(); flushErr != nil {
		m.logger.Error("stream converter flush error", zap.Error(flushErr))
	} else {
		for _, out := range final {
			if err := sseWriter.WriteRaw(out); err != nil {
				m.logger.Error("stream flush write error", zap.Error(err))
				break
			}
		}
	}

	// Assemble all chunks into a single response program for StreamEnd.
	assembled := ail.NewProgram()
	for _, c := range chunks {
		if c != nil {
			assembled = assembled.Append(c)
		}
	}

	_ = chain.RunStreamEnd(&p.Impl, r, prog, hres, assembled)

	_ = sseWriter.WriteDone()
	return nil
}

// ─── Compile-time checks ────────────────────────────────────────────────────

var (
	_ caddy.Provisioner           = (*InferenceSseModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*InferenceSseModule)(nil)
	_ InferenceHandler            = (*InferenceSseModule)(nil)
)
