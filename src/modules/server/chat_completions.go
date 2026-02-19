package server

import (
	"context"
	"io"
	"net/http"

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

// ChatCompletionsModule handles OpenAI-style chat completions requests.
// AIL rework: all data passes through *ail.Program, no more styles.PartialJSON.
type ChatCompletionsModule struct {
	RouterName string `json:"router,omitempty"`
	logger     *zap.Logger
}

// requestParser parses incoming Chat Completions requests into AIL
var requestParser = &ail.ChatCompletionsParser{}

// responseEmitter emits AIL programs as Chat Completions JSON for the client
var responseEmitter = &ail.ChatCompletionsEmitter{}

func ParseChatCompletionsModule(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var m ChatCompletionsModule
	for h.Next() {
		for h.NextBlock(0) {
			switch h.Val() {
			case "router":
				if !h.NextArg() {
					return nil, h.ArgErr()
				}
				m.RouterName = h.Val()
			default:
				return nil, h.Errf("unrecognized ai_openai_chat_completions option '%s'", h.Val())
			}
		}
	}
	return &m, nil
}

func (*ChatCompletionsModule) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.ai_openai_chat_completions",
		New: func() caddy.Module { return new(ChatCompletionsModule) },
	}
}

func (m *ChatCompletionsModule) Provision(ctx caddy.Context) error {
	m.logger = ctx.Logger(m)

	// Provision package-level loggers
	plugin.Logger = m.logger.Named("plugin")
	openai.Logger = m.logger.Named("openai")
	virtual.Logger = m.logger.Named("virtual")

	return nil
}

// serveChatCompletions handles non-streaming inference.
func (m *ChatCompletionsModule) serveChatCompletions(
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

	// Run after plugins
	resProg, err = chain.RunAfter(&p.Impl, r, prog, res, resProg)
	if err != nil {
		m.logger.Error("plugin after hook error", zap.Error(err))
		http.Error(w, "Plugin error", http.StatusInternalServerError)
		return nil
	}

	// Emit response as Chat Completions JSON
	resData, err := responseEmitter.EmitResponse(resProg)
	if err != nil {
		m.logger.Error("Failed to emit response", zap.Error(err))
		http.Error(w, "Response emission error", http.StatusInternalServerError)
		return nil
	}

	w.Header().Set("Content-Type", "application/json")
	_, err = w.Write(resData)
	return err
}

// serveChatCompletionsStream handles streaming inference.
// Uses ail.StreamConverter for proper cross-style conversion with metadata
// tracking, tool-call buffering, and multi-event splitting.
func (m *ChatCompletionsModule) serveChatCompletionsStream(
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

	// Create a stream converter: provider style → client style (ChatCompletions).
	// Handles metadata injection, tool-call buffering, and multi-event splitting.
	conv, err := ail.NewStreamConverter(p.Impl.Style, ail.StyleChatCompletions)
	if err != nil {
		m.logger.Error("failed to create stream converter", zap.Error(err))
		return err
	}

	hres, stream, err := cmd.DoInferenceStream(&p.Impl, prog, r)
	if err != nil {
		m.logger.Error("inference stream error (start)", zap.String("provider", p.Name), zap.Error(err))
		_ = chain.RunError(&p.Impl, r, prog, hres, err)
		_ = sseWriter.WriteError("start failed")
		_ = sseWriter.WriteDone()
		return err
	}

	// Accumulate chunks for assembly into a complete response for StreamEnd.
	chunks := make([]*ail.Program, 0, 10)

	for chunk := range stream {
		if chunk.RuntimeError != nil {
			_ = sseWriter.WriteError(chunk.RuntimeError.Error())
			_ = chain.RunError(&p.Impl, r, prog, hres, chunk.RuntimeError)
			return nil
		}

		chunkProg := chunk.Data

		// Run after-chunk plugins (may modify the AIL program)
		chunkProg, err = chain.RunAfterChunk(&p.Impl, r, prog, hres, chunkProg)
		if err != nil {
			m.logger.Error("plugin after chunk error", zap.Error(err))
			continue
		}

		if chunkProg != nil {
			chunks = append(chunks, chunkProg)

			// Convert chunk to client format via StreamConverter.
			// PushProgram handles metadata tracking, tool buffering,
			// and may return 0..N output chunks.
			outputs, err := conv.PushProgram(chunkProg)
			if err != nil {
				m.logger.Error("stream convert error", zap.Error(err))
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

	// Flush any buffered data from the converter (e.g., pending tool calls
	// for targets that require complete function objects).
	if final, err := conv.Flush(); err != nil {
		m.logger.Error("stream converter flush error", zap.Error(err))
	} else {
		for _, out := range final {
			if err := sseWriter.WriteRaw(out); err != nil {
				m.logger.Error("stream flush write error", zap.Error(err))
				break
			}
		}
	}

	// Assemble all chunk programs into a single response program.
	assembled := ail.NewProgram()
	for _, c := range chunks {
		if c != nil {
			assembled = assembled.Append(c)
		}
	}

	// Run stream end plugins with the fully assembled response.
	_ = chain.RunStreamEnd(&p.Impl, r, prog, hres, assembled)

	_ = sseWriter.WriteDone()
	return nil
}

func (m *ChatCompletionsModule) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	m.logger.Debug("Chat completions request received", zap.String("path", r.URL.Path), zap.String("method", r.Method))

	// Check if an AIL program is already in context (recursive call from plugin)
	var prog *ail.Program
	if ctxProg, ok := ail.ProgramFromContext(r.Context()); ok {
		prog = ctxProg
		m.logger.Debug("Using AIL program from context (recursive call)")
	} else {
		// Parse incoming request body into AIL
		reqBody, err := io.ReadAll(r.Body)
		if err != nil {
			m.logger.Error("failed to read request body", zap.Error(err))
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return nil
		}

		m.logger.Debug("Request body read", zap.Int("body_length", len(reqBody)))

		prog, err = requestParser.ParseRequest(reqBody)
		if err != nil {
			m.logger.Error("failed to parse request into AIL", zap.Error(err))
			http.Error(w, "invalid request JSON", http.StatusBadRequest)
			return nil
		}

	}

	m.logger.Debug("Request parsed",
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

	traceId := uuid.New().String()
	r = r.WithContext(context.WithValue(r.Context(), plugin.ContextTraceID(), traceId))

	// Notify plugins of the initial parsed request (e.g., sampler).
	chain.RunRequestInit(r, prog)

	// Create invoker for recursive handler plugins (fallback, parallel, etc.)
	invoker := plugin.NewCaddyModuleInvoker(m, requestParser)

	// Check if any recursive handler plugin wants to handle this request
	handled, err := chain.RunRecursiveHandlers(invoker, prog, w, r)
	if handled {
		if err != nil {
			m.logger.Error("recursive handler plugin failed", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return nil
	}

	// Normal flow - handle request directly
	err = m.handleRequest(router, chain, prog, w, r)
	if err != nil {
		m.logger.Error("request handling failed", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return nil
	}

	return nil
}

// ServeNonStreaming implements InferenceHandler for ChatCompletions.
func (m *ChatCompletionsModule) ServeNonStreaming(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	return m.serveChatCompletions(p, cmd, chain, prog, w, r)
}

// ServeStreaming implements InferenceHandler for ChatCompletions.
func (m *ChatCompletionsModule) ServeStreaming(
	p *modules.ProviderConfig,
	cmd drivers.InferenceCommand,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	return m.serveChatCompletionsStream(p, cmd, chain, prog, w, r)
}

// handleRequest handles a single request to providers (used both directly and by recursive plugins).
func (m *ChatCompletionsModule) handleRequest(
	router *modules.RouterModule,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	return RunInferencePipeline(router, chain, prog, w, r, m, m.logger)
}

// trySampleAIL* functions have been moved to src/plugins/sampler.go.

var (
	_ caddy.Provisioner           = (*ChatCompletionsModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*ChatCompletionsModule)(nil)
)
