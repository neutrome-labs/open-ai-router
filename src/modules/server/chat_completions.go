package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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

// sampleAILDir is the directory to dump AIL programs into.
// Set via the SAMPLE_AIL environment variable at startup.
var sampleAILDir = os.Getenv("SAMPLE_AIL")

// ctxKeySampleHash is the context key for the request sample hash (used to pair response samples).
type sampleHashKey struct{}

var ctxKeySampleHash = sampleHashKey{}

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

	// Sample response AIL
	if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
		trySampleAILResponse(hash, resProg, m.logger)
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

	// StreamAssembler accumulates all chunk programs into a complete response
	// for sampling and the StreamEnd plugin hook.
	chunks := make([]*ail.Program, 0, 10)
	var lastChunk *ail.Program

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
			lastChunk = chunkProg
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

	// Run stream end plugins
	_ = chain.RunStreamEnd(&p.Impl, r, prog, hres, lastChunk)

	disasm := ""
	for i, c := range chunks {
		if c == nil {
			continue
		}
		disasm += fmt.Sprintf("# Chunk %d\n", i)
		disasm += c.Disasm() + "\n"
	}

	asm, err := ail.Asm(disasm)
	if err != nil {
		m.logger.Error("failed to assemble streamed AIL", zap.Error(err))
	}

	// Sample the assembled complete response (all chunks, not just last)
	if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
		trySampleAILResponse(hash, asm, m.logger)
	}

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

		// Sample AIL to disk when SAMPLE_AIL is set
		if hash := trySampleAIL(reqBody, prog, m.logger); hash != "" {
			r = r.WithContext(context.WithValue(r.Context(), ctxKeySampleHash, hash))
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

	// Collect incoming auth
	r, err := router.Impl.Auth.CollectIncomingAuth(r)
	if err != nil {
		m.logger.Error("failed to collect incoming auth", zap.Error(err))
		http.Error(w, "authentication error", http.StatusUnauthorized)
		return nil
	}

	// Resolve virtual model aliases (may chain: virtual→virtual→real).
	// Each iteration re-resolves the plugin chain for the new model so that
	// plugins injected by the virtual mapping (target+plugins) are picked up.
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

	m.logger.Debug("Resolved plugins", zap.Int("plugin_count", len(chain.GetPlugins())))

	traceId := uuid.New().String()
	r = r.WithContext(context.WithValue(r.Context(), plugin.ContextTraceID(), traceId))

	// Create invoker for recursive handler plugins (fallback, parallel, etc.)
	invoker := plugin.NewCaddyModuleInvoker(m)

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

// handleRequest handles a single request to providers (used both directly and by recursive plugins).
func (m *ChatCompletionsModule) handleRequest(
	router *modules.RouterModule,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
) error {
	providers, model := router.ResolveProvidersOrderAndModel(prog.GetModel())

	m.logger.Debug("Resolved providers",
		zap.String("model", model),
		zap.Strings("providers", providers),
		zap.Int("plugin_count", len(chain.GetPlugins())))

	var displayErr error
	for _, name := range providers {
		m.logger.Debug("Trying provider", zap.String("provider", name))

		p, ok := router.ProviderConfigs[name]
		if !ok {
			m.logger.Error("provider not found", zap.String("name", name))
			continue
		}

		cmd, ok := p.Impl.Commands["inference"].(drivers.InferenceCommand)
		if !ok {
			m.logger.Debug("Provider does not support inference", zap.String("provider", name))
			continue
		}

		// Clone the program and set the resolved model
		providerProg := prog.Clone()
		providerProg.SetModel(model)

		// Run before plugins with provider context
		processedProg, err := chain.RunBefore(&p.Impl, r, providerProg)
		if err != nil {
			m.logger.Error("plugin before hook error", zap.String("provider", name), zap.Error(err))
			if displayErr == nil {
				displayErr = err
			}
			continue
		}
		providerProg = processedProg

		// Sample the upstream-prepared AIL (after model resolve + plugins)
		if hash, ok := r.Context().Value(ctxKeySampleHash).(string); ok {
			trySampleAILUpstream(hash, providerProg, m.logger)
		}

		m.logger.Debug("Executing inference",
			zap.String("provider", name),
			zap.String("style", string(p.Impl.Style)),
			zap.Bool("streaming", providerProg.IsStreaming()))

		// Success - set response headers
		w.Header().Set("X-Real-Provider-Id", name)
		w.Header().Set("X-Real-Model-Id", model)

		// Build plugin list for header
		var pluginNames []string
		for _, pi := range chain.GetPlugins() {
			pname := pi.Plugin.Name()
			if pi.Params != "" {
				pname += ":" + pi.Params
			}
			pluginNames = append(pluginNames, pname)
		}
		w.Header().Set("X-Plugins-Executed", strings.Join(pluginNames, ","))

		if providerProg.IsStreaming() {
			err = m.serveChatCompletionsStream(p, cmd, chain, providerProg, w, r)
		} else {
			err = m.serveChatCompletions(p, cmd, chain, providerProg, w, r)
		}

		if err != nil {
			if displayErr == nil {
				displayErr = err
			}
			continue
		}

		return nil
	}

	if displayErr != nil {
		return displayErr
	}

	return nil
}

// trySampleAIL persists the AIL program to sampleAILDir when SAMPLE_AIL is set.
// Files are keyed by the SHA-256 of the raw request body so duplicates are
// deduplicated automatically. Each request produces up to 6 files:
//   - <hash>.ail         – compact binary encoding of the original request
//   - <hash>.ail.txt     – human-readable disassembly of the original request
//   - <hash>.up.ail      – compact binary encoding of the upstream-prepared request
//   - <hash>.up.ail.txt  – human-readable disassembly of the upstream-prepared request
//   - <hash>.res.ail     – compact binary encoding of the response
//   - <hash>.res.ail.txt – human-readable disassembly of the response
//
// Returns the hex hash so callers can pair upstream/response samples with the same key.
func trySampleAIL(reqBody []byte, prog *ail.Program, logger *zap.Logger) string {
	if sampleAILDir == "" {
		return ""
	}

	// Ensure the directory exists (once per unique path, mkdir is idempotent)
	if err := os.MkdirAll(sampleAILDir, 0o755); err != nil {
		logger.Error("SAMPLE_AIL: failed to create directory", zap.String("dir", sampleAILDir), zap.Error(err))
		return ""
	}

	hash := sha256.Sum256(reqBody)
	name := hex.EncodeToString(hash[:])

	// Binary encoding
	binPath := filepath.Join(sampleAILDir, name+".ail")
	if _, err := os.Stat(binPath); err == nil {
		// Already sampled this exact request — still return name for response pairing
		return name
	}

	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		logger.Error("SAMPLE_AIL: binary encode failed", zap.Error(err))
		return name
	}
	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write binary failed", zap.String("path", binPath), zap.Error(err))
		return name
	}

	// Human-readable disassembly
	txtPath := filepath.Join(sampleAILDir, name+".ail.txt")
	if err := os.WriteFile(txtPath, []byte(prog.Disasm()), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write disasm failed", zap.String("path", txtPath), zap.Error(err))
		return name
	}

	logger.Debug("SAMPLE_AIL: saved request", zap.String("hash", name), zap.String("dir", sampleAILDir))
	return name
}

// trySampleAILUpstream persists the fully-prepared upstream request AIL program
// (after model resolution and all before-plugins have run). Files are written as:
//   - <hash>.up.ail      – compact binary encoding of the upstream request
//   - <hash>.up.ail.txt  – human-readable disassembly of the upstream request
func trySampleAILUpstream(reqHash string, prog *ail.Program, logger *zap.Logger) {
	if sampleAILDir == "" || reqHash == "" || prog == nil {
		return
	}

	binPath := filepath.Join(sampleAILDir, reqHash+".up.ail")

	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		logger.Error("SAMPLE_AIL: upstream binary encode failed", zap.Error(err))
		return
	}
	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write upstream binary failed", zap.String("path", binPath), zap.Error(err))
		return
	}

	txtPath := filepath.Join(sampleAILDir, reqHash+".up.ail.txt")
	if err := os.WriteFile(txtPath, []byte(prog.Disasm()), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write upstream disasm failed", zap.String("path", txtPath), zap.Error(err))
		return
	}

	logger.Debug("SAMPLE_AIL: saved upstream request", zap.String("hash", reqHash), zap.String("dir", sampleAILDir))
}

// trySampleAILResponse persists a response AIL program paired with the request
// that produced it. Files are written as:
//   - <hash>.res.ail      – compact binary encoding of the response
//   - <hash>.res.ail.txt  – human-readable disassembly of the response
func trySampleAILResponse(reqHash string, prog *ail.Program, logger *zap.Logger) {
	if sampleAILDir == "" || reqHash == "" || prog == nil {
		return
	}

	binPath := filepath.Join(sampleAILDir, reqHash+".res.ail")

	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		logger.Error("SAMPLE_AIL: response binary encode failed", zap.Error(err))
		return
	}
	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write response binary failed", zap.String("path", binPath), zap.Error(err))
		return
	}

	txtPath := filepath.Join(sampleAILDir, reqHash+".res.ail.txt")
	if err := os.WriteFile(txtPath, []byte(prog.Disasm()), 0o644); err != nil {
		logger.Error("SAMPLE_AIL: write response disasm failed", zap.String("path", txtPath), zap.Error(err))
		return
	}

	logger.Debug("SAMPLE_AIL: saved response", zap.String("hash", reqHash), zap.String("dir", sampleAILDir))
}

var (
	_ caddy.Provisioner           = (*ChatCompletionsModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*ChatCompletionsModule)(nil)
)
