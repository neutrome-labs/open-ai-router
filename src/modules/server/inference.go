package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/modules"
	"github.com/neutrome-labs/open-ai-router/src/plugin"

	"github.com/neutrome-labs/ail"
	"go.uber.org/zap"
)

// exportsCheckBypassedKey is the context key set by RequestPreamble when a
// virtual provider rewrites the model. When present the exports gate in
// RunInferencePipeline is skipped so virtual providers can target any model.
type exportsCheckBypassedKey struct{}

// InferenceHandler provides module-specific inference serving.
// Each endpoint module (ChatCompletions, AIL, ...) implements this to
// control how non-streaming and streaming responses are written.
type InferenceHandler interface {
	ServeNonStreaming(
		p *modules.ProviderConfig,
		cmd drivers.InferenceCommand,
		chain *plugin.PluginChain,
		prog *ail.Program,
		w http.ResponseWriter,
		r *http.Request,
	) error

	ServeStreaming(
		p *modules.ProviderConfig,
		cmd drivers.InferenceCommand,
		chain *plugin.PluginChain,
		prog *ail.Program,
		w http.ResponseWriter,
		r *http.Request,
	) error
}

// RunInferencePipeline executes the common provider iteration loop used by
// every endpoint module. It resolves providers, iterates them in order,
// runs before-plugins, samples AIL, sets response headers, builds X-Plugins-Executed,
// and delegates the actual inference call to the InferenceHandler.
func RunInferencePipeline(
	router *modules.RouterModule,
	chain *plugin.PluginChain,
	prog *ail.Program,
	w http.ResponseWriter,
	r *http.Request,
	handler InferenceHandler,
	logger *zap.Logger,
) error {
	providers, model := router.ResolveProvidersOrderAndModel(prog.GetModel())

	logger.Debug("Resolved providers",
		zap.String("model", model),
		zap.Strings("providers", providers),
		zap.Int("plugin_count", len(chain.GetPlugins())))

	var displayErr error
	bypassExports, _ := r.Context().Value(exportsCheckBypassedKey{}).(bool)
	modelNotExported := false

	for _, name := range providers {
		logger.Debug("Trying provider", zap.String("provider", name))

		p, ok := router.ProviderConfigs[name]
		if !ok {
			logger.Error("provider not found", zap.String("name", name))
			continue
		}

		// Check exports filter. When a virtual provider rewrote the model
		// (exportsCheckBypassed is true) we skip this gate so that virtual
		// aliases can target non-exported models.
		if !bypassExports && !p.Impl.IsModelExported(model) {
			logger.Debug("Model not exported by provider, skipping",
				zap.String("provider", name),
				zap.String("model", model))
			modelNotExported = true
			continue
		}

		cmd, ok := p.Impl.Commands["inference"].(drivers.InferenceCommand)
		if !ok {
			logger.Debug("Provider does not support inference", zap.String("provider", name))
			continue
		}

		// Clone the program and set the resolved model.
		providerProg := prog.Clone()
		providerProg.SetModel(model)

		// Run before plugins with provider context.
		processedProg, err := chain.RunBefore(&p.Impl, r, providerProg)
		if err != nil {
			logger.Error("plugin before hook error",
				zap.String("provider", name), zap.Error(err))
			if displayErr == nil {
				displayErr = err
			}
			continue
		}
		providerProg = processedProg

		logger.Debug("Executing inference",
			zap.String("provider", name),
			zap.String("style", string(p.Impl.Style)),
			zap.Bool("streaming", providerProg.IsStreaming()))

		// Set response headers.
		w.Header().Set("X-Real-Provider-Id", name)
		w.Header().Set("X-Real-Model-Id", model)

		// Build X-Plugins-Executed header.
		var pluginNames []string
		for _, pi := range chain.GetPlugins() {
			pname := pi.Plugin.Name()
			if pi.Params != "" {
				pname += ":" + pi.Params
			}
			pluginNames = append(pluginNames, pname)
		}
		if len(pluginNames) > 0 {
			w.Header().Set("X-Plugins-Executed", strings.Join(pluginNames, ","))
		}

		// Dispatch to module-specific handler.
		if providerProg.IsStreaming() {
			err = handler.ServeStreaming(p, cmd, chain, providerProg, w, r)
		} else {
			err = handler.ServeNonStreaming(p, cmd, chain, providerProg, w, r)
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

	// If every candidate provider was skipped because of exports filtering,
	// emit a proper model-not-found JSON error so the client sees a clear
	// 404 rather than an empty response.
	if modelNotExported {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": fmt.Sprintf("The model `%s` does not exist or you do not have access to it.", model),
				"type":    "invalid_request_error",
				"param":   nil,
				"code":    "model_not_found",
			},
		})
		return nil
	}

	return nil
}

// RequestPreamble performs the common request setup shared by all endpoint
// modules: auth collection, virtual model aliasing, plugin resolution, and
// trace ID generation.
func RequestPreamble(
	router *modules.RouterModule,
	prog *ail.Program,
	r *http.Request,
	logger *zap.Logger,
) (*plugin.PluginChain, *http.Request, error) {
	// Collect incoming auth.
	r, err := router.Impl.Auth.CollectIncomingAuth(r)
	if err != nil {
		logger.Error("failed to collect incoming auth", zap.Error(err))
		return nil, r, err
	}

	// Resolve virtual model aliases (may chain: virtual→virtual→real).
	model := prog.GetModel()
	var chain *plugin.PluginChain
	virtualResolved := false
	const maxRewriteDepth = 10
	for i := 0; i < maxRewriteDepth; i++ {
		chain = plugin.TryResolvePlugins(*r.URL, model)
		rewritten, rewriter := chain.RunModelRewrite(model)
		if rewritten != model {
			logger.Debug("Virtual model resolved",
				zap.String("from", model),
				zap.String("to", rewritten),
				zap.String("rewriter", rewriter))
			// Track if a virtual provider plugin did the rewrite so we can
			// bypass the exports check later (virtual aliases may target
			// non-exported models).
			if strings.HasPrefix(rewriter, "virtual:") {
				virtualResolved = true
			}
			model = rewritten
			continue
		}
		break
	}
	prog.SetModel(model)

	// When a virtual provider rewrote the model, store a flag in the
	// request context so RunInferencePipeline skips exports filtering.
	if virtualResolved {
		r = r.WithContext(context.WithValue(r.Context(), exportsCheckBypassedKey{}, true))
	}

	logger.Debug("Resolved plugins", zap.Int("plugin_count", len(chain.GetPlugins())))

	return chain, r, nil
}
