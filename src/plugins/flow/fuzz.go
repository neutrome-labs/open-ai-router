package flow

import (
	"context"
	"net/http"
	"strings"
	"sync"

	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"go.uber.org/zap"
)

// Fuzz provides fuzzy model name matching via ModelRewrite.
// It waterfalls over all provisioned providers, optimistically
// trying list_models on each, and returns the first match whose
// ID contains the requested partial name.
type Fuzz struct {
	cache sync.Map // providerName → []string
}

func (f *Fuzz) Name() string { return "fuzz" }

// RewriteModel tries to fuzzy-match the model across all providers.
// Supports both "provider/model" (scoped) and bare "model" (waterfall).
func (f *Fuzz) RewriteModel(model string) (string, bool) {
	// Separate plugin suffixes (+fuzz+logger etc.)
	base, suffix := model, ""
	if i := strings.IndexByte(model, '+'); i >= 0 {
		base, suffix = model[:i], model[i:]
	}

	// If there's a provider prefix, scope to that provider only.
	if i := strings.Index(base, "/"); i >= 0 {
		provider := strings.ToLower(base[:i])
		partial := base[i+1:]
		if matched, ok := f.tryMatch(provider, partial); ok {
			return provider + "/" + matched + suffix, true
		}
		return model, false
	}

	// No prefix — waterfall over all providers.
	if plugin.ProviderLister == nil {
		return model, false
	}
	for _, p := range plugin.ProviderLister() {
		if matched, ok := f.tryMatch(p.Name, base); ok {
			return p.Name + "/" + matched + suffix, true
		}
	}
	return model, false
}

// tryMatch checks if partial fuzzy-matches a model from the given provider.
// Returns ("", false) on exact match to prevent infinite rewrite loops.
func (f *Fuzz) tryMatch(providerName, partial string) (string, bool) {
	models := f.getModels(providerName)
	for _, m := range models {
		if m == partial {
			return "", false // exact → no rewrite
		}
	}
	for _, m := range models {
		if strings.Contains(m, partial) {
			plugin.Logger.Debug("fuzz matched",
				zap.String("provider", providerName),
				zap.String("partial", partial),
				zap.String("resolved", m))
			return m, true
		}
	}
	return "", false
}

// getModels returns cached model IDs, lazily fetching via list_models.
func (f *Fuzz) getModels(providerName string) []string {
	if cached, ok := f.cache.Load(providerName); ok {
		return cached.([]string)
	}

	if plugin.ProviderLister == nil {
		return nil
	}

	for _, p := range plugin.ProviderLister() {
		if p.Name != providerName {
			continue
		}
		cmd, ok := p.Commands["list_models"].(drivers.ListModelsCommand)
		if !ok {
			return nil
		}

		req, _ := http.NewRequestWithContext(context.Background(), "GET", "/", nil)
		listed, err := cmd.DoListModels(p, req)
		if err != nil {
			plugin.Logger.Debug("fuzz: list_models failed",
				zap.String("provider", providerName),
				zap.Error(err))
			return nil
		}

		ids := make([]string, len(listed))
		for i, m := range listed {
			ids[i] = m.ID
		}
		f.cache.Store(providerName, ids)
		return ids
	}
	return nil
}

var _ plugin.ModelRewritePlugin = (*Fuzz)(nil)
