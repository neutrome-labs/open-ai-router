// Package virtual provides a virtual driver for model aliasing and presets.
// Virtual providers don't connect to external APIs — they rewrite the model
// name so the request is routed to a real provider.
package virtual

import (
	"net/http"
	"strings"

	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// Logger for virtual driver - can be set by modules
var Logger *zap.Logger = zap.NewNop()

// VirtualPlugin implements ModelRewritePlugin for virtual providers.
// It rewrites virtual model names to their real targets before plugin
// resolution and provider routing.
type VirtualPlugin struct {
	// ProviderName is the name of this virtual provider
	ProviderName string
	// ModelMappings maps virtual model names to target model specs (e.g., "provider/model+plugins")
	ModelMappings map[string]string
}

// Name returns the plugin name
func (v *VirtualPlugin) Name() string {
	return "virtual:" + v.ProviderName
}

// RewriteModel checks whether the model targets this virtual provider and,
// if so, returns the mapped real model (preserving any user plugin suffixes).
func (v *VirtualPlugin) RewriteModel(model string) (string, bool) {
	// Expect "virtualProvider/modelName" or "virtualProvider/modelName+plugins"
	idx := strings.Index(model, "/")
	if idx < 0 {
		return model, false
	}
	providerPrefix := strings.ToLower(model[:idx])
	actualModel := model[idx+1:]

	if providerPrefix != v.ProviderName {
		return model, false
	}

	// Split base model from user plugin suffix
	baseModel := actualModel
	pluginSuffix := ""
	if plusIdx := strings.IndexByte(actualModel, '+'); plusIdx >= 0 {
		baseModel = actualModel[:plusIdx]
		pluginSuffix = actualModel[plusIdx:] // includes leading '+'
	}

	targetModel, ok := v.ModelMappings[baseModel]
	if !ok || targetModel == "" {
		return model, false
	}

	// Target plugins come first, then user plugins
	finalModel := targetModel + pluginSuffix

	Logger.Debug("VirtualPlugin resolved model",
		zap.String("provider", v.ProviderName),
		zap.String("from", model),
		zap.String("to", finalModel))

	return finalModel, true
}

// VirtualListModels implements ListModelsCommand for virtual providers.
type VirtualListModels struct {
	ProviderName  string
	ModelMappings map[string]string
}

// DoListModels returns the list of virtual models.
func (v *VirtualListModels) DoListModels(p *services.ProviderService, r *http.Request) ([]drivers.ListModelsModel, error) {
	Logger.Debug("VirtualListModels.DoListModels", zap.String("provider", p.Name))

	var models []drivers.ListModelsModel
	for modelName := range v.ModelMappings {
		models = append(models, drivers.ListModelsModel{
			Object:  "model",
			ID:      modelName,
			Name:    modelName,
			OwnedBy: v.ProviderName,
		})
	}

	return models, nil
}
