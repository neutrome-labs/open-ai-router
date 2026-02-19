package drivers

import (
	"net/http"

	"github.com/neutrome-labs/open-ai-router/src/services"
)

// ExportFilteredListModels wraps a ListModelsCommand and filters its results
// to only include models that appear in the provider's ExportedModels set.
// When ExportedModels is nil or empty the inner command's results pass through
// untouched.
//
// This wrapper is applied during provider provisioning so that every consumer
// (the /models endpoint, the fuzz plugin, etc.) automatically sees only the
// exported models without any extra logic.
type ExportFilteredListModels struct {
	Inner ListModelsCommand
}

// DoListModels delegates to the inner command and filters the result set.
// For private providers it returns an empty list without calling upstream.
func (f *ExportFilteredListModels) DoListModels(p *services.ProviderService, r *http.Request) ([]ListModelsModel, error) {
	// Private providers expose nothing.
	if p.Private {
		return []ListModelsModel{}, nil
	}

	models, err := f.Inner.DoListModels(p, r)
	if err != nil {
		return nil, err
	}

	if len(p.ExportedModels) == 0 {
		return models, nil
	}

	filtered := make([]ListModelsModel, 0, len(p.ExportedModels))
	for _, m := range models {
		if p.ExportedModels[m.ID] {
			filtered = append(filtered, m)
		}
	}
	return filtered, nil
}

var _ ListModelsCommand = (*ExportFilteredListModels)(nil)
