package services

import (
	"net/url"

	"github.com/neutrome-labs/ail"
)

// ProviderService provides the runtime implementation for a provider
type ProviderService struct {
	Name      string
	ParsedURL url.URL
	Style     ail.Style
	Router    *RouterService
	Commands  map[string]any

	// ExportedModels restricts which models this provider exposes externally.
	// When nil or empty, all models are available (no filtering).
	// When set, only listed model IDs are returned by /models and accepted
	// for direct inference. Virtual providers bypass this check on their
	// target providers.
	ExportedModels map[string]bool

	// Private marks a provider as completely hidden from external access.
	// A private provider exports no models and rejects all direct inference.
	// It can only be used as an upstream target for virtual providers.
	Private bool
}

// IsModelExported returns true if the given model is allowed by the exports
// filter. Returns true unconditionally when no exports are configured.
// Always returns false for private providers.
func (p *ProviderService) IsModelExported(model string) bool {
	if p.Private {
		return false
	}
	if len(p.ExportedModels) == 0 {
		return true
	}
	return p.ExportedModels[model]
}
