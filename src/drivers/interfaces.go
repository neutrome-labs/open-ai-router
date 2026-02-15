// Package drivers provides command interfaces for provider interactions.
package drivers

import (
	"net/http"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

// ListModelsModel represents a model from a provider
type ListModelsModel struct {
	Object  string `json:"object,omitempty"`
	ID      string `json:"id,omitempty"`
	Name    string `json:"name,omitempty"`
	Created int64  `json:"created,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

// ListModelsCommand lists available models from a provider
type ListModelsCommand interface {
	DoListModels(p *services.ProviderService, r *http.Request) ([]ListModelsModel, error)
}

// InferenceStreamChunk represents a streaming response chunk as an AIL program fragment.
type InferenceStreamChunk struct {
	Data         *ail.Program
	RuntimeError error
}

// InferenceCommand is the unified interface for all inference APIs.
// Each driver takes an AIL Program, converts it to the provider's native format
// internally, makes the HTTP call, and parses the response back into an AIL Program.
type InferenceCommand interface {
	// DoInference sends a non-streaming inference request.
	// Takes an AIL program, returns the response as an AIL program.
	DoInference(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, *ail.Program, error)
	// DoInferenceStream sends a streaming inference request.
	// Returns a channel of AIL program chunks.
	DoInferenceStream(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, chan InferenceStreamChunk, error)
}
