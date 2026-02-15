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
}
