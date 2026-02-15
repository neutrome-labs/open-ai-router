package styles

import (
	"fmt"

	"github.com/neutrome-labs/ail"
	"go.uber.org/zap"
)

// Logger for debug output (set by module during Provision)
var Logger *zap.Logger = zap.NewNop()

// Style is an alias for ail.Style so the rest of the router code keeps working
// without importing `ail` directly. When ail is extracted to its own module,
// this alias will point to the external package.
type Style = ail.Style

// Re-export style constants from the ail package — single source of truth.
// StyleVirtual is router-only: it has no upstream provider, so AIL doesn't
// (and shouldn't) define a parser/emitter for it.
const (
	StyleUnknown         Style = ""
	StyleVirtual         Style = "virtual"
	StyleChatCompletions       = ail.StyleChatCompletions
	StyleResponses             = ail.StyleResponses
	StyleAnthropic             = ail.StyleAnthropic
	StyleGoogleGenAI           = ail.StyleGoogleGenAI
	StyleCfAiGateway           = ail.StyleCfAiGateway
	StyleCfWorkersAi           = ail.StyleCfWorkersAi
)

// ParseStyle parses a style string, defaulting to OpenAI chat completions.
// This is the router-level parser that knows about all styles including
// router-only ones like "virtual".
func ParseStyle(s string) (Style, error) {
	switch s {
	case "virtual":
		return StyleVirtual, nil
	case "openai-chat-completions", "openai", "":
		return StyleChatCompletions, nil
	case "openai-responses", "responses":
		return StyleResponses, nil
	/*case "anthropic-messages", "anthropic":
		return StyleAnthropic, nil
	case "google-genai", "google":
		return StyleGoogleGenAI, nil
	case "cloudflare-ai-gateway":
		return StyleCfAiGateway, nil
	case "cloudflare-workers-ai", "cloudflare", "cf":
		return StyleCfWorkersAi, nil*/
	default:
		return StyleUnknown, fmt.Errorf("unknown style: %s", s)
	}
}
