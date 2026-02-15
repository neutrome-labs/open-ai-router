// Package modules provides Caddy v2 HTTP handler modules for AI routing.
package modules

import (
	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/plugins"
	"github.com/neutrome-labs/open-ai-router/src/plugins/flow"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

var APP_VERSION = "5.0.0"

func init() {
	services.TryInstrumentAppObservability()

	plugin.RegisterPlugin("fuzz", &flow.Fuzz{})
	plugin.RegisterPlugin("stools", &plugins.StripTools{})
	plugin.RegisterPlugin("slwin", &plugins.SlidingWindow{})

	defer func() {
		_ = services.FireObservabilityEvent("app", "", "init", map[string]any{
			"version": APP_VERSION,
		})
	}()

	caddy.RegisterModule(&EnvAuthModule{})
	httpcaddyfile.RegisterHandlerDirective("ai_auth_env", ParseEnvAuthModule)
	httpcaddyfile.RegisterDirectiveOrder("ai_auth_env", httpcaddyfile.Before, "header")

	caddy.RegisterModule(&RouterModule{})
	httpcaddyfile.RegisterHandlerDirective("ai_router", ParseRouterModule)
	httpcaddyfile.RegisterDirectiveOrder("ai_router", httpcaddyfile.Before, "header")
}
