// Package modules provides Caddy v2 HTTP handler modules for AI routing.
package server

import (
	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
)

func init() {
	caddy.RegisterModule(&ListModelsModule{})
	httpcaddyfile.RegisterHandlerDirective("ai_list_models", ParseListModelsModule)
	httpcaddyfile.RegisterDirectiveOrder("ai_list_models", httpcaddyfile.Before, "header")

	caddy.RegisterModule(&InferenceAILModule{})
	httpcaddyfile.RegisterHandlerDirective("ai_inference_ail", ParseInferenceAILModule)
	httpcaddyfile.RegisterDirectiveOrder("ai_inference_ail", httpcaddyfile.Before, "header")

	caddy.RegisterModule(&InferenceSseModule{})
	httpcaddyfile.RegisterHandlerDirective("ai_inference_sse", ParseInferenceSseModule)
	httpcaddyfile.RegisterDirectiveOrder("ai_inference_sse", httpcaddyfile.Before, "header")
}
