package modules

import (
	"context"
	"net/http"
	"os"
	"strings"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// EnvAuthModule provides authentication from environment variables.
type EnvAuthModule struct {
	Name   string `json:"name,omitempty"`
	logger *zap.Logger
}

func ParseEnvAuthModule(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var m EnvAuthModule
	for h.Next() {
		for h.NextBlock(0) {
			switch h.Val() {
			case "name":
				if !h.NextArg() {
					return nil, h.ArgErr()
				}
				m.Name = h.Val()
				if m.Name == "" {
					m.Name = "default"
				}
			default:
				return nil, h.Errf("unrecognized ai_auth_env option '%s'", h.Val())
			}
		}
	}
	return &m, nil
}

func (*EnvAuthModule) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.ai_auth_env",
		New: func() caddy.Module { return new(EnvAuthModule) },
	}
}

func (m *EnvAuthModule) Provision(ctx caddy.Context) error {
	m.logger = ctx.Logger(m)
	if m.Name == "" {
		m.Name = "default"
	}
	services.RegisterAuthService(m.Name, m)
	m.logger.Info("Registered env auth manager", zap.String("name", m.Name))
	return nil
}

func (m *EnvAuthModule) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	return next.ServeHTTP(w, r)
}

// CollectIncomingAuth is called early in request handling to set up context values.
// For EnvAuthManager, this is a no-op since we don't have user-specific auth from the incoming request.
// The context values are set later in CollectTargetAuth when we know which provider is being used.
func (m *EnvAuthModule) CollectIncomingAuth(r *http.Request) (*http.Request, error) {
	return r, nil
}

func (m *EnvAuthModule) CollectTargetAuth(scope string, p *services.ProviderService, rIn, rOut *http.Request) (string, error) {
	// Try multiple environment variable patterns
	patterns := []string{
		strings.ToUpper(p.Name) + "_KEY",
		strings.ToUpper(p.Name) + "_API_KEY",
		strings.ToUpper(strings.ReplaceAll(p.Name, "-", "_")) + "_KEY",
		strings.ToUpper(strings.ReplaceAll(p.Name, "-", "_")) + "_API_KEY",
	}

	var key string
	for _, pattern := range patterns {
		key = os.Getenv(pattern)
		if key != "" {
			break
		}
	}

	if key == "" {
		m.logger.Warn("no key found in environment variables for provider",
			zap.String("provider", p.Name),
			zap.Strings("tried_patterns", patterns))
		return "", nil
	}

	ctx := context.WithValue(rIn.Context(), plugin.ContextKeyID(), "env:"+p.Name)
	ctx = context.WithValue(ctx, plugin.ContextUserID(), "env:"+p.Name)
	*rIn = *rIn.WithContext(ctx)

	return key, nil
}

var (
	_ caddy.Provisioner           = (*EnvAuthModule)(nil)
	_ caddyhttp.MiddlewareHandler = (*EnvAuthModule)(nil)
	_ services.AuthService        = (*EnvAuthModule)(nil)
)
