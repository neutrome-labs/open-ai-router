package plugin

import (
	"fmt"
	"net/http"

	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/neutrome-labs/ail"
)

// CaddyModuleInvoker invokes Caddy HTTP modules as plugins.
// Uses context to pass AIL programs without re-serialization.
type CaddyModuleInvoker struct {
	module caddyhttp.MiddlewareHandler
}

func NewCaddyModuleInvoker(module caddyhttp.MiddlewareHandler) *CaddyModuleInvoker {
	return &CaddyModuleInvoker{
		module: module,
	}
}

// InvokeHandler invokes the handler with an AIL program stored in request context.
func (inv *CaddyModuleInvoker) InvokeHandler(prog *ail.Program, w http.ResponseWriter, r *http.Request) error {
	// Store the program in context so the handler can pick it up without re-parsing
	newR := r.Clone(r.Context())
	newR = newR.WithContext(ail.ContextWithProgram(newR.Context(), prog))
	return inv.module.ServeHTTP(w, newR, nil)
}

// InvokeHandlerCapture invokes the handler and captures the response.
// Currently a stub — parallel plugin is disabled in the AIL rework.
func (inv *CaddyModuleInvoker) InvokeHandlerCapture(prog *ail.Program, r *http.Request) (*ail.Program, error) {
	return nil, fmt.Errorf("InvokeHandlerCapture not yet implemented in AIL mode")
}
