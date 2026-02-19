package plugin

import (
	"net/http"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// PluginChain manages the execution of plugins
type PluginChain struct {
	plugins []PluginInstance
}

// NewPluginChain creates a new plugin chain
func NewPluginChain() *PluginChain {
	return &PluginChain{
		plugins: make([]PluginInstance, 0),
	}
}

// Add adds a plugin to the chain
func (c *PluginChain) Add(p Plugin, params string) {
	c.plugins = append(c.plugins, PluginInstance{Plugin: p, Params: params})
}

// RunBefore executes all BeforePlugin implementations
func (c *PluginChain) RunBefore(p *services.ProviderService, r *http.Request, prog *ail.Program) (*ail.Program, error) {
	Logger.Debug("RunBefore starting", zap.Int("plugin_count", len(c.plugins)))
	current := prog
	for _, pi := range c.plugins {
		if bp, ok := pi.Plugin.(BeforePlugin); ok {
			Logger.Debug("Running Before plugin", zap.String("plugin", pi.Plugin.Name()), zap.String("params", pi.Params))
			next, err := bp.Before(pi.Params, p, r, current)
			if err != nil {
				Logger.Error("Before plugin failed", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
				return nil, err
			}
			current = next
		}
	}
	Logger.Debug("RunBefore completed")
	return current, nil
}

// RunAfter executes all AfterPlugin implementations
func (c *PluginChain) RunAfter(p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, resProg *ail.Program) (*ail.Program, error) {
	Logger.Debug("RunAfter starting", zap.Int("plugin_count", len(c.plugins)))
	current := resProg
	for _, pi := range c.plugins {
		if ap, ok := pi.Plugin.(AfterPlugin); ok {
			Logger.Debug("Running After plugin", zap.String("plugin", pi.Plugin.Name()), zap.String("params", pi.Params))
			next, err := ap.After(pi.Params, p, r, reqProg, res, current)
			if err != nil {
				Logger.Error("After plugin failed", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
				return nil, err
			}
			current = next
		}
	}
	Logger.Debug("RunAfter completed")
	return current, nil
}

// RunAfterChunk executes all StreamChunkPlugin implementations
func (c *PluginChain) RunAfterChunk(p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, chunk *ail.Program) (*ail.Program, error) {
	current := chunk
	for _, pi := range c.plugins {
		if sp, ok := pi.Plugin.(StreamChunkPlugin); ok {
			next, err := sp.AfterChunk(pi.Params, p, r, reqProg, res, current)
			if err != nil {
				Logger.Error("AfterChunk plugin failed", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
				return nil, err
			}
			current = next
		}
	}
	return current, nil
}

// RunStreamEnd executes all StreamEndPlugin implementations
func (c *PluginChain) RunStreamEnd(p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, lastChunk *ail.Program) error {
	Logger.Debug("RunStreamEnd starting", zap.Int("plugin_count", len(c.plugins)))
	for _, pi := range c.plugins {
		if sep, ok := pi.Plugin.(StreamEndPlugin); ok {
			Logger.Debug("Running StreamEnd plugin", zap.String("plugin", pi.Plugin.Name()), zap.String("params", pi.Params))
			if err := sep.StreamEnd(pi.Params, p, r, reqProg, res, lastChunk); err != nil {
				Logger.Error("StreamEnd plugin failed", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
				return err
			}
		}
	}
	Logger.Debug("RunStreamEnd completed")
	return nil
}

// RunError executes all ErrorPlugin implementations
func (c *PluginChain) RunError(p *services.ProviderService, r *http.Request, reqProg *ail.Program, res *http.Response, providerErr error) error {
	Logger.Debug("RunError starting", zap.Int("plugin_count", len(c.plugins)), zap.Error(providerErr))
	for _, pi := range c.plugins {
		if ep, ok := pi.Plugin.(ErrorPlugin); ok {
			Logger.Debug("Running Error plugin", zap.String("plugin", pi.Plugin.Name()), zap.String("params", pi.Params))
			if err := ep.OnError(pi.Params, p, r, reqProg, res, providerErr); err != nil {
				Logger.Error("Error plugin failed", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
			}
		}
	}
	Logger.Debug("RunError completed")
	return nil
}

// RunModelRewrite iterates ModelRewritePlugins and returns the first
// successful rewrite along with the name of the plugin that matched.
// Returns the original model and an empty name if nothing matched.
func (c *PluginChain) RunModelRewrite(model string) (string, string) {
	for _, pi := range c.plugins {
		if mr, ok := pi.Plugin.(ModelRewritePlugin); ok {
			if rewritten, matched := mr.RewriteModel(model); matched {
				Logger.Debug("ModelRewrite matched",
					zap.String("plugin", pi.Plugin.Name()),
					zap.String("from", model),
					zap.String("to", rewritten))
				return rewritten, pi.Plugin.Name()
			}
		}
	}
	return model, ""
}

// RunRecursiveHandlers executes all RecursiveHandlerPlugin implementations.
func (c *PluginChain) RunRecursiveHandlers(invoker HandlerInvoker, prog *ail.Program, w http.ResponseWriter, r *http.Request) (bool, error) {
	Logger.Debug("RunRecursiveHandlers starting", zap.Int("plugin_count", len(c.plugins)))
	for _, pi := range c.plugins {
		if rh, ok := pi.Plugin.(RecursiveHandlerPlugin); ok {
			Logger.Debug("Running RecursiveHandler plugin", zap.String("plugin", pi.Plugin.Name()), zap.String("params", pi.Params))
			handled, err := rh.RecursiveHandler(pi.Params, invoker, prog, w, r)
			if handled {
				if err != nil {
					Logger.Debug("RecursiveHandler plugin handled with error", zap.String("plugin", pi.Plugin.Name()), zap.Error(err))
				} else {
					Logger.Debug("RecursiveHandler plugin handled successfully", zap.String("plugin", pi.Plugin.Name()))
				}
				return true, err
			}
		}
	}
	Logger.Debug("RunRecursiveHandlers completed - no plugin handled")
	return false, nil
}

// GetPlugins returns all plugins in the chain
func (c *PluginChain) GetPlugins() []PluginInstance {
	return c.plugins
}
