package plugin

import (
	"net/url"
	"strings"
)

// HeadPlugins are plugins that are always executed before others.
var HeadPlugins = [][2]string{}

// TailPlugins are plugins that are always executed after others.
var TailPlugins = [][2]string{}

func TryResolvePlugins(url url.URL, model string) *PluginChain {
	chain := NewPluginChain()

	// Add all virtual provider plugins (model rewriters).
	// They run via RunModelRewrite before the main plugin flow.
	for name, p := range Registry {
		if strings.HasPrefix(name, "virtual:") {
			if _, ok := p.(ModelRewritePlugin); ok {
				chain.Add(p, "")
			}
		}
	}

	// Add mandatory plugins
	for _, mp := range HeadPlugins {
		if p, ok := GetPlugin(mp[0]); ok {
			chain.Add(p, mp[1])
		}
	}

	// Plugins from path: /plugin1:arg1/plugin2:arg2
	path := strings.TrimPrefix(url.Path, "/")
	if path != "" {
		pathParts := strings.Split(path, "/")
		for _, part := range pathParts {
			if part == "" {
				continue
			}
			if idx := strings.IndexByte(part, ':'); idx > 0 {
				if p, ok := GetPlugin(part[:idx]); ok {
					chain.Add(p, part[idx+1:])
				}
			} else if part != "" {
				if p, ok := GetPlugin(part); ok {
					chain.Add(p, "")
				}
			}
		}
	}

	// Plugins from model suffix: model="gpt-4+plugin1:arg1+plugin2"
	if idx := strings.IndexByte(model, '+'); idx >= 0 {
		pluginPart := model[idx+1:]
		for len(pluginPart) > 0 {
			var part string
			if nextIdx := strings.IndexByte(pluginPart, '+'); nextIdx >= 0 {
				part = pluginPart[:nextIdx]
				pluginPart = pluginPart[nextIdx+1:]
			} else {
				part = pluginPart
				pluginPart = ""
			}
			if part == "" {
				continue
			}
			if colonIdx := strings.IndexByte(part, ':'); colonIdx >= 0 {
				name := part[:colonIdx]
				if name != "" {
					if p, ok := GetPlugin(name); ok {
						chain.Add(p, part[colonIdx+1:])
					}
				}
			} else {
				if p, ok := GetPlugin(part); ok {
					chain.Add(p, "")
				}
			}
		}
	}

	// Add tail plugins
	for _, mp := range TailPlugins {
		if p, ok := GetPlugin(mp[0]); ok {
			chain.Add(p, mp[1])
		}
	}

	return chain
}
