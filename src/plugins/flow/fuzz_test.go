package flow

import (
	"testing"

	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

func newTestFuzz(providerModels map[string][]string) *Fuzz {
	f := &Fuzz{}
	// Pre-populate cache so tests don't need real HTTP calls.
	for provider, models := range providerModels {
		f.cache.Store(provider, models)
	}
	// Wire up a fake ProviderLister so waterfall works.
	var providers []*services.ProviderService
	for name := range providerModels {
		providers = append(providers, &services.ProviderService{Name: name})
	}
	plugin.ProviderLister = func() []*services.ProviderService { return providers }
	return f
}

func TestFuzz_ExactMatch_NoRewrite(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai": {"gpt-4", "gpt-4-0613", "gpt-3.5-turbo"},
	})
	rewritten, matched := f.RewriteModel("openai/gpt-4")
	if matched {
		t.Errorf("expected no rewrite for exact match, got %q", rewritten)
	}
}

func TestFuzz_PartialMatch_WithPrefix(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai": {"gpt-4-0613", "gpt-4-turbo", "gpt-3.5-turbo"},
	})
	rewritten, matched := f.RewriteModel("openai/gpt-4")
	if !matched {
		t.Fatal("expected fuzz to match")
	}
	if rewritten != "openai/gpt-4-0613" {
		t.Errorf("expected 'openai/gpt-4-0613', got %q", rewritten)
	}
}

func TestFuzz_Waterfall_NoPrefix(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai":    {"gpt-4-0613"},
		"anthropic": {"claude-3-opus"},
	})
	rewritten, matched := f.RewriteModel("claude-3")
	if !matched {
		t.Fatal("expected fuzz to match via waterfall")
	}
	if rewritten != "anthropic/claude-3-opus" {
		t.Errorf("expected 'anthropic/claude-3-opus', got %q", rewritten)
	}
}

func TestFuzz_PreservesPluginSuffix(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai": {"gpt-4-0613", "gpt-3.5-turbo"},
	})
	rewritten, matched := f.RewriteModel("openai/gpt-4+fuzz+logger")
	if !matched {
		t.Fatal("expected fuzz to match")
	}
	if rewritten != "openai/gpt-4-0613+fuzz+logger" {
		t.Errorf("expected 'openai/gpt-4-0613+fuzz+logger', got %q", rewritten)
	}
}

func TestFuzz_UnknownProvider_Skips(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai": {"gpt-4-0613"},
	})
	rewritten, matched := f.RewriteModel("nonexistent/gpt-4")
	if matched {
		t.Errorf("expected no match for unknown provider, got %q", rewritten)
	}
}

func TestFuzz_NoMatchingModel(t *testing.T) {
	f := newTestFuzz(map[string][]string{
		"openai": {"gpt-4-0613", "gpt-3.5-turbo"},
	})
	rewritten, matched := f.RewriteModel("openai/nonexistent")
	if matched {
		t.Errorf("expected no match, got %q", rewritten)
	}
}
