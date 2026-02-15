package plugin_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/neutrome-labs/ail"
	_ "github.com/neutrome-labs/open-ai-router/src/modules"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
)

func TestPluginRegistry(t *testing.T) {
	// The showcase fuzz plugin should be registered via modules/init.go
	p, ok := plugin.GetPlugin("fuzz")
	if !ok {
		t.Fatal("Plugin \"fuzz\" not found in registry")
	}
	if p.Name() != "fuzz" {
		t.Errorf("Plugin name mismatch: got %q, want %q", p.Name(), "fuzz")
	}
}

func TestPluginChain_Add(t *testing.T) {
	chain := plugin.NewPluginChain()

	p, _ := plugin.GetPlugin("fuzz")
	chain.Add(p, "test-param")

	if len(chain.GetPlugins()) != 1 {
		t.Errorf("Expected 1 plugin, got %d", len(chain.GetPlugins()))
	}

	if chain.GetPlugins()[0].Params != "test-param" {
		t.Errorf("Params wrong: %s", chain.GetPlugins()[0].Params)
	}
}

func TestPluginChain_RunBefore(t *testing.T) {
	chain := plugin.NewPluginChain()

	p, _ := plugin.GetPlugin("fuzz")
	chain.Add(p, "")

	prog := ail.NewProgram()
	prog.EmitString(ail.SET_MODEL, "gpt-4")
	prog.Emit(ail.MSG_START)
	prog.Emit(ail.ROLE_USR)
	prog.EmitString(ail.TXT_CHUNK, "Hello")
	prog.Emit(ail.MSG_END)

	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	provider := &services.ProviderService{Name: "test"}

	resultProg, err := chain.RunBefore(provider, httpReq, prog)
	if err != nil {
		t.Fatalf("RunBefore failed: %v", err)
	}

	if resultProg.GetModel() != "gpt-4" {
		t.Errorf("Model wrong: %s", resultProg.GetModel())
	}
}

func TestPluginChain_RunAfter(t *testing.T) {
	chain := plugin.NewPluginChain()

	p, _ := plugin.GetPlugin("fuzz")
	chain.Add(p, "")

	reqProg := ail.NewProgram()
	reqProg.EmitString(ail.SET_MODEL, "gpt-4")

	resProg := ail.NewProgram()
	resProg.EmitString(ail.SET_MODEL, "gpt-4")
	resProg.Emit(ail.MSG_START)
	resProg.Emit(ail.ROLE_AST)
	resProg.EmitString(ail.TXT_CHUNK, "Hi")
	resProg.Emit(ail.MSG_END)

	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	httpResp := &http.Response{StatusCode: 200}
	provider := &services.ProviderService{Name: "test"}

	resultProg, err := chain.RunAfter(provider, httpReq, reqProg, httpResp, resProg)
	if err != nil {
		t.Fatalf("RunAfter failed: %v", err)
	}

	if resultProg.GetModel() != "gpt-4" {
		t.Errorf("Model wrong: %s", resultProg.GetModel())
	}
}

func TestMandatoryPlugins(t *testing.T) {
	// HeadPlugins and TailPlugins are empty during AIL rework
	if len(plugin.HeadPlugins) != 0 {
		t.Errorf("Expected empty HeadPlugins, got %d", len(plugin.HeadPlugins))
	}
	if len(plugin.TailPlugins) != 0 {
		t.Errorf("Expected empty TailPlugins, got %d", len(plugin.TailPlugins))
	}
}
