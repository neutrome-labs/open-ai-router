package dspy

import (
	"testing"
)

func TestParseParams_Defaults(t *testing.T) {
	kind, sig := parseParams("")
	if kind != "react" {
		t.Errorf("expected kind=react, got %q", kind)
	}
	if sig != "history, question -> answer" {
		t.Errorf("expected default signature, got %q", sig)
	}
}

func TestParseParams_KindOnly(t *testing.T) {
	kind, sig := parseParams("react")
	if kind != "react" {
		t.Errorf("expected kind=react, got %q", kind)
	}
	if sig != "history, question -> answer" {
		t.Errorf("expected default signature, got %q", sig)
	}
}

func TestParseParams_KindAndSignature(t *testing.T) {
	kind, sig := parseParams("cot:context,%20question%20->%20answer")
	if kind != "cot" {
		t.Errorf("expected kind=cot, got %q", kind)
	}
	expected := "context, question -> answer"
	if sig != expected {
		t.Errorf("expected sig=%q, got %q", expected, sig)
	}
}

func TestParseParams_EmptyKindExplicitSig(t *testing.T) {
	kind, sig := parseParams(":question%20->%20answer")
	if kind != "react" {
		t.Errorf("expected default kind=react, got %q", kind)
	}
	if sig != "question -> answer" {
		t.Errorf("expected sig=%q, got %q", "question -> answer", sig)
	}
}

func TestParseSignatureFields_Standard(t *testing.T) {
	in, out := parseSignatureFields("history, question -> answer")
	if len(in) != 2 || in[0] != "history" || in[1] != "question" {
		t.Errorf("unexpected inputs: %v", in)
	}
	if len(out) != 1 || out[0] != "answer" {
		t.Errorf("unexpected outputs: %v", out)
	}
}

func TestParseSignatureFields_WithTypes(t *testing.T) {
	in, out := parseSignatureFields("context: str, question: str -> answer: str, reasoning: str")
	if len(in) != 2 || in[0] != "context" || in[1] != "question" {
		t.Errorf("unexpected inputs: %v", in)
	}
	if len(out) != 2 || out[0] != "answer" || out[1] != "reasoning" {
		t.Errorf("unexpected outputs: %v", out)
	}
}

func TestParseSignatureFields_NoArrow(t *testing.T) {
	in, out := parseSignatureFields("question")
	if len(in) != 1 || in[0] != "question" {
		t.Errorf("unexpected inputs: %v", in)
	}
	if len(out) != 1 || out[0] != "answer" {
		t.Errorf("unexpected outputs: %v", out)
	}
}

func TestValidKinds(t *testing.T) {
	for _, k := range []string{"predict", "cot", "react", "rlm"} {
		if !validKinds[k] {
			t.Errorf("expected %q to be a valid kind", k)
		}
	}
	if validKinds["invalid"] {
		t.Error("expected 'invalid' to not be a valid kind")
	}
}

func TestStripDspySuffix(t *testing.T) {
	cases := []struct{ in, want string }{
		{"openai/gpt-4.1-mini+dspy", "openai/gpt-4.1-mini"},
		{"openai/gpt-4.1-mini+dspy:cot", "openai/gpt-4.1-mini"},
		{"openai/gpt-4.1-mini+dspy:cot:question%20->%20answer", "openai/gpt-4.1-mini"},
		{"gpt-4o", "gpt-4o"},
		{"openai/gpt-4o+stools+dspy:react", "openai/gpt-4o+stools"},
	}
	for _, tc := range cases {
		got := stripDspySuffix(tc.in)
		if got != tc.want {
			t.Errorf("stripDspySuffix(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
