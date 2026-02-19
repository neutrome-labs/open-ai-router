package plugins

import (
	"encoding/json"
	"testing"

	"github.com/neutrome-labs/ail"
)

// helper to build an AIL program from a message list specification.
// Each msg is described by: role ("system"|"user"|"assistant"|"tool"),
// optional text content, optional tool calls ([]string of call IDs),
// optional tool_call_id for tool results.
type testMsg struct {
	role       string
	text       string
	toolCalls  []string // call IDs for assistant tool_calls
	toolCallID string   // for tool-result messages
	resultData string   // for tool-result messages
}

func buildAILProgram(model string, msgs []testMsg) *ail.Program {
	p := ail.NewProgram()
	p.EmitString(ail.SET_MODEL, model)

	for _, m := range msgs {
		p.Emit(ail.MSG_START)
		switch m.role {
		case "system":
			p.Emit(ail.ROLE_SYS)
		case "user":
			p.Emit(ail.ROLE_USR)
		case "assistant":
			p.Emit(ail.ROLE_AST)
		case "tool":
			p.Emit(ail.ROLE_TOOL)
		}
		if m.text != "" {
			p.EmitString(ail.TXT_CHUNK, m.text)
		}
		for _, callID := range m.toolCalls {
			p.EmitString(ail.CALL_START, callID)
			p.EmitString(ail.CALL_NAME, "some_func")
			p.EmitJSON(ail.CALL_ARGS, json.RawMessage(`{}`))
			p.Emit(ail.CALL_END)
		}
		if m.toolCallID != "" {
			p.EmitString(ail.RESULT_START, m.toolCallID)
			data := m.resultData
			if data == "" {
				data = "result"
			}
			p.EmitString(ail.RESULT_DATA, data)
			p.Emit(ail.RESULT_END)
		}
		p.Emit(ail.MSG_END)
	}
	return p
}

// countMessages counts messages in the program using ailmanip.
func countMessages(p *ail.Program) int {
	return p.CountMessages()
}

// hasToolCalls returns true if the program contains any CALL_START opcodes.
func hasToolCalls(p *ail.Program) bool {
	return p.HasOpcode(ail.CALL_START)
}

// roleString maps an AIL role opcode to a human-readable string.
func roleString(op ail.Opcode) string {
	switch op {
	case ail.ROLE_SYS:
		return "system"
	case ail.ROLE_USR:
		return "user"
	case ail.ROLE_AST:
		return "assistant"
	case ail.ROLE_TOOL:
		return "tool"
	default:
		return "unknown"
	}
}

// collectRoles returns roles of all messages in order using ailmanip.
func collectRoles(p *ail.Program) []string {
	msgs := p.Messages()
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i] = roleString(m.Role)
	}
	return roles
}

func TestStripTools(t *testing.T) {
	tests := []struct {
		name            string
		msgs            []testMsg
		expectedCount   int
		expectToolCalls bool // whether final program should have any CALL_START
		expectedRoles   []string
	}{
		{
			name:          "no messages",
			msgs:          []testMsg{},
			expectedCount: 0,
		},
		{
			name: "no tool calls - unchanged",
			msgs: []testMsg{
				{role: "system", text: "You are helpful"},
				{role: "user", text: "Hello"},
				{role: "assistant", text: "Hi there!"},
			},
			expectedCount:   3,
			expectToolCalls: false,
		},
		{
			name: "single tool interaction - unchanged",
			msgs: []testMsg{
				{role: "user", text: "What's the weather?"},
				{role: "assistant", toolCalls: []string{"call_1"}},
				{role: "tool", toolCallID: "call_1", resultData: "Sunny, 72F"},
				{role: "assistant", text: "The weather is sunny!"},
			},
			expectedCount:   4,
			expectToolCalls: true,
		},
		{
			name: "two tool interactions - first stripped",
			msgs: []testMsg{
				{role: "user", text: "What's the weather?"},
				{role: "assistant", toolCalls: []string{"call_1"}},
				{role: "tool", toolCallID: "call_1", resultData: "Sunny, 72F"},
				{role: "assistant", text: "The weather is sunny. Need anything else?"},
				{role: "user", text: "What about LA?"},
				{role: "assistant", toolCalls: []string{"call_2"}},
				{role: "tool", toolCallID: "call_2", resultData: "Cloudy, 65F"},
			},
			expectedCount:   5, // user, assistant(text), user, assistant+toolcall, tool
			expectToolCalls: true,
			expectedRoles:   []string{"user", "assistant", "user", "assistant", "tool"},
		},
		{
			name: "three tool interactions - first two stripped",
			msgs: []testMsg{
				{role: "user", text: "Start"},
				{role: "assistant", toolCalls: []string{"c1"}},
				{role: "tool", toolCallID: "c1", resultData: "Result 1"},
				{role: "assistant", toolCalls: []string{"c2"}},
				{role: "tool", toolCallID: "c2", resultData: "Result 2"},
				{role: "assistant", toolCalls: []string{"c3"}},
				{role: "tool", toolCallID: "c3", resultData: "Result 3"},
			},
			expectedCount:   3, // user, assistant+toolcall, tool
			expectToolCalls: true,
			expectedRoles:   []string{"user", "assistant", "tool"},
		},
		{
			name: "tool interaction with content preserved",
			msgs: []testMsg{
				{role: "user", text: "Question"},
				{role: "assistant", text: "Let me check...", toolCalls: []string{"c1"}},
				{role: "tool", toolCallID: "c1", resultData: "Data"},
				{role: "assistant", toolCalls: []string{"c2"}},
				{role: "tool", toolCallID: "c2", resultData: "More data"},
			},
			expectedCount:   4, // user, assistant(content only), assistant+toolcall, tool
			expectToolCalls: true,
			expectedRoles:   []string{"user", "assistant", "assistant", "tool"},
		},
		{
			name: "multiple tool responses in one interaction",
			msgs: []testMsg{
				{role: "user", text: "Do multiple things"},
				{role: "assistant", toolCalls: []string{"c1", "c2"}},
				{role: "tool", toolCallID: "c1", resultData: "Result 1"},
				{role: "tool", toolCallID: "c2", resultData: "Result 2"},
				{role: "assistant", text: "Done with first batch"},
				{role: "assistant", toolCalls: []string{"c3"}},
				{role: "tool", toolCallID: "c3", resultData: "Result 3"},
			},
			expectedCount:   4, // user, assistant(text), assistant+toolcall, tool
			expectToolCalls: true,
			expectedRoles:   []string{"user", "assistant", "assistant", "tool"},
		},
	}

	plug := &StripTools{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog := buildAILProgram("gpt-4", tt.msgs)

			result, err := plug.Before("", nil, nil, prog)
			if err != nil {
				t.Fatalf("Plugin returned error: %v", err)
			}

			got := countMessages(result)
			if got != tt.expectedCount {
				t.Errorf("Expected %d messages, got %d", tt.expectedCount, got)
				t.Logf("Program:\n%s", result.Disasm())
			}

			if tt.expectedRoles != nil {
				roles := collectRoles(result)
				if len(roles) != len(tt.expectedRoles) {
					t.Errorf("Expected roles %v, got %v", tt.expectedRoles, roles)
				} else {
					for i, r := range roles {
						if r != tt.expectedRoles[i] {
							t.Errorf("Role[%d]: expected %q, got %q", i, tt.expectedRoles[i], r)
						}
					}
				}
			}
		})
	}
}

func TestStripTools_PreservesModel(t *testing.T) {
	prog := buildAILProgram("gpt-4o", []testMsg{
		{role: "user", text: "hi"},
		{role: "assistant", toolCalls: []string{"c1"}},
		{role: "tool", toolCallID: "c1", resultData: "r1"},
		{role: "assistant", toolCalls: []string{"c2"}},
		{role: "tool", toolCallID: "c2", resultData: "r2"},
	})

	plug := &StripTools{}
	result, err := plug.Before("", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	if result.GetModel() != "gpt-4o" {
		t.Errorf("expected model 'gpt-4o', got %q", result.GetModel())
	}
}
