package plugins

import (
	"testing"

	"github.com/neutrome-labs/ail"
)

func TestSlidingWindow_Defaults(t *testing.T) {
	// 15 messages, default slwin → keep 1 from start + 10 from end = 11
	msgs := make([]testMsg, 15)
	msgs[0] = testMsg{role: "system", text: "You are helpful"}
	for i := 1; i < 15; i++ {
		if i%2 == 1 {
			msgs[i] = testMsg{role: "user", text: "msg"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "reply"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 11 {
		t.Errorf("Expected 11 messages (1+10), got %d", got)
	}
}

func TestSlidingWindow_CustomEndOnly(t *testing.T) {
	// slwin:5 → keep 1 from start, 5 from end = 6
	msgs := make([]testMsg, 20)
	msgs[0] = testMsg{role: "system", text: "sys"}
	for i := 1; i < 20; i++ {
		if i%2 == 1 {
			msgs[i] = testMsg{role: "user", text: "u"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "a"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("5", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 6 {
		t.Errorf("Expected 6 messages (1+5), got %d", got)
	}

	// First message should still be system
	roles := collectRoles(result)
	if roles[0] != "system" {
		t.Errorf("First message should be system, got %s", roles[0])
	}
}

func TestSlidingWindow_CustomBoth(t *testing.T) {
	// slwin:5:3 → keep 3 from start, 5 from end = 8
	msgs := make([]testMsg, 20)
	msgs[0] = testMsg{role: "system", text: "sys"}
	msgs[1] = testMsg{role: "user", text: "context1"}
	msgs[2] = testMsg{role: "assistant", text: "ack1"}
	for i := 3; i < 20; i++ {
		if i%2 == 1 {
			msgs[i] = testMsg{role: "user", text: "u"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "a"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("5:3", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 8 {
		t.Errorf("Expected 8 messages (3+5), got %d", got)
	}
}

func TestSlidingWindow_FitsInWindow(t *testing.T) {
	// Only 3 messages with default window (1+10=11) → nothing removed
	msgs := []testMsg{
		{role: "system", text: "sys"},
		{role: "user", text: "hi"},
		{role: "assistant", text: "hello"},
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 3 {
		t.Errorf("Expected 3 messages (all fit), got %d", got)
	}
}

func TestSlidingWindow_PreservesNonMessageInstructions(t *testing.T) {
	// Verify SET_MODEL and tool definitions survive the window.
	msgs := make([]testMsg, 15)
	msgs[0] = testMsg{role: "system", text: "sys"}
	for i := 1; i < 15; i++ {
		if i%2 == 1 {
			msgs[i] = testMsg{role: "user", text: "u"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "a"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	// Add a tool definition before messages
	defProg := ail.NewProgram()
	defProg.EmitString(ail.SET_MODEL, "gpt-4")
	defProg.Emit(ail.DEF_START)
	defProg.EmitString(ail.TXT_CHUNK, "test_tool")
	defProg.Emit(ail.DEF_END)
	// Copy messages from prog
	for _, inst := range prog.Code {
		if inst.Op == ail.SET_MODEL {
			continue // already added
		}
		defProg.Code = append(defProg.Code, inst)
	}

	plug := &SlidingWindow{}
	result, err := plug.Before("5:1", nil, nil, defProg)
	if err != nil {
		t.Fatal(err)
	}

	if result.GetModel() != "gpt-4" {
		t.Errorf("Model lost: got %q", result.GetModel())
	}

	// Check DEF_START survived
	hasDef := false
	for _, inst := range result.Code {
		if inst.Op == ail.DEF_START {
			hasDef = true
			break
		}
	}
	if !hasDef {
		t.Error("DEF_START was removed by sliding window")
	}
}

func TestSlidingWindow_ZeroStart(t *testing.T) {
	// slwin:3:0 → keep 0 from start, 3 from end
	msgs := make([]testMsg, 10)
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			msgs[i] = testMsg{role: "user", text: "u"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "a"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("3:0", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 3 {
		t.Errorf("Expected 3 messages, got %d", got)
	}
}

func TestSlidingWindow_OverlappingWindows(t *testing.T) {
	// slwin:8:5 with 10 messages → start keeps 0-4, end keeps 2-9 → overlap, keep all unique = 10
	msgs := make([]testMsg, 10)
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			msgs[i] = testMsg{role: "user", text: "u"}
		} else {
			msgs[i] = testMsg{role: "assistant", text: "a"}
		}
	}

	prog := buildAILProgram("gpt-4", msgs)
	plug := &SlidingWindow{}

	result, err := plug.Before("8:5", nil, nil, prog)
	if err != nil {
		t.Fatal(err)
	}

	got := countMessages(result)
	if got != 10 {
		t.Errorf("Expected 10 messages (overlapping windows keep all), got %d", got)
	}
}
