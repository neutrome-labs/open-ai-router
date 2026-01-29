package flow

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/plugins"
	"github.com/neutrome-labs/open-ai-router/src/styles"
	"go.uber.org/zap"
)

// Swarm implements a bee-hive style multi-agent orchestration system.
// The "mother" swarm manager decomposes tasks and spawns 0.5N to 2N sub-agents
// to work in parallel, then synthesizes the results.
//
// Usage: model="gpt-4+swarm:20" (where 20 is the target agent count, default: 20)
type Swarm struct{}

func (s *Swarm) Name() string { return "swarm" }

// SwarmTask represents a single sub-task assigned to a worker agent
type SwarmTask struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	Priority    int    `json:"priority"`
}

// SwarmResult represents the result from a worker agent
type SwarmResult struct {
	TaskID   string `json:"task_id"`
	Output   string `json:"output"`
	Complete bool   `json:"complete"`
}

// RecursiveHandler implements the swarm orchestration logic.
// The mother agent analyzes the request, decomposes it into sub-tasks,
// spawns worker agents in parallel, and synthesizes the final response.
func (s *Swarm) RecursiveHandler(
	params string,
	invoker plugin.HandlerInvoker,
	reqJson styles.PartialJSON,
	w http.ResponseWriter,
	r *http.Request,
) (handled bool, err error) {
	// Parse agent count parameter (default: 20)
	targetAgentCount := 20
	if params != "" {
		if n, err := parseInt(params); err == nil && n > 0 {
			targetAgentCount = n
		}
	}

	// Get the original model and messages
	originalModel := styles.TryGetFromPartialJSON[string](reqJson, "model")
	messages, err := styles.GetFromPartialJSON[[]styles.ChatCompletionsMessage](reqJson, "messages")
	if err != nil {
		plugins.Logger.Error("swarm plugin: failed to get messages", zap.Error(err))
		return false, nil
	}

	// Check if streaming - swarm doesn't support streaming
	stream := styles.TryGetFromPartialJSON[bool](reqJson, "stream")
	if stream {
		plugins.Logger.Warn("swarm plugin: streaming not supported, processing as non-streaming")
	}

	// Extract base model (remove plugin suffix)
	baseModel := extractBaseModel(originalModel)

	plugins.Logger.Info("swarm plugin starting orchestration",
		zap.Int("target_agents", targetAgentCount),
		zap.String("model", baseModel))

	// Get decomposition prompt from environment or use default
	decompPrompt := getDecompositionPrompt()

	// === PHASE 1: Mother Agent Task Decomposition ===
	tasks, err := s.decomposeTask(invoker, r, reqJson, baseModel, messages, decompPrompt, targetAgentCount)
	if err != nil {
		plugins.Logger.Error("swarm plugin: task decomposition failed", zap.Error(err))
		return false, nil
	}

	if len(tasks) == 0 {
		plugins.Logger.Debug("swarm plugin: no tasks to process, letting normal flow handle")
		return false, nil
	}

	plugins.Logger.Info("swarm plugin: task decomposition complete",
		zap.Int("task_count", len(tasks)))

	// === PHASE 2: Parallel Worker Agent Execution ===
	results := s.executeWorkersInParallel(invoker, r, reqJson, baseModel, messages, tasks)

	plugins.Logger.Info("swarm plugin: worker execution complete",
		zap.Int("completed_tasks", len(results)))

	// === PHASE 3: Mother Agent Result Synthesis ===
	finalResponse, err := s.synthesizeResults(invoker, r, reqJson, baseModel, messages, tasks, results)
	if err != nil {
		plugins.Logger.Error("swarm plugin: result synthesis failed", zap.Error(err))
		return false, nil
	}

	// Write the final synthesized response
	w.Header().Set("Content-Type", "application/json")
	respData, err := finalResponse.Marshal()
	if err != nil {
		plugins.Logger.Error("swarm plugin: failed to marshal final response", zap.Error(err))
		return true, err
	}
	w.Write(respData)

	plugins.Logger.Info("swarm plugin: orchestration complete")
	return true, nil
}

// decomposeTask uses the mother agent to analyze and decompose the task into sub-tasks
func (s *Swarm) decomposeTask(
	invoker plugin.HandlerInvoker,
	r *http.Request,
	reqJson styles.PartialJSON,
	model string,
	messages []styles.ChatCompletionsMessage,
	decompPrompt string,
	targetAgentCount int,
) ([]SwarmTask, error) {
	// Build the decomposition request
	userPrompt := buildUserPromptForDecomposition(messages)

	decompMessages := []styles.ChatCompletionsMessage{
		{
			Role:    "system",
			Content: decompPrompt,
		},
		{
			Role:    "user",
			Content: userPrompt,
		},
	}

	decompReq, err := reqJson.CloneWith("messages", decompMessages)
	if err != nil {
		return nil, err
	}

	// Set response format to get structured output
	decompReq, _ = decompReq.CloneWith("model", model)

	// Remove stream if present
	delete(decompReq, "stream")
	delete(decompReq, "stream_options")

	reqData, err := decompReq.Marshal()
	if err != nil {
		return nil, err
	}

	clonedReq := r.Clone(r.Context())
	clonedReq.Body = io.NopCloser(strings.NewReader(string(reqData)))

	plugins.Logger.Debug("swarm plugin: calling mother agent for decomposition")

	respJson, err := invoker.InvokeHandlerCapture(clonedReq)
	if err != nil {
		return nil, fmt.Errorf("decomposition call failed: %w", err)
	}

	// Parse the decomposition response
	resp, err := styles.ParseChatCompletionsResponse(respJson)
	if err != nil {
		return nil, fmt.Errorf("failed to parse decomposition response: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
		return nil, fmt.Errorf("no decomposition response")
	}

	// Extract tasks from the response content
	content := getMessageContent(resp.Choices[0].Message)
	tasks := parseTasksFromResponse(content, targetAgentCount)

	return tasks, nil
}

// executeWorkersInParallel spawns worker agents to process tasks in parallel
func (s *Swarm) executeWorkersInParallel(
	invoker plugin.HandlerInvoker,
	r *http.Request,
	reqJson styles.PartialJSON,
	model string,
	originalMessages []styles.ChatCompletionsMessage,
	tasks []SwarmTask,
) []SwarmResult {
	type result struct {
		taskID string
		output string
		err    error
	}

	resultsChan := make(chan result, len(tasks))
	var wg sync.WaitGroup

	// Get tools from original request if any
	tools, _ := styles.GetFromPartialJSON[[]styles.ChatCompletionsTool](reqJson, "tools")

	for _, task := range tasks {
		wg.Add(1)
		go func(t SwarmTask) {
			defer wg.Done()

			output, err := s.executeWorker(invoker, r, reqJson, model, originalMessages, t, tools)
			resultsChan <- result{
				taskID: t.ID,
				output: output,
				err:    err,
			}
		}(task)
	}

	// Close channel when all workers complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results
	var results []SwarmResult
	for res := range resultsChan {
		if res.err != nil {
			plugins.Logger.Warn("swarm plugin: worker failed",
				zap.String("task_id", res.taskID),
				zap.Error(res.err))
			results = append(results, SwarmResult{
				TaskID:   res.taskID,
				Output:   fmt.Sprintf("Error: %v", res.err),
				Complete: false,
			})
		} else {
			results = append(results, SwarmResult{
				TaskID:   res.taskID,
				Output:   res.output,
				Complete: true,
			})
		}
	}

	return results
}

// executeWorker executes a single worker agent for a specific task
func (s *Swarm) executeWorker(
	invoker plugin.HandlerInvoker,
	r *http.Request,
	reqJson styles.PartialJSON,
	model string,
	originalMessages []styles.ChatCompletionsMessage,
	task SwarmTask,
	tools []styles.ChatCompletionsTool,
) (string, error) {
	// Build worker prompt with context from original messages
	workerSystemPrompt := fmt.Sprintf(`You are a specialized worker agent in a swarm. Your task is:

TASK ID: %s
TASK DESCRIPTION: %s

Focus solely on completing this specific task. Be concise but thorough. Return your findings in a clear, structured format.`, task.ID, task.Description)

	// Include relevant context from original conversation
	var workerMessages []styles.ChatCompletionsMessage
	workerMessages = append(workerMessages, styles.ChatCompletionsMessage{
		Role:    "system",
		Content: workerSystemPrompt,
	})

	// Add original conversation context (last few messages for context)
	contextMessages := getContextMessages(originalMessages, 5)
	workerMessages = append(workerMessages, contextMessages...)

	// Add the specific task
	workerMessages = append(workerMessages, styles.ChatCompletionsMessage{
		Role:    "user",
		Content: fmt.Sprintf("Complete this task: %s", task.Description),
	})

	workerReq, err := reqJson.CloneWith("messages", workerMessages)
	if err != nil {
		return "", err
	}

	workerReq, _ = workerReq.CloneWith("model", model)
	delete(workerReq, "stream")
	delete(workerReq, "stream_options")

	// Add tools if available
	if len(tools) > 0 {
		workerReq, _ = workerReq.CloneWith("tools", tools)
	}

	reqData, err := workerReq.Marshal()
	if err != nil {
		return "", err
	}

	clonedReq := r.Clone(r.Context())
	clonedReq.Body = io.NopCloser(strings.NewReader(string(reqData)))

	plugins.Logger.Debug("swarm plugin: executing worker",
		zap.String("task_id", task.ID))

	respJson, err := invoker.InvokeHandlerCapture(clonedReq)
	if err != nil {
		return "", err
	}

	resp, err := styles.ParseChatCompletionsResponse(respJson)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
		return "", fmt.Errorf("no response from worker")
	}

	return getMessageContent(resp.Choices[0].Message), nil
}

// synthesizeResults uses the mother agent to synthesize all worker results into a final response
func (s *Swarm) synthesizeResults(
	invoker plugin.HandlerInvoker,
	r *http.Request,
	reqJson styles.PartialJSON,
	model string,
	originalMessages []styles.ChatCompletionsMessage,
	tasks []SwarmTask,
	results []SwarmResult,
) (styles.PartialJSON, error) {
	// Build synthesis prompt
	synthesisPrompt := `You are the Swarm Orchestrator (Mother Agent). Your role is to synthesize the results from multiple worker agents into a coherent, comprehensive final response.

TASKS AND RESULTS:
`

	for _, result := range results {
		taskDesc := ""
		for _, t := range tasks {
			if t.ID == result.TaskID {
				taskDesc = t.Description
				break
			}
		}
		synthesisPrompt += fmt.Sprintf("\n--- Task: %s ---\nDescription: %s\nResult: %s\nComplete: %v\n",
			result.TaskID, taskDesc, result.Output, result.Complete)
	}

	synthesisPrompt += `

INSTRUCTIONS:
1. Analyze all worker results carefully
2. Integrate the findings into a coherent response
3. Resolve any contradictions or inconsistencies
4. Provide a comprehensive answer that addresses the original request
5. If tasks failed, note what information is missing
6. Maintain the tone and style appropriate to the original request

Provide your synthesized response directly. Do not mention the swarm process unless relevant to the answer.`

	// Build synthesis messages preserving original context
	var synthesisMessages []styles.ChatCompletionsMessage
	synthesisMessages = append(synthesisMessages, styles.ChatCompletionsMessage{
		Role:    "system",
		Content: synthesisPrompt,
	})

	// Add original user request for context
	originalUserPrompt := buildUserPromptForDecomposition(originalMessages)
	synthesisMessages = append(synthesisMessages, styles.ChatCompletionsMessage{
		Role:    "user",
		Content: "Original request: " + originalUserPrompt + "\n\nPlease provide the final synthesized response based on the worker results above.",
	})

	synthesisReq, err := reqJson.CloneWith("messages", synthesisMessages)
	if err != nil {
		return nil, err
	}

	synthesisReq, _ = synthesisReq.CloneWith("model", model)
	delete(synthesisReq, "stream")
	delete(synthesisReq, "stream_options")

	reqData, err := synthesisReq.Marshal()
	if err != nil {
		return nil, err
	}

	clonedReq := r.Clone(r.Context())
	clonedReq.Body = io.NopCloser(strings.NewReader(string(reqData)))

	plugins.Logger.Debug("swarm plugin: calling mother agent for synthesis")

	respJson, err := invoker.InvokeHandlerCapture(clonedReq)
	if err != nil {
		return nil, fmt.Errorf("synthesis call failed: %w", err)
	}

	// Parse and enhance the response
	// Parse and return the response
	_, err = styles.ParseChatCompletionsResponse(respJson)
	if err != nil {
		return nil, err
	}

	return respJson, nil
}

// Helper functions

func parseInt(s string) (int, error) {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

func extractBaseModel(model string) string {
	// Remove plugin suffix (everything after +)
	if idx := strings.IndexByte(model, '+'); idx >= 0 {
		return model[:idx]
	}
	return model
}

func getDecompositionPrompt() string {
	// Check environment variable first
	if prompt := os.Getenv("SWARM_DECOMPOSITION_PROMPT"); prompt != "" {
		return prompt
	}

	// Default decomposition prompt
	return `You are a Swarm Orchestrator (Mother Agent). Your task is to analyze the user's request and decompose it into smaller, parallelizable sub-tasks.

ANALYSIS INSTRUCTIONS:
1. Understand the overall goal of the request
2. Identify distinct components or aspects that can be worked on in parallel
3. Break down complex tasks into manageable sub-tasks (0.5N to 2N tasks where N is the target count)
4. Ensure each sub-task is specific, actionable, and self-contained
5. Prioritize tasks by importance/dependency

OUTPUT FORMAT:
Return your response as a JSON array of tasks in this exact format:
[
  {"id": "task-1", "description": "Detailed description of sub-task 1", "priority": 1},
  {"id": "task-2", "description": "Detailed description of sub-task 2", "priority": 2},
  ...
]

Each task should be completable independently by a worker agent. Aim for comprehensive coverage of the original request.`
}

func buildUserPromptForDecomposition(messages []styles.ChatCompletionsMessage) string {
	// Extract the user's intent from the conversation
	var parts []string
	for _, msg := range messages {
		if msg.Role == "user" || msg.Role == "system" {
			content := getMessageContent(&msg)
			if content != "" {
				parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, content))
			}
		}
	}
	return strings.Join(parts, "\n")
}

func getMessageContent(msg *styles.ChatCompletionsMessage) string {
	if msg.Content == nil {
		return ""
	}

	switch v := msg.Content.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		// Try to marshal complex content
		data, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(data)
	}
}

func getContextMessages(messages []styles.ChatCompletionsMessage, count int) []styles.ChatCompletionsMessage {
	if len(messages) <= count {
		return messages
	}
	return messages[len(messages)-count:]
}

func parseTasksFromResponse(content string, targetCount int) []SwarmTask {
	// Try to extract JSON array from the response
	content = strings.TrimSpace(content)

	// Look for JSON array in the content
	startIdx := strings.Index(content, "[")
	endIdx := strings.LastIndex(content, "]")

	if startIdx >= 0 && endIdx > startIdx {
		jsonPart := content[startIdx : endIdx+1]
		var tasks []SwarmTask
		if err := json.Unmarshal([]byte(jsonPart), &tasks); err == nil {
			// Validate and return tasks
			var validTasks []SwarmTask
			for _, t := range tasks {
				if t.ID != "" && t.Description != "" {
					validTasks = append(validTasks, t)
				}
			}
			if len(validTasks) > 0 {
				return validTasks
			}
		}
	}

	// Fallback: create a single task from the content
	return []SwarmTask{
		{
			ID:          "task-1",
			Description: content,
			Priority:    1,
		},
	}
}

var (
	_ plugin.RecursiveHandlerPlugin = (*Swarm)(nil)
)
