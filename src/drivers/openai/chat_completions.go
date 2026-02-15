package openai

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/drivers"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"go.uber.org/zap"
)

// Logger for OpenAI driver - can be set by modules
var Logger *zap.Logger = zap.NewNop()

// ChatCompletions implements chat completions for OpenAI-compatible APIs.
// Takes an AIL Program, emits it as OpenAI Chat Completions JSON, and parses
// the response back into an AIL Program.
type ChatCompletions struct{}

var emitter = &ail.ChatCompletionsEmitter{}
var parser = &ail.ChatCompletionsParser{}

func (c *ChatCompletions) createRequest(p *services.ProviderService, prog *ail.Program, r *http.Request, endpoint string) (*http.Request, error) {
	targetUrl := p.ParsedURL
	targetUrl.Path += endpoint

	targetHeader := r.Header.Clone()
	targetHeader.Del("Accept-Encoding")
	targetHeader.Set("Content-Type", "application/json")

	// Emit the AIL program as Chat Completions JSON
	reqBody, err := emitter.EmitRequest(prog)
	if err != nil {
		return nil, fmt.Errorf("openai: emit request: %w", err)
	}

	httpReq := &http.Request{
		Method:        "POST",
		URL:           &targetUrl,
		Header:        targetHeader,
		Body:          io.NopCloser(bytes.NewReader(reqBody)),
		ContentLength: int64(len(reqBody)),
	}
	httpReq = httpReq.WithContext(r.Context())

	authVal, err := p.Router.Auth.CollectTargetAuth("chat_completions", p, r, httpReq)
	if err != nil {
		return nil, err
	}
	if authVal != "" {
		httpReq.Header.Set("Authorization", "Bearer "+authVal)
	}

	return httpReq, nil
}

// DoInference implements InferenceCommand for OpenAI Chat Completions API.
func (c *ChatCompletions) DoInference(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, *ail.Program, error) {
	Logger.Debug("DoInference (chat_completions) starting",
		zap.String("provider", p.Name),
		zap.String("model", prog.GetModel()),
		zap.String("base_url", p.ParsedURL.String()))

	httpReq, err := c.createRequest(p, prog, r, "/chat/completions")
	if err != nil {
		Logger.Error("DoInference (chat_completions) createRequest failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInference (chat_completions) sending request", zap.String("url", httpReq.URL.String()))

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		Logger.Error("DoInference (chat_completions) HTTP request failed", zap.Error(err))
		return nil, nil, err
	}
	defer res.Body.Close()

	Logger.Debug("DoInference (chat_completions) response received", zap.Int("status", res.StatusCode))

	respData, _ := io.ReadAll(res.Body)

	if res.StatusCode != 200 {
		Logger.Error("DoInference (chat_completions) non-200 response",
			zap.Int("status", res.StatusCode),
			zap.String("body", string(respData)))
		return res, nil, fmt.Errorf("%s", string(respData))
	}

	// Parse response into AIL
	respProg, err := parser.ParseResponse(respData)
	if err != nil {
		Logger.Error("DoInference (chat_completions) response parse failed", zap.Error(err))
		return res, nil, err
	}

	Logger.Debug("DoInference (chat_completions) completed successfully")

	return res, respProg, nil
}

// DoInferenceStream implements InferenceCommand for streaming OpenAI Chat Completions.
func (c *ChatCompletions) DoInferenceStream(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, chan drivers.InferenceStreamChunk, error) {
	Logger.Debug("DoInferenceStream (chat_completions) starting",
		zap.String("provider", p.Name),
		zap.String("model", prog.GetModel()))

	httpReq, err := c.createRequest(p, prog, r, "/chat/completions")
	if err != nil {
		Logger.Error("DoInferenceStream (chat_completions) createRequest failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInferenceStream (chat_completions) sending request", zap.String("url", httpReq.URL.String()))

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		Logger.Error("DoInferenceStream (chat_completions) HTTP request failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInferenceStream (chat_completions) response received",
		zap.Int("status", res.StatusCode),
		zap.String("content_type", res.Header.Get("Content-Type")))

	chunks := make(chan drivers.InferenceStreamChunk)

	go func() {
		defer close(chunks)
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			respData, _ := io.ReadAll(res.Body)
			Logger.Error("DoInferenceStream (chat_completions) non-200 response",
				zap.Int("status", res.StatusCode),
				zap.String("body", string(respData)))
			chunks <- drivers.InferenceStreamChunk{
				RuntimeError: fmt.Errorf("%s - %s", res.Status, string(respData)),
			}
			return
		}

		ct := res.Header.Get("Content-Type")
		isSSE := strings.HasPrefix(strings.ToLower(ct), "text/event-stream")

		if !isSSE {
			respData, err := io.ReadAll(res.Body)
			if err != nil {
				chunks <- drivers.InferenceStreamChunk{RuntimeError: err}
				return
			}

			respProg, err := parser.ParseResponse(respData)
			if err != nil {
				chunks <- drivers.InferenceStreamChunk{RuntimeError: err}
				return
			}

			chunks <- drivers.InferenceStreamChunk{Data: respProg}
			return
		}

		reader := sse.NewDefaultReader(res.Body)
		for event := range reader.ReadEvents() {
			if event.Error != nil {
				chunks <- drivers.InferenceStreamChunk{RuntimeError: event.Error}
				return
			}
			if event.Done {
				return
			}
			if event.Data != nil {
				chunkProg, err := parser.ParseStreamChunk(event.Data)
				if err != nil {
					chunks <- drivers.InferenceStreamChunk{RuntimeError: err}
					return
				}
				chunks <- drivers.InferenceStreamChunk{Data: chunkProg}
			}
		}
	}()

	return res, chunks, nil
}
