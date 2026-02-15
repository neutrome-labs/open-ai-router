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

// Responses implements the OpenAI Responses API using AIL.
type Responses struct{}

var responsesEmitter = &ail.ResponsesEmitter{}
var responsesParser = &ail.ResponsesParser{}

func (c *Responses) createRequest(p *services.ProviderService, prog *ail.Program, r *http.Request, endpoint string) (*http.Request, error) {
	targetUrl := p.ParsedURL
	targetUrl.Path += endpoint

	targetHeader := r.Header.Clone()
	targetHeader.Del("Accept-Encoding")
	targetHeader.Set("Content-Type", "application/json")

	// Emit the AIL program as Responses API JSON
	reqBody, err := responsesEmitter.EmitRequest(prog)
	if err != nil {
		return nil, fmt.Errorf("openai responses: emit request: %w", err)
	}

	httpReq := &http.Request{
		Method:        "POST",
		URL:           &targetUrl,
		Header:        targetHeader,
		Body:          io.NopCloser(bytes.NewReader(reqBody)),
		ContentLength: int64(len(reqBody)),
	}
	httpReq = httpReq.WithContext(r.Context())

	authVal, err := p.Router.Auth.CollectTargetAuth("responses", p, r, httpReq)
	if err != nil {
		return nil, err
	}
	if authVal != "" {
		httpReq.Header.Set("Authorization", "Bearer "+authVal)
	}

	return httpReq, nil
}

// DoInference implements InferenceCommand for OpenAI Responses API.
func (c *Responses) DoInference(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, *ail.Program, error) {
	Logger.Debug("DoInference (responses) starting",
		zap.String("provider", p.Name),
		zap.String("model", prog.GetModel()),
		zap.String("base_url", p.ParsedURL.String()))

	httpReq, err := c.createRequest(p, prog, r, "/responses")
	if err != nil {
		Logger.Error("DoInference (responses) createRequest failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInference (responses) sending request", zap.String("url", httpReq.URL.String()))

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		Logger.Error("DoInference (responses) HTTP request failed", zap.Error(err))
		return nil, nil, err
	}
	defer res.Body.Close()

	Logger.Debug("DoInference (responses) response received", zap.Int("status", res.StatusCode))

	respData, _ := io.ReadAll(res.Body)

	if res.StatusCode != 200 {
		Logger.Error("DoInference (responses) non-200 response",
			zap.Int("status", res.StatusCode),
			zap.String("body", string(respData)))
		return res, nil, fmt.Errorf("%s", string(respData))
	}

	respProg, err := responsesParser.ParseResponse(respData)
	if err != nil {
		Logger.Error("DoInference (responses) response parse failed", zap.Error(err))
		return res, nil, err
	}

	Logger.Debug("DoInference (responses) completed successfully")

	return res, respProg, nil
}

// DoInferenceStream implements InferenceCommand for streaming OpenAI Responses.
func (c *Responses) DoInferenceStream(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, chan drivers.InferenceStreamChunk, error) {
	Logger.Debug("DoInferenceStream (responses) starting",
		zap.String("provider", p.Name))

	httpReq, err := c.createRequest(p, prog, r, "/responses")
	if err != nil {
		Logger.Error("DoInferenceStream (responses) createRequest failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInferenceStream (responses) sending request", zap.String("url", httpReq.URL.String()))

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		Logger.Error("DoInferenceStream (responses) HTTP request failed", zap.Error(err))
		return nil, nil, err
	}

	Logger.Debug("DoInferenceStream (responses) response received",
		zap.Int("status", res.StatusCode),
		zap.String("content_type", res.Header.Get("Content-Type")))

	chunks := make(chan drivers.InferenceStreamChunk)

	go func() {
		defer close(chunks)
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			respData, _ := io.ReadAll(res.Body)
			Logger.Error("DoInferenceStream (responses) non-200 response",
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

			respProg, err := responsesParser.ParseResponse(respData)
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
				chunkProg, err := responsesParser.ParseStreamChunk(event.Data)
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
