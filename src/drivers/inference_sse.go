package drivers

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"github.com/neutrome-labs/open-ai-router/src/sse"
	"go.uber.org/zap"
)

// Logger for driver operations — set by endpoint modules during Provision.
var Logger *zap.Logger = zap.NewNop()

// InferenceSse is a generic InferenceCommand that works with any ail.Style
// over HTTP+SSE transport. It resolves emitters and parsers from the ail
// registry at construction time, eliminating per-style driver boilerplate.
//
// Adding support for a new upstream API style requires only that the ail
// package registers parsers/emitters for that style — no driver code changes.
//
// For non-SSE transports (e.g. WebSocket for realtime APIs) a separate
// driver implementation would be needed.
type InferenceSse struct {
	style       ail.Style
	endpoint    string // e.g. "/chat/completions", "/responses", "/messages"
	emitter     ail.Emitter
	respParser  ail.ResponseParser
	chunkParser ail.StreamChunkParser
}

// NewInferenceSse creates an InferenceCommand for the given upstream style
// using HTTP+SSE transport.
// Returns an error if the ail package lacks parsers/emitters for this style.
func NewInferenceSse(style ail.Style, endpoint string) (*InferenceSse, error) {
	emitter, err := ail.GetEmitter(style)
	if err != nil {
		return nil, fmt.Errorf("no emitter for style %s: %w", style, err)
	}
	respParser, err := ail.GetResponseParser(style)
	if err != nil {
		return nil, fmt.Errorf("no response parser for style %s: %w", style, err)
	}
	chunkParser, err := ail.GetStreamChunkParser(style)
	if err != nil {
		return nil, fmt.Errorf("no stream chunk parser for style %s: %w", style, err)
	}
	return &InferenceSse{
		style:       style,
		endpoint:    endpoint,
		emitter:     emitter,
		respParser:  respParser,
		chunkParser: chunkParser,
	}, nil
}

func (d *InferenceSse) createRequest(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Request, error) {
	targetURL := p.ParsedURL
	targetURL.Path += d.endpoint

	targetHeader := r.Header.Clone()
	targetHeader.Del("Accept-Encoding")
	targetHeader.Set("Content-Type", "application/json")

	reqBody, err := d.emitter.EmitRequest(prog)
	if err != nil {
		return nil, fmt.Errorf("%s driver: emit request: %w", d.style, err)
	}

	httpReq := &http.Request{
		Method:        "POST",
		URL:           &targetURL,
		Header:        targetHeader,
		Body:          io.NopCloser(bytes.NewReader(reqBody)),
		ContentLength: int64(len(reqBody)),
	}
	httpReq = httpReq.WithContext(r.Context())

	authVal, err := p.Router.Auth.CollectTargetAuth(string(d.style), p, r, httpReq)
	if err != nil {
		return nil, err
	}
	if authVal != "" {
		httpReq.Header.Set("Authorization", "Bearer "+authVal)
	}

	return httpReq, nil
}

// DoInference implements InferenceCommand for non-streaming requests.
func (d *InferenceSse) DoInference(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, *ail.Program, error) {
	Logger.Debug("DoInference starting",
		zap.String("style", string(d.style)),
		zap.String("provider", p.Name),
		zap.String("model", prog.GetModel()),
		zap.String("base_url", p.ParsedURL.String()))

	httpReq, err := d.createRequest(p, prog, r)
	if err != nil {
		return nil, nil, err
	}

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, nil, err
	}
	defer res.Body.Close()

	respData, _ := io.ReadAll(res.Body)

	if res.StatusCode != http.StatusOK {
		Logger.Error("non-200 response",
			zap.String("style", string(d.style)),
			zap.Int("status", res.StatusCode),
			zap.String("body", string(respData)))
		return res, nil, fmt.Errorf("%s", string(respData))
	}

	respProg, err := d.respParser.ParseResponse(respData)
	if err != nil {
		return res, nil, err
	}

	return res, respProg, nil
}

// DoInferenceStream implements InferenceCommand for streaming requests.
func (d *InferenceSse) DoInferenceStream(p *services.ProviderService, prog *ail.Program, r *http.Request) (*http.Response, chan InferenceStreamChunk, error) {
	Logger.Debug("DoInferenceStream starting",
		zap.String("style", string(d.style)),
		zap.String("provider", p.Name),
		zap.String("model", prog.GetModel()))

	httpReq, err := d.createRequest(p, prog, r)
	if err != nil {
		return nil, nil, err
	}

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, nil, err
	}

	chunks := make(chan InferenceStreamChunk)

	go func() {
		defer close(chunks)
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			respData, _ := io.ReadAll(res.Body)
			Logger.Error("non-200 streaming response",
				zap.String("style", string(d.style)),
				zap.Int("status", res.StatusCode),
				zap.String("body", string(respData)))
			chunks <- InferenceStreamChunk{
				RuntimeError: fmt.Errorf("%s - %s", res.Status, string(respData)),
			}
			return
		}

		ct := res.Header.Get("Content-Type")
		isSSE := strings.HasPrefix(strings.ToLower(ct), "text/event-stream")

		if !isSSE {
			// Non-SSE response to a streaming request — parse as full response.
			respData, err := io.ReadAll(res.Body)
			if err != nil {
				chunks <- InferenceStreamChunk{RuntimeError: err}
				return
			}
			respProg, err := d.respParser.ParseResponse(respData)
			if err != nil {
				chunks <- InferenceStreamChunk{RuntimeError: err}
				return
			}
			chunks <- InferenceStreamChunk{Data: respProg}
			return
		}

		reader := sse.NewDefaultReader(res.Body)
		for event := range reader.ReadEvents() {
			if event.Error != nil {
				chunks <- InferenceStreamChunk{RuntimeError: event.Error}
				return
			}
			if event.Done {
				return
			}
			if event.Data != nil {
				chunkProg, err := d.chunkParser.ParseStreamChunk(event.Data)
				if err != nil {
					chunks <- InferenceStreamChunk{RuntimeError: err}
					return
				}
				chunks <- InferenceStreamChunk{Data: chunkProg}
			}
		}
	}()

	return res, chunks, nil
}

// EndpointForStyle returns the default upstream API path for a provider style.
func EndpointForStyle(style ail.Style) string {
	switch style {
	case ail.StyleChatCompletions:
		return "/chat/completions"
	case ail.StyleResponses:
		return "/responses"
	case ail.StyleAnthropic:
		return "/messages"
	case ail.StyleGoogleGenAI:
		return "/chat/completions" // Google GenAI via OpenAI-compat REST endpoint
	default:
		return "/chat/completions"
	}
}

var _ InferenceCommand = (*InferenceSse)(nil)
