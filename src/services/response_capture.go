package services

import "net/http"

// ResponseCaptureWriter captures response data instead of writing to HTTP.
// Implements http.ResponseWriter and http.Flusher so that it can be used
// as a drop-in replacement wherever a real ResponseWriter is expected
// (including SSE writers that require Flush support).
type ResponseCaptureWriter struct {
	Response   []byte
	Headers    http.Header
	StatusCode int
}

func (w *ResponseCaptureWriter) Header() http.Header {
	if w.Headers == nil {
		w.Headers = make(http.Header)
	}
	return w.Headers
}

func (w *ResponseCaptureWriter) Write(data []byte) (int, error) {
	w.Response = append(w.Response, data...)
	return len(data), nil
}

func (w *ResponseCaptureWriter) WriteHeader(statusCode int) {
	w.StatusCode = statusCode
}

// Flush implements http.Flusher — no-op for capture.
func (w *ResponseCaptureWriter) Flush() {}

// Compile-time interface checks.
var (
	_ http.ResponseWriter = (*ResponseCaptureWriter)(nil)
	_ http.Flusher        = (*ResponseCaptureWriter)(nil)
)
