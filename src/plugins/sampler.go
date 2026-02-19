package plugins

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/neutrome-labs/ail"
	"github.com/neutrome-labs/open-ai-router/src/plugin"
	"github.com/neutrome-labs/open-ai-router/src/services"
	"go.uber.org/zap"
)

// Sampler persists AIL programs (request, upstream-prepared, response) to
// disk for debugging and test-corpus collection.
//
// File layout under Dir:
//
//	<dir>/<hash>/request.ail       – initial parsed request (binary)
//	<dir>/<hash>/request.up.ail    – upstream-prepared after before-plugins (binary)
//	<dir>/<hash>/response.ail      – complete response (binary)
//	<dir>/<hash>.txt               – human-readable disassembly of all three
//
// The hash is derived from the binary encoding of the initial request program.
// Identical requests are deduplicated (the request.ail file is written only once).
//
// The plugin is auto-enabled when registered in plugin.TailPlugins; it is
// registered by modules.init() when the SAMPLER environment variable is set.
type Sampler struct {
	Dir string
	// hashes maps traceID → request hash for the current request so that
	// Before, After, and StreamEnd can reference the right sample directory.
	hashes sync.Map
}

// NewSampler creates a Sampler that writes samples into dir.
func NewSampler(dir string) *Sampler {
	return &Sampler{Dir: dir}
}

func (s *Sampler) Name() string { return "sampler" }

// OnRequestInit is called once per request with the original parsed program.
// It computes the sample hash, creates the per-request directory, and writes
// the initial request AIL.
func (s *Sampler) OnRequestInit(r *http.Request, prog *ail.Program) {
	traceID, _ := r.Context().Value(plugin.ContextTraceID()).(string)
	if traceID == "" {
		return
	}

	// Derive a stable hash from the binary encoding of the initial request.
	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		Logger.Error("SAMPLER: encode failed for request", zap.Error(err))
		return
	}
	sum := sha256.Sum256(buf.Bytes())
	hash := hex.EncodeToString(sum[:])

	s.hashes.Store(traceID, hash)

	detailsDir := filepath.Join(s.Dir, hash)
	if err := os.MkdirAll(detailsDir, 0o755); err != nil {
		Logger.Error("SAMPLER: failed to create directory", zap.String("dir", detailsDir), zap.Error(err))
		return
	}

	binPath := filepath.Join(detailsDir, "request.ail")
	if _, err := os.Stat(binPath); err == nil {
		// Already sampled this exact request — directory exists; skip writing.
		Logger.Debug("SAMPLER: duplicate request, skipping write", zap.String("hash", hash))
		return
	}

	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		Logger.Error("SAMPLER: write request binary failed", zap.String("path", binPath), zap.Error(err))
		return
	}

	txtPath := filepath.Join(s.Dir, hash+".txt")
	if err := os.WriteFile(txtPath, []byte(prog.Disasm()), 0o644); err != nil {
		Logger.Error("SAMPLER: write request disasm failed", zap.String("path", txtPath), zap.Error(err))
		return
	}

	Logger.Debug("SAMPLER: saved request", zap.String("hash", hash))
}

// Before is called after all other before-plugins have run (sampler lives in
// TailPlugins). The prog received is the upstream-prepared program — i.e.
// the state that will actually be sent to the provider.
func (s *Sampler) Before(_ string, _ *services.ProviderService, r *http.Request, prog *ail.Program) (*ail.Program, error) {
	traceID, _ := r.Context().Value(plugin.ContextTraceID()).(string)
	hashVal, ok := s.hashes.Load(traceID)
	if !ok {
		return prog, nil
	}
	hash := hashVal.(string)

	detailsDir := filepath.Join(s.Dir, hash)

	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		Logger.Error("SAMPLER: encode failed for upstream request", zap.Error(err))
		return prog, nil
	}

	binPath := filepath.Join(detailsDir, "request.up.ail")
	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		Logger.Error("SAMPLER: write upstream binary failed", zap.String("path", binPath), zap.Error(err))
		return prog, nil
	}

	txtPath := filepath.Join(s.Dir, hash+".txt")
	f, err := os.OpenFile(txtPath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		Logger.Error("SAMPLER: open disasm file failed", zap.String("path", txtPath), zap.Error(err))
		return prog, nil
	}
	_, _ = f.WriteString("\n\n--- --- ---\n\n; upstream request\n" + prog.Disasm())
	_ = f.Close()

	Logger.Debug("SAMPLER: saved upstream request", zap.String("hash", hash))
	return prog, nil
}

// After is called after a complete (non-streaming) response is received.
func (s *Sampler) After(_ string, _ *services.ProviderService, r *http.Request, _ *ail.Program, _ *http.Response, resProg *ail.Program) (*ail.Program, error) {
	s.writeResponse(r, resProg)
	return resProg, nil
}

// StreamEnd is called once the stream is fully received.
// lastChunk is the final chunk; the caller is expected to pass the assembled
// program via the StreamEnd hook signature.
func (s *Sampler) StreamEnd(_ string, _ *services.ProviderService, r *http.Request, _ *ail.Program, _ *http.Response, lastChunk *ail.Program) error {
	s.writeResponse(r, lastChunk)
	return nil
}

// writeResponse persists the response AIL and appends its disassembly.
func (s *Sampler) writeResponse(r *http.Request, prog *ail.Program) {
	if prog == nil {
		return
	}
	traceID, _ := r.Context().Value(plugin.ContextTraceID()).(string)
	hashVal, ok := s.hashes.Load(traceID)
	if !ok {
		return
	}
	hash := hashVal.(string)
	defer s.hashes.Delete(traceID)

	detailsDir := filepath.Join(s.Dir, hash)

	var buf bytes.Buffer
	if err := prog.Encode(&buf); err != nil {
		Logger.Error("SAMPLER: encode failed for response", zap.Error(err))
		return
	}

	binPath := filepath.Join(detailsDir, "response.ail")
	if err := os.WriteFile(binPath, buf.Bytes(), 0o644); err != nil {
		Logger.Error("SAMPLER: write response binary failed", zap.String("path", binPath), zap.Error(err))
		return
	}

	txtPath := filepath.Join(s.Dir, hash+".txt")
	f, err := os.OpenFile(txtPath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		Logger.Error("SAMPLER: open disasm file failed for response", zap.String("path", txtPath), zap.Error(err))
		return
	}
	_, _ = f.WriteString("\n\n--- --- ---\n\n; response\n" + prog.Disasm())
	_ = f.Close()

	Logger.Debug("SAMPLER: saved response", zap.String("hash", hash))
}

// Ensure Sampler implements the required plugin interfaces.
var (
	_ plugin.RequestInitPlugin = (*Sampler)(nil)
	_ plugin.BeforePlugin      = (*Sampler)(nil)
	_ plugin.AfterPlugin       = (*Sampler)(nil)
	_ plugin.StreamEndPlugin   = (*Sampler)(nil)
)
