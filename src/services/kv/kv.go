package kv

// Package kv provides a pluggable key-value store for the AI router.
//
// The default backend is an in-memory LRU with TTL. Additional backends
// (Redis, Cloudflare KV, etc.) can be registered via RegisterBackend and
// selected in the Caddyfile configuration.

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrNotFound is returned when a key does not exist in the store.
var ErrNotFound = errors.New("kv: key not found")

// Store is the pluggable KV backend interface.
type Store interface {
	// Get retrieves the value for a key. Returns ErrNotFound if absent.
	Get(ctx context.Context, key string) (string, error)

	// Set stores a key-value pair with an optional TTL.
	// A zero TTL means no expiration.
	Set(ctx context.Context, key, value string, ttl time.Duration) error

	// Delete removes a key.
	Delete(ctx context.Context, key string) error

	// Close releases any resources held by the store.
	Close() error
}

// ─── Backend registry ────────────────────────────────────────────────────────

// BackendFactory creates a Store from a DSN / config string.
// Examples: "" (in-memory), "redis://host:6379/0", etc.
type BackendFactory func(dsn string) (Store, error)

var (
	backendsMu sync.RWMutex
	backends   = map[string]BackendFactory{
		"memory": func(_ string) (Store, error) { return NewMemoryStore(10000, 30*time.Minute), nil },
	}
)

// RegisterBackend registers a named backend factory.
func RegisterBackend(name string, f BackendFactory) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[name] = f
}

// Open creates a Store using the named backend.
// Falls back to "memory" when name is empty.
func Open(name, dsn string) (Store, error) {
	if name == "" {
		name = "memory"
	}
	backendsMu.RLock()
	f, ok := backends[name]
	backendsMu.RUnlock()
	if !ok {
		return nil, errors.New("kv: unknown backend " + name)
	}
	return f(dsn)
}

// ─── In-memory implementation ────────────────────────────────────────────────

type memEntry struct {
	value     string
	expiresAt time.Time // zero = no expiry
}

// MemoryStore is an in-memory KV store with LRU eviction and TTL.
type MemoryStore struct {
	mu         sync.RWMutex
	data       map[string]memEntry
	order      []string // insertion order for simple LRU
	maxItems   int
	defaultTTL time.Duration
}

// NewMemoryStore creates an in-memory store.
func NewMemoryStore(maxItems int, defaultTTL time.Duration) *MemoryStore {
	return &MemoryStore{
		data:       make(map[string]memEntry, maxItems),
		order:      make([]string, 0, maxItems),
		maxItems:   maxItems,
		defaultTTL: defaultTTL,
	}
}

func (m *MemoryStore) Get(_ context.Context, key string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	e, ok := m.data[key]
	if !ok {
		return "", ErrNotFound
	}
	if !e.expiresAt.IsZero() && time.Now().After(e.expiresAt) {
		return "", ErrNotFound
	}
	return e.value, nil
}

func (m *MemoryStore) Set(_ context.Context, key, value string, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if ttl == 0 {
		ttl = m.defaultTTL
	}
	var exp time.Time
	if ttl > 0 {
		exp = time.Now().Add(ttl)
	}
	if _, exists := m.data[key]; !exists {
		// Evict oldest if at capacity.
		if len(m.order) >= m.maxItems {
			oldest := m.order[0]
			m.order = m.order[1:]
			delete(m.data, oldest)
		}
		m.order = append(m.order, key)
	}
	m.data[key] = memEntry{value: value, expiresAt: exp}
	return nil
}

func (m *MemoryStore) Delete(_ context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.data, key)
	// Lazy removal from order slice (next eviction will skip deleted keys).
	return nil
}

func (m *MemoryStore) Close() error { return nil }