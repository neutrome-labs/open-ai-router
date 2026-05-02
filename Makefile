.PHONY: build run clean tidy test test-formats test-styles test-plugins test-all test-server workers-build workers-clean dspy-sidecar dspy-sidecar-install

build:
	go build -o caddy ./src

workers-build:
	mkdir -p dist/workers
	rm -f dist/workers/wasm_exec.js dist/workers/caddy.wasm dist/workers/worker.js dist/workers/wrangler.jsonc
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" dist/workers/wasm_exec.js
	GOOS=js GOARCH=wasm go build -ldflags "-s -w" -o dist/workers/caddy.wasm ./src
	cp worker.js dist/workers/worker.js
	cp wrangler.jsonc dist/workers/wrangler.jsonc

workers-deploy:
	cd dist/workers && bunx wrangler deploy

run: build
	bash -c "[ -f .env ] && set -a && source .env && set +a && ./caddy run --config Caddyfile"

clean:
	rm -f caddy

workers-clean:
	rm -rf dist/workers

tidy:
	go mod tidy

upgrade:
	GOPRIVATE=github.com/neutrome-labs go get github.com/neutrome-labs/ail

upgrade-all:
	go get -u ./...
	go mod tidy

test:
	go test ./src/...

coverage:
	go test -v ./src/... -cover -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

dspy-sidecar-install:
	pip3 install -r sidecar/dspy/requirements.txt

dspy-sidecar:
	python3 sidecar/dspy/main.py
