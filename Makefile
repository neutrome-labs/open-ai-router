.PHONY: build run clean tidy test test-formats test-styles test-plugins test-all test-server dspy-sidecar dspy-sidecar-install

build:
	go build -o caddy ./src

run: build
	bash -c "[ -f .env ] && set -a && source .env && set +a && ./caddy run --config Caddyfile"

clean:
	rm -f caddy

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
