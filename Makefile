.PHONY: build run clean tidy test test-formats test-styles test-plugins test-all test-server

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

# Run all unit tests
test:
	go test -v ./src/... -count=1

# Run format tests only
test-formats:
	go test -v ./src/formats/... -count=1

# Run style tests only
test-styles:
	go test -v ./src/styles/... -count=1

# Run plugin tests only
test-plugins:
	go test -v ./src/plugins/... -count=1

# Start test server (run in separate terminal)
test-server: build
	@echo "Starting test server on :19111..."
	@bash -c "[ -f .env ] && set -a && source .env && set +a && ./caddy run --config tests/Caddyfile.test"

# Run specific test by name
test-one:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-one TEST=TestName"; \
		exit 1; \
	fi
	go test -v ./src/... -run "$(TEST)" -count=1

# Run all tests with coverage
test-coverage:
	go test -v ./src/... -cover -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Run tests in short mode (skip slow tests)
test-short:
	go test -v ./src/... -short -count=1
