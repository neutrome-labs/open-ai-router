//go:build !js && !wasm

package main

import (
	caddycmd "github.com/caddyserver/caddy/v2/cmd"

	_ "github.com/caddyserver/caddy/v2/modules/standard"
	_ "github.com/neutrome-labs/open-ai-router/src/modules"
	_ "github.com/neutrome-labs/open-ai-router/src/modules/server"
)

func main() {
	caddycmd.Main()
}
