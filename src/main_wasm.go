//go:build js && wasm

package main

import (
	"log"
	"syscall/js"

	"github.com/caddyserver/caddy/v2/workerd"
	"github.com/syumai/workers"

	_ "github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	_ "github.com/caddyserver/caddy/v2/modules/caddyhttp/headers"
	_ "github.com/caddyserver/caddy/v2/modules/caddyhttp/requestbody"
	_ "github.com/caddyserver/caddy/v2/modules/caddyhttp/reverseproxy"
	_ "github.com/caddyserver/caddy/v2/modules/caddyhttp/rewrite"

	_ "github.com/neutrome-labs/open-ai-router/src/modules"
	_ "github.com/neutrome-labs/open-ai-router/src/modules/server"
)

func main() {
	server, cancel, err := workerd.BuildServer(workerd.CaddyfileFromEnv())
	if err != nil {
		panic(err)
	}
	defer cancel(nil)

	workers.ServeNonBlock(server)
	js.Global().Set("caddyWorkerFetch", js.Global().Get("context").Get("binding").Get("handleRequest"))

	workers.Ready()
	log.Println("[Caddy] Workers runtime ready")

	select {}
}
