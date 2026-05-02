import wasmModule from "./caddy.wasm";
import "./wasm_exec.js";

let bootPromise;

async function loadCaddyfile(request, env) {
	const url = new URL("/Caddyfile", request.url);
	const response = await env.ASSETS.fetch(url);
	if (response.ok) {
		return response.text();
	}
	return "";
}

async function boot(request, env) {
	if (!bootPromise) {
		bootPromise = (async () => {
			const go = new Go();
			globalThis.context = globalThis.context || { binding: {}, env: {}, ctx: null };
			globalThis.context.env = env;
			go.env = go.env || {};
			for (const [key, value] of Object.entries(env)) {
				if (typeof value === "string") {
					go.env[key] = value;
				}
			}
			const caddyfile = await loadCaddyfile(request, env);
			if (caddyfile) {
				go.env.CADDYFILE = caddyfile;
			}

			go.importObject.workers = {
				ready: () => {
					const cb = globalThis.__caddy_on_ready;
					if (typeof cb === "function") {
						cb();
						globalThis.__caddy_on_ready = null;
					}
				},
			};

			const readyPromise = new Promise((resolve) => {
				globalThis.__caddy_on_ready = resolve;
			});
			const result = await WebAssembly.instantiate(wasmModule, go.importObject);
			go.run(result.instance || result);
			await readyPromise;
		})().catch((err) => {
			bootPromise = null;
			throw err;
		});
	}
	return bootPromise;
}

export default {
	async fetch(request, env, ctx) {
		await boot(request, env);
		globalThis.context.env = env;
		globalThis.context.ctx = ctx;
		if (typeof globalThis.caddyWorkerFetch !== "function") {
			throw new Error("Caddy fetch handler was not registered");
		}
		return globalThis.caddyWorkerFetch(request);
	},
};
