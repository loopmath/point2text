# Tailnet hosting

The `--tailnet` CLI option resolves `tailscale ip -4` and binds directly to that address. It rejects missing/invalid addresses rather than falling back to a broader bind. Tailscale must already be connected and its CLI available on PATH. Existing Serve routes remain untouched.

```bash
uv run --extra encoder point2text --tailnet --port 8477 --model YOUR_INSTALLED_MODEL
```

Use the printed host address or `tailscale ip -4` to form `http://TAILNET_IP:8477`. The web interface and API work at the same origin; the host's Ollama service can remain on loopback. Tailnet access rules apply. The app itself is a single local worker without separate user authentication or tenant isolation.

## Background process on macOS

After installing dependencies, these shell commands keep the process alive independently of the terminal and prevent idle system sleep for that process's lifetime:

```bash
mkdir -p .cache/point2text
nohup caffeinate -i uv run --extra encoder point2text --tailnet --port 8477 --model YOUR_INSTALLED_MODEL > .cache/point2text/tailnet.log 2>&1 &
echo $! > .cache/point2text/tailnet.pid
```

Inspect `.cache/point2text/tailnet.log` for startup/errors. This does not configure automatic launch at login. Stop the owned process with its saved PID or the PID reported for this specific listener by `lsof -nP -iTCP:8477 -sTCP:LISTEN`; do not kill unrelated services. Model loading can take time on first launch; `/api/status` reports readiness. Pick a different port if 8477 is already occupied.

The background process and run/cache files are local runtime state, excluded from Git. All application code, bundled assets, and these hosting instructions live in the public repository.
