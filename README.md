# DeepSeek API: a free LLM API powered by DeepSeek

**Using your own DeepSeek account.** No API key, no credits, no paid plan: it turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an API you can call from code.

You can use it in two ways:

- 🐍 **As a Python library:** just call `client.chat("Hi")`. Supports streaming and multi-turn conversations.
- 🔌 **As a local OpenAI-compatible API:** runs a server at `http://localhost:8000/v1` that speaks the OpenAI format, so the official `openai` SDK (and any OpenAI-compatible app) works as a drop-in, with `localhost` in place of OpenAI.

You sign in once in a browser with your DeepSeek account; your session is saved and refreshed automatically after that.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It automates the consumer DeepSeek web experience for personal use, so use it responsibly and within DeepSeek's terms.

---

## Table of contents

- [Why use this?](#why-use-this)
- [Requirements](#requirements)
- [Setup (2 minutes)](#setup-2-minutes)
- [Usage 1: In Python (no server)](#usage-1-in-python-no-server)
- [Usage 2: As an OpenAI-compatible server](#usage-2-as-an-openai-compatible-server)
- [Command line](#command-line)
- [Human-check & proof-of-work (automatic)](#human-check--proof-of-work-automatic)
- [Models, DeepThink & web search](#models-deepthink--web-search)
- [Concurrency](#concurrency)
- [Rate limiting](#rate-limiting)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [License](#license)

---

## Why use this?

- **Free:** uses your normal signed-in DeepSeek account, no API billing.
- **Drop-in OpenAI replacement:** point any OpenAI client at `localhost` and it just works.
- **Full DeepSeek toolset:** pick the fast or expert model, and toggle DeepThink reasoning and web search per request.
- **Streaming + conversations:** token-by-token output and multi-turn threads addressed by `conversation_id`.

---

## Requirements

- **Python 3.9+**
- A **DeepSeek account** (the free one you use for [chat.deepseek.com](https://chat.deepseek.com) is fine)
- Works on Windows, macOS, and Linux

---

## Setup (2 minutes)

```bash
# 1. Clone the project
git clone https://github.com/sums001/Deepseek-API.git
cd "Deepseek-API"
```

**2. Create and activate a virtual environment**

On **macOS / Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

On **Windows** (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> On Windows you may need to allow script execution once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activate with `venv\Scripts\activate.bat` instead.

**3. Install dependencies and sign in**

```bash
# Install dependencies
pip install -r requirements.txt

# Install the browser Playwright needs (one-time)
playwright install chromium

# Sign in once: a browser opens, log into your DeepSeek account
python -m deepseek.auth
```

The login window opens so you can sign in by hand and solve the human-check once. After that your session (bearer token + cookies) is saved under `session/` (git-ignored, never shared) and reused on every run — the cached session is refreshed automatically, so your first request works right away.

> The server can also open this window for you on demand the first time it needs a session, so this step is optional for local single-user use.

---

## Usage 1: In Python (no server)

The simplest way if your code is already Python.

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # loads your signed-in session

# Get a full reply
reply = client.chat("Say hello in one short sentence.")
print(reply.text)

# Continue the SAME conversation — pass the id back
reply2 = client.chat("And now in French?", conversation_id=reply.conversation_id)
print(reply2.text)

# Stream the answer as it's typed
for chunk in client.stream("Tell me a short joke"):
    print(chunk, end="", flush=True)
```

`chat()` returns the full text plus a `conversation_id`; pass that id back to keep the thread going, or omit it to start fresh. `stream()` yields the reply piece by piece.

👉 More: [examples/01_direct_chat.py](examples/01_direct_chat.py), [02_direct_conversation.py](examples/02_direct_conversation.py), [03_direct_stream.py](examples/03_direct_stream.py)

---

## Usage 2: As an OpenAI-compatible server

Start a local server that speaks the OpenAI API, so existing OpenAI tools and SDKs work unchanged.

```bash
python app.py
# -> DeepSeek OpenAI-compatible API on http://127.0.0.1:8000
```

Then point any OpenAI client at it (the API key is required by the SDK but ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Or call it with plain HTTP / `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Endpoints**

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat (supports `"stream": true`, plus optional `"conversation_id"`, `"thinking"`, `"search"`) |
| `POST` | `/v1/files` | Upload an attachment (multipart `file`), returns an OpenAI-style file object |
| `GET`  | `/v1/files/{id}` | File metadata / parse status |
| `GET`  | `/v1/files/{id}/content` | Download the stored file |
| `DELETE` | `/v1/files/{id}` | Forget a file locally |
| `GET`  | `/v1/models` | Lists the available models |
| `GET`  | `/healthz` | Liveness (rate-limit exempt) |
| `GET`  | `/readyz` | Deep check: is the DeepSeek session usable? |

> Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 More: [examples/04_server_http.py](examples/04_server_http.py), [examples/05_server_stream.py](examples/05_server_stream.py), [examples/06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

---

## Command line

```bash
python -m deepseek.auth          # sign in and save the session
```

---

## Human-check & proof-of-work (automatic)

DeepSeek's chat sits behind two gates, both handled for you:

- **AWS WAF human-check:** access needs a signed-in browser session that has
  cleared the "verify you're human" check. `python -m deepseek.auth` opens a real
  browser so you can sign in and solve it once; the resulting token + cookies are
  cached under `session/` and reused on every request.
- **Proof-of-work:** every completion is gated by a PoW challenge. The bridge
  solves it by running DeepSeek's own `sha3_wasm_bg.wasm` module — the same one
  the browser loads — inside a `wasmtime` sandbox, so there's nothing to do on
  your end.

A cached session is reused for ~6 hours and refreshed headlessly from your saved
Chrome profile when possible; only a full expiry sends you back to the browser.

---

## Models, DeepThink & web search

The `model` name selects **which model** answers. DeepThink and web search are
**not** models — they're orthogonal toggles you pass per request.

| Model | DeepSeek mode | Notes |
| --- | --- | --- |
| `deepseek-chat` | Instant | Fast default model |
| `deepseek-expert` | Expert | Stronger, slower |

Pass `thinking: true` (DeepThink reasoning) and/or `search: true` (web search) in
the request body — or via the OpenAI SDK's `extra_body`:

```python
resp = client.chat.completions.create(
    model="deepseek-expert",
    messages=[{"role": "user", "content": "What changed in the news today?"}],
    extra_body={"thinking": True, "search": True},
)
```

`conversation_id`, `thinking`, and `search` are non-OpenAI extras. A thread's
model is fixed at creation, so `model` can't be combined with `conversation_id`
on resume. Unknown model names return a `404` (no silent fallback). See
[server/config.py](server/config.py).

### Model aliases

Because the model and the two toggles are orthogonal, each combination is
exposed as its own id:

| Id | Model | DeepThink | Web |
| --- | --- | --- | --- |
| `deepseek-chat` | Instant | – | – |
| `deepseek-chat-think` | Instant | ✓ | – |
| `deepseek-chat-search` | Instant | – | ✓ |
| `deepseek-chat-think-search` | Instant | ✓ | ✓ |
| `deepseek-expert` / `-think` / `-search` / `-think-search` | Expert | per id | per id |
| `deepseek-vision` | Vision (images/files) | – | – |

Aliases `deepseek-reasoner` and `deepseek-r1` map to `deepseek-chat-think`.

---

## Agent bridge: tools, files and multi-turn

Pointing a coding agent (opencode, Cline, …) at this bridge needs three things
that the raw DeepSeek web chat does not offer. All three are implemented here:

**1. Tool calling (emulated).** `chat.deepseek.com` has no function-calling
channel — a `tools` field is silently ignored by the backend. The bridge
serialises the tool definitions into the prompt, tells the model to answer with
`<tool_call>{"name":…,"arguments":…}</tool_call>` blocks, then parses those
blocks back into OpenAI `tool_calls` (with `finish_reason: "tool_calls"`, in
both streaming and non-streaming mode). It also understands DeepSeek's native
DSML dialect (`<||DSML|| invoke name="…"> … </||DSML|| invoke>`), which the model
emits on its own when a host like opencode presents its tool schemas. Tool
results are rendered back into the next turn, so agent loops work. See [server/toolbridge.py](server/toolbridge.py).

**2. Real multi-turn threading.** Instead of re-sending the whole transcript,
the bridge maps an OpenAI history to a server-side DeepSeek session and resumes
it with `parent_message_id`, sending only the new turn.

Threads live in `session/bridge.db` and carry two identifiers:

- **key** — system prompt + model + toggles + tool set.
- **anchor** — hash of the first non-system message, i.e. the conversation
  itself. If the agent rewrites its system prompt or the tool set mid-flight,
  the anchor still finds the thread instead of opening a new session.

Every message DeepSeek assigns an id to is recorded, so the bridge can compare
the incoming history against the chain it already saw and choose:

| Situation | What happens |
| --- | --- |
| History extends the thread | **append** — only the new turn is sent |
| A message was edited / retried | **branch** — rewind to the last common ancestor and resend only from there (DeepSeek stores it as a sibling branch) |
| No common prefix | **new session** |

Editing used to open a brand-new session and replay the entire transcript;
now it costs a single turn.

**`@edit` marker.** Put `@edit [N] <instruction>` as the last user message to
rewind to the N-th answer from the end (default 1) and steer it:

```text
@edit 1 sé más breve y cita la fuente
```

The bridge truncates the history at that answer and sends the instruction as
the new turn, so the fork is reused instead of replayed.

Set `THREAD_TTL_HOURS` to control retention.

### Turn directives

Capabilities are toggled **per turn**, without switching models or starting a
new chat. Write the directive at the very start of the message:

| Directive | Effect |
| --- | --- |
| `@think <text>` | enables DeepThink reasoning for this turn |
| `@search <text>` | runs DeepSeek's server-side web search |
| `@think @search <text>` | both |
| `@edit [N] <text>` | rewinds to the N-th answer (default 1) and rewrites it |

Aliases are accepted because opencode binds `@` to file mentions and may swallow
it: `!think`, `/think`, `#think`, `@piensa`, `@busca`, `@web`.

The directive is stripped before the prompt is built, and the flags are sent in
the completion body, so the thread is reused. Verified against the live backend:
a second turn on an existing thread returned `reasoning_content` (~500 chars)
and `annotations` with real sources.

### Context guarantees (agent loops)

Verified end-to-end over the HTTP surface that opencode uses:

| Behaviour | Result |
| --- | --- |
| Several tools in one assistant turn | parsed and returned as multiple `tool_calls` |
| Tool results fed back | rendered into the next turn; the loop continues |
| Context after a tool round | the next question is answered without re-reading |
| History extends | **append** — only the new turn travels upstream |
| Message edited / retried | **branch** — rewinds to the last common ancestor |

What can still reset context, and why:

- **`THREAD_TTL_HOURS` (12 by default).** An idle thread is dropped; the next
  request starts a new session and replays the transcript once.
- **A failed session.** If DeepSeek invalidates the login, the request fails with
  `503 login_required`; the thread survives, so context resumes after re-login.
- **Tool output larger than `MAX_TOOL_RESULT_CHARS` (16000).** It is *condensed*,
  not blindly cut: a header with the real size, an index of structural lines
  (`def`, `class`, headings, `key: value` blocks) with line numbers, the
  beginning, the end, and a hint to re-read with offset/limit. Raise the limit
  with the env var if you prefer the raw text.
- **A fork point with no recorded id.** The bridge walks back to the nearest
  ancestor it knows; if there is none, it starts a new session.

The fork point is always an **assistant** message: a prompt creates a user turn,
and DeepSeek rejects user → user with `invalid message role`.

**3. Files and vision.** `POST /v1/files` uploads through DeepSeek's own
attachment endpoint (proof-of-work gated), waits for the file to finish parsing,
and returns an id. Attach it in a message as
`{"type":"file","file_id":"file-…"}`, or send an `image_url` part and the bridge
will upload it for you (switching the thread to the vision model).

**Reasoning and citations.** With `@think` (or a `-think` model), chain-of-thought
is returned as `reasoning_content` (streamed as `delta.reasoning_content`) instead
of being mixed into the answer. With `@search` (or a `-search` model), sources come
back in `search_results` + `annotations`, and a compact `Fuentes:` list is appended
to the answer so `[citation:N]` markers are resolvable.

**Recommended setup for opencode.** Expose exactly two models and let the
directives do the rest:

| Model id | Maps to | Use |
| --- | --- | --- |
| `deepseek` | Instant (default) | fast, "flash"-like |
| `deepseek-expert` | Expert | hard problems |

Both support tools, attachments, `@think`, `@search` and `@edit`. The `-think` /
`-search` aliases remain available but are no longer needed
(`deepseek-chat-think`, `deepseek-chat-search`, …).

> opencode configures tools per **agent**, not per model, so the bridge cannot
> intercept opencode's own `websearch`. `@search` uses DeepSeek's search instead,
> which runs server-side and returns `annotations`.

**3. Files and vision.** `POST /v1/files` uploads through DeepSeek's own
attachment endpoint (proof-of-work gated), waits for the file to finish parsing,
and returns an id. Attach it in a message as
`{"type":"file","file_id":"file-…"}`, or send an `image_url` part and the bridge
will upload it for you (switching the thread to the vision model).

**Reasoning and citations.** With a `-think` model, chain-of-thought is returned
as `reasoning_content` (streamed as `delta.reasoning_content`) instead of being
mixed into the answer. With a `-search` model, sources come back in
`search_results` + `annotations`, and a compact `Fuentes:` list is appended to
the answer so `[citation:N]` markers are resolvable.

**Reality check.** This is emulation over a consumer web app, not a model API:
upstream calls are serialised over one account, tool-calling is prompt-driven
(slightly less reliable than native), and DeepSeek's own rate limits still
apply. Use it in moderation.

---

## Concurrency

The server bridges a **single** signed-in DeepSeek account behind one shared
client. The PoW solver's `wasmtime` store isn't reentrant, so upstream calls are
**serialized**: parallel HTTP requests queue behind a lock and run one at a time
(see [server/api.py](server/api.py)). This is intentional — throughput is
sequential, not parallel. Keep concurrent in-flight requests low, and please
don't hammer your account.

---

## Rate limiting

On top of serialization, the bridge enforces a self-imposed rate limit with a
dependency-free sliding-window limiter ([server/ratelimit.py](server/ratelimit.py)):
it caps accepted requests **per client IP** and returns a standard `429` +
`Retry-After` when you exceed it. `/healthz` is exempt.

| Env var | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests/minute accepted per client IP |

```bash
RATE_LIMIT_PER_MINUTE=60 python app.py   # raise it
```

**On the client side, use exponential backoff.** Transient `429`s clear if you
retry with growing delays (e.g. 1s, 2s, 4s). The official `openai` SDK does this
automatically and honours `Retry-After`; with plain HTTP, add a few retries
yourself.

---

## Project layout

| Path | What it does |
| --- | --- |
| [deepseek/](deepseek/) | The core library: `DeepSeekClient`, auth/browser sign-in ([auth.py](deepseek/auth.py)), the HTTP driver ([client.py](deepseek/client.py)), and the PoW solver ([pow.py](deepseek/pow.py)) |
| [server/](server/) | The FastAPI OpenAI-compatible server |
| [examples/](examples/) | Runnable examples for every feature ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Starts the server |

---

## Notes & limitations

- **Sign in once, then reuse.** The cached session refreshes automatically; you only re-sign-in if it fully expires.
- **Be reasonable.** Please use it in moderation, and don't spam or hammer it with automated bulk requests.
- **Token counts are approximate on the prompt side.** Completion tokens come from DeepSeek's own counter; prompt tokens are estimated (~3 chars/token).
- **Most OpenAI params are accepted but ignored** (`temperature`, `top_p`, `max_tokens`); `model`, `messages`, `stream`, `tools`, `tool_choice`, `conversation_id`, `thinking` and `search` are the ones that do something.
- **Vision and files work** through `POST /v1/files` and `image_url`/`file` content parts.
- **Your session is private.** Everything in `session/` (cookies + token) stays on your machine and is git-ignored.

## License

Released under the [MIT License](LICENSE). As this is an unofficial project, you remain responsible for complying with DeepSeek's terms of service.

---



