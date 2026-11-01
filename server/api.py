"""OpenAI-compatible FastAPI server for DeepSeek's web chat.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(model="deepseek-chat", messages=[...])

Endpoints:
    GET    /v1/models
    POST   /v1/chat/completions      (stream, tools, reasoning, web search)
    POST   /v1/files                 (multipart upload)
    GET    /v1/files/{id}
    GET    /v1/files/{id}/content
    DELETE /v1/files/{id}
    GET    /healthz                  (liveness, rate-limit exempt)
    GET    /readyz                   (deep check: session usable)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import re
import threading
import time
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired, get_session
from deepseek.client import DeepSeekAPIError, DeepSeekClient
from deepseek.files import FileError, FileRejected, FileService

from .config import (
    MODELS,
    RATE_LIMIT_EXEMPT_LOCALHOST,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    SESSION_PREWARM,
    THREAD_TTL_HOURS,
    UPSTREAM_SERIALIZE,
    resolve_model,
)
from .content import as_message_dicts, canonical_hash, file_parts, image_parts, text_of
from .openai_format import (
    Answer,
    collect_answer,
    completion_response,
    extract_system,
    messages_to_prompt,
    messages_to_prompt_tail,
    stream_chunks,
    tail_messages,
    with_tools_block,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest
from .threads import ThreadStore
from .toolbridge import tool_choice_mode

load_dotenv()

_log = logging.getLogger("deepseek.bridge")

ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = ROOT / "session"
MAX_MEDIA_BYTES = 25 * 1024 * 1024


class Services:
    def __init__(self, client: DeepSeekClient, threads: ThreadStore, files: FileService):
        self.client = client
        self.threads = threads
        self.files = files

    def close(self) -> None:
        self.files.close()
        self.client.close()


_services: Optional[Services] = None
_services_lock = threading.Lock()


def get_services() -> Services:
    """Build (once) the shared client, thread store and file service."""
    global _services
    if _services is None:
        with _services_lock:
            if _services is None:
                client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
                SESSION_DIR.mkdir(parents=True, exist_ok=True)
                threads = ThreadStore(
                    SESSION_DIR / "bridge.db",
                    ttl_seconds=THREAD_TTL_HOURS * 3600.0,
                )
                files = FileService(client, store_path=SESSION_DIR / "files.json")
                _services = Services(client, threads, files)
    return _services


def _prewarm() -> None:
    try:
        get_session(allow_interactive=True)
    except Exception as exc:
        _log.warning("session prewarm failed: %s", exc)
    try:
        get_services()
    except Exception as exc:
        _log.warning("service prewarm failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SESSION_PREWARM:
        threading.Thread(target=_prewarm, name="prewarm", daemon=True).start()
    yield
    if _services is not None:
        _services.close()


app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.2.0", lifespan=lifespan)
install_rate_limit(
    app,
    RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0),
    exempt_localhost=RATE_LIMIT_EXEMPT_LOCALHOST,
)

_upstream_lock = threading.Lock() if UPSTREAM_SERIALIZE else None

# Serialises the whole completion flow (plan -> upstream request -> save).
# UPSTREAM_SERIALIZE only protects the stream iterator; two requests could
# still plan from the same parent and overwrite each other's thread row.
_flow_lock = threading.Lock()


def _serialized(func):
    """Hold the flow lock until a streaming response has been fully sent."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        await run_in_threadpool(_flow_lock.acquire)
        try:
            response = await func(*args, **kwargs)
        except BaseException:
            _flow_lock.release()
            raise

        if not isinstance(response, StreamingResponse):
            _flow_lock.release()
            return response

        original = response.body_iterator

        async def body():
            try:
                async for chunk in original:
                    yield chunk
            finally:
                _flow_lock.release()

        response.body_iterator = body()
        return response

    return wrapper


class _LockedStream:
    """Serialise upstream generations over the single signed-in account."""

    def __init__(self, stream, lock: threading.Lock):
        self._stream = stream
        self._lock = lock

    def __iter__(self):
        with self._lock:
            yield from self._stream

    @property
    def state(self):
        return self._stream.state

    @property
    def session_id(self) -> str:
        return self._stream.session_id

    @property
    def message_id(self):
        return self._stream.message_id

    @property
    def conversation_id(self) -> str:
        return self._stream.conversation_id


def _serialize(stream):
    return _LockedStream(stream, _upstream_lock) if _upstream_lock else stream


def _error(message: str, status: int = 500, err_type: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


def _sse_error(exc: Exception) -> str:
    payload = {
        "error": {
            "message": str(exc),
            "type": "api_error",
        }
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _map_exception(exc: Exception) -> JSONResponse:
    if isinstance(exc, LoginRequired):
        return _error(str(exc), 503, "login_required")
    if isinstance(exc, FileRejected):
        return _error(str(exc), 400, "invalid_request_error")
    if isinstance(exc, FileError):
        return _error(f"File error: {exc}", 400, "invalid_request_error")
    if isinstance(exc, DeepSeekAPIError):
        if exc.is_rate_limit or exc.status == 429:
            response = _error(str(exc), 429, "rate_limit_error")
            response.headers["Retry-After"] = "5"
            return response
        if exc.biz_code in (3, 4, 9):
            return _error(str(exc), 400, "invalid_request_error")
        if exc.status and 400 <= exc.status < 500:
            return _error(str(exc), exc.status, "invalid_request_error")
        return _error(str(exc), 502, "api_error")
    if isinstance(exc, httpx.HTTPError):
        return _error(f"Upstream connection error: {exc}", 502, "api_error")
    return _error(f"Internal bridge error: {exc}", 500, "server_error")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


_ready_cache: Dict[str, Any] = {"at": 0.0, "payload": None, "status": 200}


@app.get("/readyz")
async def readyz():
    if time.time() - _ready_cache["at"] < 30.0 and _ready_cache["payload"] is not None:
        return JSONResponse(status_code=_ready_cache["status"], content=_ready_cache["payload"])

    try:
        services = await run_in_threadpool(get_services)
        user = await run_in_threadpool(services.client.current_user)
        payload = {
            "status": "ready",
            "user": user.get("email") or user.get("id"),
            "threads": "ok",
        }
        status = 200
    except Exception as exc:
        payload = {"status": "unavailable", "detail": str(exc)}
        status = 503

    _ready_cache.update({"at": time.time(), "payload": payload, "status": status})
    return JSONResponse(status_code=status, content=payload)


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": spec.id,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "name": spec.label,
                "context_length": spec.context,
                "capabilities": {
                    "tool_call": True,
                    "reasoning": spec.thinking,
                    "web_search": spec.search,
                    "vision": spec.vision,
                },
            }
            for spec in MODELS.values()
        ],
    }


def _thread_key(messages: List[Dict], model_id: str, thinking: bool, search: bool, tools) -> str:
    from .openai_format import extract_system
    from .toolbridge import tools_fingerprint

    blob = json.dumps(
        {
            "system": extract_system(messages),
            "model": model_id,
            "thinking": thinking,
            "search": search,
            "tools": tools_fingerprint(tools),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _profile_hash(messages: List[Dict]) -> str:
    return hashlib.sha256(extract_system(messages).encode("utf-8")).hexdigest()[:32]


def _thread_anchor(messages: List[Dict]) -> str:
    """Conversation identity: the first two non-system messages.

    Using only the first message made two unrelated sessions that opened with
    the same greeting (or the same bootstrap prompt) share a thread.  Two
    messages make that collision much less likely while still surviving a
    system-prompt or tool-set change.
    """
    picked = []
    for message in messages:
        if (message.get("role") or "") in ("system", "developer"):
            continue
        picked.append(message)
        if len(picked) == 2:
            break
    if not picked:
        return ""
    return canonical_hash(picked)[:32]


_EDIT_RE = re.compile(r"^@edit(?:\s+(\d+))?(?:\s+([\s\S]+))?$", re.IGNORECASE)

_DIRECTIVE_RE = re.compile(
    r"^\s*[@!/#](think|piensa|search|busca|web)\b[\s,:-]*",
    re.IGNORECASE,
)


def _apply_directives(messages: List[Dict]) -> tuple[List[Dict], Optional[bool], Optional[bool]]:
    """`@think` / `@search` como prefijo del ultimo turno de usuario.

    Devuelve (mensajes sin el prefijo, forzar_think, forzar_search). Se aceptan
    varios sigilos porque opencode usa `@` para mencionar archivos y puede
    interceptarlo: `@think`, `!think`, `/think` y `#think` son equivalentes.
    """
    if not messages:
        return messages, None, None
    last = messages[-1]
    if (last.get("role") or "") != "user":
        return messages, None, None

    text = text_of(last.get("content"))
    think: Optional[bool] = None
    search: Optional[bool] = None
    changed = False

    while True:
        match = _DIRECTIVE_RE.match(text)
        if not match:
            break
        changed = True
        token = match.group(1).lower()
        if token in ("think", "piensa"):
            think = True
        else:
            search = True
        text = text[match.end():]

    if not changed:
        return messages, None, None

    stripped = text.strip() or "Responde a mi mensaje anterior."
    out = list(messages)
    out[-1] = {**last, "content": stripped}
    return out, think, search



def _apply_edit_marker(messages: List[Dict]) -> List[Dict]:
    """`@edit [N] instrucción` — rewind to an earlier answer and steer it.

    DeepSeek has no "modify this message" call: a turn always appends a user
    message and an answer. So the marker truncates the history at the target
    answer and appends the instruction as the new user turn; the planner then
    sees the divergence and forks from the right ancestor, reusing the session.
    """
    if not messages:
        return messages
    last = messages[-1]
    if (last.get("role") or "") != "user":
        return messages
    text = text_of(last.get("content")).strip()
    match = _EDIT_RE.match(text)
    if not match:
        return messages

    count = int(match.group(1)) if match.group(1) else 1
    instruction = (match.group(2) or "").strip() or "Reescribe tu respuesta anterior, corrigiéndola."
    assistants = [i for i, m in enumerate(messages) if (m.get("role") or "") == "assistant"]
    if len(assistants) < count:
        return messages

    target = assistants[-count]
    trimmed = list(messages[:target])
    trimmed.append({"role": "user", "content": instruction})
    return trimmed


def _tool_reminder(tools) -> str:
    names = []
    for tool in tools or []:
        function = tool.function if hasattr(tool, "function") else None
        name = getattr(function, "name", None)
        if name:
            names.append(name)
    if not names:
        return ""
    return (
        "\n\nRecordatorio de herramientas: para llamar una usa "
        '<tool_call>{"name":"...","arguments":{...}}</tool_call>. '
        f"Disponibles: {', '.join(names)}."
    )


def _resolve_attachments(services: Services, messages: List[Dict]) -> Tuple[List[str], bool]:
    ref_ids: List[str] = []
    has_image = False

    for message in messages:
        content = message.get("content")
        for part in file_parts(content):
            file_id = _file_id_of(part)
            if file_id:
                ref_ids.append(file_id)
        for part in image_parts(content):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url:
                continue
            data, mime, filename = _load_media(url)
            record = services.files.upload(data, filename, mime)
            ref_ids.append(record.id)
            has_image = True

    deduped: List[str] = []
    for file_id in ref_ids:
        if file_id not in deduped:
            deduped.append(file_id)
    return deduped, has_image


def _file_id_of(part: Dict) -> Optional[str]:
    if isinstance(part.get("file_id"), str):
        return part["file_id"]
    file_obj = part.get("file")
    if isinstance(file_obj, dict):
        for key in ("file_id", "id"):
            if isinstance(file_obj.get(key), str):
                return file_obj[key]
    return None


def _load_media(url: str) -> Tuple[bytes, str, str]:
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = header[5:].split(";")[0] or "application/octet-stream"
        data = base64.b64decode(payload)
        return data, mime, _filename_for(mime)

    if url.startswith(("http://", "https://")):
        response = httpx.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        data = response.content
        if len(data) > MAX_MEDIA_BYTES:
            raise FileError(f"Remote media exceeds {MAX_MEDIA_BYTES} bytes")
        mime = response.headers.get("content-type", "").split(";")[0]
        mime = mime or mimetypes.guess_type(url)[0] or "application/octet-stream"
        return data, mime, _filename_for(mime)

    path = Path(url[7:] if url.startswith("file://") else url)
    if not path.exists():
        raise FileError(f"Media not found: {url}")
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return data, mime, path.name


def _filename_for(mime: str) -> str:
    extension = mimetypes.guess_extension(mime) or ".bin"
    return "image" + extension if mime.startswith("image/") else "upload" + extension


def _assistant_record(result: Optional[Dict[str, Any]], answer: Answer = None) -> Dict[str, Any]:
    if answer is not None:
        text = answer.text
        calls = [
            {"function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
            for c in answer.calls
        ]
    else:
        text = (result or {}).get("text") or ""
        calls = [
            {"function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in (result or {}).get("calls") or []
        ]
    record: Dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        record["tool_calls"] = calls
    return record


def _remember_thread(
    services: Services,
    *,
    key: str,
    stream,
    messages: List[Dict],
    model_id: str,
    assistant: Dict[str, Any],
    anchor: Optional[str] = None,
    profile_hash: Optional[str] = None,
    send_from: int = 0,
) -> None:
    try:
        state = getattr(stream, "state", None)
        request_id = getattr(state, "request_message_id", None)
        response_id = getattr(state, "response_message_id", None) or stream.message_id

        message_ids: Dict[int, Optional[int]] = {}
        if messages and request_id is not None:
            message_ids[len(messages) - 1] = request_id
        if response_id is not None:
            message_ids[len(messages)] = response_id

        services.threads.save(
            key=key,
            session_id=stream.session_id,
            last_message_id=response_id,
            messages_seen=messages + [assistant],
            model_id=model_id,
            anchor=anchor,
            profile_hash=profile_hash,
            message_ids=message_ids,
            keep_ids_before=send_from,
            last_response=json.dumps(assistant, ensure_ascii=False),
        )
    except Exception as exc:
        _log.warning("could not persist thread: %s", exc)


@app.post("/v1/chat/completions")
@_serialized
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", 400, "invalid_request_error")

    try:
        spec = resolve_model(req.model)
    except KeyError:
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODELS)}",
            404,
            "model_not_found",
        )

    messages = as_message_dicts(req.messages)
    messages, want_think, want_search = _apply_directives(messages)
    messages = _apply_edit_marker(messages)
    thinking = spec.thinking if req.thinking is None else bool(req.thinking)
    search = spec.search if req.search is None else bool(req.search)
    if want_think:
        thinking = True
    if want_search:
        search = True
    mode, _forced = tool_choice_mode(req.tool_choice)
    tools_enabled = bool(req.tools) and mode != "none"
    tool_names = [
        tool.function.name
        for tool in (req.tools or [])
        if getattr(getattr(tool, "function", None), "name", None)
    ]

    try:
        services = await run_in_threadpool(get_services)
    except Exception as exc:
        return _map_exception(exc)

    key = _thread_key(messages, spec.id, thinking, search, req.tools if tools_enabled else None)
    anchor = _thread_anchor(messages)
    profile = _profile_hash(messages)
    plan = None
    if req.conversation_id:
        slice_messages = tail_messages(messages)
    else:
        plan = await run_in_threadpool(services.threads.plan, key, messages, anchor, profile)
        slice_messages = messages[plan.send_from:]

    try:
        ref_ids, has_image = await run_in_threadpool(
            _resolve_attachments, services, slice_messages
        )
    except Exception as exc:
        return _map_exception(exc)

    model_type = "vision" if has_image else spec.model_type
    try:
        request_kwargs = dict(thinking=thinking, search=search, ref_file_ids=ref_ids)

        if req.conversation_id:
            prompt = messages_to_prompt_tail(slice_messages)
            if tools_enabled:
                prompt += _tool_reminder(req.tools)
            stream = services.client.request(
                prompt, conversation_id=req.conversation_id, **request_kwargs
            )
        elif plan is not None and plan.mode in ("append", "branch"):
            prompt = messages_to_prompt_tail(slice_messages)
            if plan.profile_changed:
                system_text = extract_system(messages)
                if system_text.strip():
                    prompt = f"System:\n{system_text}\n\n{prompt}"
            if tools_enabled:
                # Re-send the protocol reminder every turn: the model drifts and
                # an agent loop that forgets the format stalls.
                prompt += _tool_reminder(req.tools)
            stream = services.client.request(
                prompt,
                session_id=plan.thread.session_id,
                parent_id=plan.parent_id,
                model=model_type,
                **request_kwargs,
            )
        else:
            prompt_messages = (
                with_tools_block(messages, req.tools, req.tool_choice)
                if tools_enabled
                else messages
            )
            prompt = messages_to_prompt(prompt_messages)
            stream = services.client.request(prompt, model=model_type, **request_kwargs)
    except Exception as exc:
        return _map_exception(exc)

    serialized = _serialize(stream)

    if req.stream:
        def generate():
            outcome: Dict[str, Any] = {}
            try:
                yield from stream_chunks(
                    model=req.model,
                    stream=serialized,
                    prompt=prompt,
                    include_usage=req.includes_usage(),
                    tools_enabled=tools_enabled,
                    tool_names=tool_names,
                    result=outcome,
                )
            except Exception as exc:
                _log.warning("stream failed: %s", exc)
                yield _sse_error(exc)
            finally:
                _remember_thread(
                    services,
                    key=key,
                    stream=stream,
                    messages=messages,
                    model_id=spec.id,
                    assistant=_assistant_record(outcome),
                    anchor=anchor,
                    profile_hash=profile,
                    send_from=(plan.send_from if plan else 0),
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        answer = await run_in_threadpool(
            lambda: collect_answer(
                serialized,
                tools_enabled=tools_enabled,
                tool_names=tool_names,
            )
        )
    except Exception as exc:
        return _map_exception(exc)

    _remember_thread(
        services,
        key=key,
        stream=stream,
        messages=messages,
        model_id=spec.id,
        assistant=_assistant_record(None, answer),
        anchor=anchor,
        profile_hash=profile,
        send_from=(plan.send_from if plan else 0),
    )

    return completion_response(
        model=req.model,
        content=answer.text,
        prompt=prompt,
        conversation_id=stream.conversation_id,
        reasoning=answer.reasoning,
        tool_calls=answer.calls,
        search_results=answer.state.sorted_results() if answer.state else [],
        token_usage=answer.state.token_usage if answer.state else None,
    )


@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        services = await run_in_threadpool(get_services)
        record = await run_in_threadpool(
            services.files.upload, data, file.filename or "upload.bin", file.content_type
        )
    except Exception as exc:
        return _map_exception(exc)
    return record.to_openai()


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str):
    try:
        services = await run_in_threadpool(get_services)
        record = await run_in_threadpool(services.files.get, file_id)
    except Exception as exc:
        return _map_exception(exc)
    return record.to_openai()


@app.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    try:
        services = await run_in_threadpool(get_services)
        data, content_type, filename = await run_in_threadpool(
            services.files.download, file_id
        )
    except Exception as exc:
        return _map_exception(exc)
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    try:
        services = await run_in_threadpool(get_services)
        await run_in_threadpool(services.files.forget, file_id)
    except Exception as exc:
        return _map_exception(exc)
    return {"id": file_id, "object": "file", "deleted": True}


