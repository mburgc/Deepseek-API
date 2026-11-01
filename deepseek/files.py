"""File upload / status / download against chat.deepseek.com.

Uploading is a two-step protocol: a multipart POST gated by its own proof of
work, then polling `file/fetch_files` until the server finishes parsing the
file. A file id is only valid for `ref_file_ids` once its status is SUCCESS —
attaching a PENDING id fails with `biz_code 9 invalid ref file id`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from .client import DeepSeekAPIError, DeepSeekClient, _unwrap

UPLOAD_PATH = "/api/v0/file/upload_file"
FETCH_PATH = "/api/v0/file/fetch_files"
DEFAULT_FILES_HOST = "https://files.deepseeksvc.com"

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILURES = {
    "FAILED",
    "CONTENT_FILTER",
    "CONTENT_TOO_LONG",
    "CANCELLED",
    "CONTENT_EMPTY",
}
STATUS_PENDING = {"PENDING", "PARSING"}


class FileError(RuntimeError):
    pass


class FileRejected(FileError):
    def __init__(self, record: "FileRecord"):
        super().__init__(
            f"DeepSeek rejected {record.file_name!r}: {record.status}"
            + (f" ({record.error_code})" if record.error_code else "")
        )
        self.record = record


@dataclass
class FileRecord:
    id: str
    file_name: str
    status: str
    file_size: int = 0
    token_usage: Optional[int] = None
    is_image: bool = False
    signed_path: Optional[str] = None
    model_kind: Optional[str] = None
    error_code: Optional[str] = None
    mime: Optional[str] = None
    digest: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def ready(self) -> bool:
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        return self.status in STATUS_FAILURES

    def to_openai(self) -> Dict:
        return {
            "id": self.id,
            "object": "file",
            "bytes": self.file_size,
            "created_at": int(self.created_at),
            "filename": self.file_name,
            "status": self.status,
            "purpose": "vision" if self.is_image else "assistants",
        }


class FileService:
    def __init__(
        self,
        client: DeepSeekClient,
        *,
        files_host: str = DEFAULT_FILES_HOST,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
        store_path: Optional[Path] = None,
    ):
        self.client = client
        self.files_host = files_host.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.store_path = Path(store_path) if store_path else None
        self._cache: Dict[str, FileRecord] = {}
        self._lock = threading.Lock()
        self._download = httpx.Client(timeout=httpx.Timeout(120.0, read=300.0))
        self._load()

    # --- upload ---------------------------------------------------------

    def upload(self, data: bytes, filename: str, mime: Optional[str] = None) -> FileRecord:
        mime = mime or "application/octet-stream"
        digest = hashlib.sha256(data).hexdigest()

        reused = self._find_by_digest(digest)
        if reused is not None:
            return reused

        last_error: Optional[DeepSeekAPIError] = None
        for attempt in range(3):
            try:
                return self._upload_once(data, filename, mime, digest)
            except DeepSeekAPIError as exc:
                if not exc.is_rate_limit or attempt == 2:
                    raise
                last_error = exc
                time.sleep(10.0 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise FileError("Upload failed")

    def _upload_once(self, data: bytes, filename: str, mime: str, digest: str) -> FileRecord:
        headers = {"x-ds-pow-response": self.client.pow_header(UPLOAD_PATH)}
        body, content_type = _encode_multipart({"file": (filename, data, mime)})
        headers["content-type"] = content_type
        response = self.client._http.post(UPLOAD_PATH, content=body, headers=headers)
        if response.status_code >= 400:
            raise FileError(
                f"Upload failed with HTTP {response.status_code}: {response.text[:200]}"
            )
        biz = _unwrap(response.json(), response.status_code)
        record = _record_from(biz, mime)
        record.digest = digest
        self._remember(record)

        if record.failed:
            raise FileRejected(record)
        if not record.ready:
            record = self.wait_ready([record.id])[0]
        return record

    def _find_by_digest(self, digest: str) -> Optional[FileRecord]:
        with self._lock:
            for record in self._cache.values():
                if record.digest == digest and record.ready:
                    return record
        return None

    def wait_ready(self, file_ids: List[str]) -> List[FileRecord]:
        deadline = time.time() + self.timeout
        pending = list(file_ids)
        while pending:
            records = self.fetch(pending)
            pending = [r.id for r in records if not r.ready and not r.failed]
            failed = [r for r in records if r.failed]
            if failed:
                raise FileRejected(failed[0])
            if not pending:
                return records
            if time.time() >= deadline:
                raise FileError(
                    f"File(s) still {', '.join(r.status for r in records)} after "
                    f"{self.timeout:.0f}s: {', '.join(pending)}"
                )
            time.sleep(self.poll_interval)
        return self.fetch(file_ids)

    # --- status ---------------------------------------------------------

    def fetch(self, file_ids: List[str]) -> List[FileRecord]:
        ids = [fid for fid in file_ids if fid]
        if not ids:
            return []

        files = self._fetch_raw({"file_ids": ",".join(ids)})
        if not files:
            files = self._fetch_raw([("file_ids", fid) for fid in ids])

        records = []
        for item in files:
            record = _record_from(item, None)
            self._remember(record)
            records.append(record)
        return records

    def _fetch_raw(self, params) -> List[Dict]:
        response = self.client._http.get(FETCH_PATH, params=params)
        if response.status_code >= 400:
            raise FileError(
                f"File status failed with HTTP {response.status_code}: {response.text[:200]}"
            )
        biz = _unwrap(response.json(), response.status_code)
        return biz.get("files") or []

    def get(self, file_id: str) -> FileRecord:
        cached = self._cache.get(file_id)
        if cached is not None:
            return cached
        records = self.fetch([file_id])
        if not records:
            raise FileError(f"Unknown file id: {file_id}")
        return records[0]

    def forget(self, file_id: str) -> None:
        with self._lock:
            self._cache.pop(file_id, None)
            self._persist()

    # --- download -------------------------------------------------------

    def signed_url(self, record: FileRecord) -> str:
        if not record.signed_path:
            raise FileError(f"File {record.id} has no download path yet")
        if record.signed_path.startswith("http"):
            return record.signed_path
        return self.files_host + record.signed_path

    def download(self, file_id: str) -> Tuple[bytes, str, str]:
        record = self.get(file_id)
        url = self.signed_url(record)
        response = self._download.get(url, follow_redirects=True)
        if response.status_code >= 400:
            raise FileError(
                f"Download failed with HTTP {response.status_code} for {file_id}"
            )
        content_type = response.headers.get("content-type", "application/octet-stream")
        filename = _filename_from_headers(response.headers, record.file_name)
        return response.content, content_type, filename

    # --- bookkeeping ----------------------------------------------------

    def _remember(self, record: FileRecord) -> None:
        with self._lock:
            previous = self._cache.get(record.id)
            if previous is not None:
                if not record.signed_path and previous.signed_path:
                    record.signed_path = previous.signed_path
                if not record.mime and previous.mime:
                    record.mime = previous.mime
            self._cache[record.id] = record
            self._persist()

    def _load(self) -> None:
        if not self.store_path or not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        for item in raw.get("files", []):
            try:
                record = FileRecord(**item)
            except TypeError:
                continue
            self._cache[record.id] = record

    def _persist(self) -> None:
        if not self.store_path:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"files": [asdict(r) for r in self._cache.values()]}
            tmp = self.store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.store_path)
        except OSError:
            pass

    def close(self) -> None:
        self._download.close()


def _record_from(biz: Dict, mime: Optional[str]) -> FileRecord:
    return FileRecord(
        id=biz.get("id") or "",
        file_name=biz.get("file_name") or biz.get("filename") or "",
        status=(biz.get("status") or "UNKNOWN").upper(),
        file_size=int(biz.get("file_size") or 0),
        token_usage=biz.get("token_usage"),
        is_image=bool(biz.get("is_image")),
        signed_path=biz.get("signed_path"),
        model_kind=biz.get("model_kind"),
        error_code=biz.get("error_code"),
        mime=mime,
    )


def _filename_from_headers(headers, fallback: str) -> str:
    disposition = headers.get("content-disposition", "")
    for chunk in disposition.split(";"):
        chunk = chunk.strip()
        if chunk.lower().startswith("filename="):
            return chunk.split("=", 1)[1].strip('"') or fallback
    return fallback


def _encode_multipart(files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    """Encode a multipart body without letting the client's default JSON header win."""
    request = httpx.Request("POST", "https://upload.invalid/", files=files)
    return request.read(), request.headers["content-type"]
