"""Diagnostico del DeepSeek Bridge.

Comprueba todo lo que suele romper una primera instalacion:
Python, dependencias, Chromium, el modulo PoW, la sesion y el puerto.

    python doctor.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK = "[ OK ]"
WARN = "[AVISO]"
BAD = "[FALLA]"

problems: list[str] = []
warnings: list[str] = []


def line(level: str, text: str) -> None:
    print(f"  {level} {text}")


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 9):
        line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        line(BAD, f"Python {v.major}.{v.minor} — hace falta 3.9 o superior")
        problems.append("Actualiza Python a 3.9+.")


def check_deps() -> None:
    required = {
        "playwright": "playwright",
        "httpx": "httpx",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "wasmtime": "wasmtime",
        "multipart": "python-multipart",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if not missing:
        line(OK, f"dependencias ({len(required)}/{len(required)})")
    else:
        line(BAD, "faltan: " + ", ".join(missing))
        problems.append("Corre 1-setup.bat de nuevo o `pip install -r requirements.txt`.")


def check_chromium() -> None:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
        Path.home() / ".cache" / "ms-playwright",
    ]
    for base in candidates:
        if base.is_dir() and any(p.name.startswith("chromium") for p in base.iterdir()):
            line(OK, "Chromium de Playwright")
            return
    line(WARN, "Chromium de Playwright no encontrado")
    warnings.append("Si el login falla: `python -m playwright install chromium`.")


def check_wasm() -> None:
    wasm = ROOT / "deepseek" / "sha3_wasm_bg.wasm"
    if wasm.is_file() and wasm.stat().st_size > 1000:
        line(OK, f"modulo PoW ({wasm.stat().st_size // 1024} KB)")
    else:
        line(BAD, "falta deepseek/sha3_wasm_bg.wasm")
        problems.append("Reinstala el paquete: el .wasm es imprescindible.")


def check_session() -> None:
    path = ROOT / "session" / "session.json"
    if not path.is_file():
        line(WARN, "sin sesion guardada (todavia no hiciste login)")
        warnings.append("Ejecuta 2-login.bat antes de 3-run.bat.")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        line(WARN, "session.json existe pero no se puede leer")
        warnings.append("Borra session/ y volve a hacer login.")
        return
    keys = ", ".join(sorted(data)[:4]) if isinstance(data, dict) else "?"
    age_h = (path.stat().st_mtime and (__import__("time").time() - path.stat().st_mtime) / 3600) or 0
    line(OK, f"sesion guardada hace {age_h:.1f} h ({keys})")
    if age_h > 12:
        warnings.append("La sesion tiene mas de 12 h: puede pedirte login de nuevo.")


def check_port(port: int = 8000) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.6)
        in_use = sock.connect_ex(("127.0.0.1", port)) == 0
    if in_use:
        line(WARN, f"puerto {port} ya esta ocupado (puede ser el servidor corriendo)")
    else:
        line(OK, f"puerto {port} libre")


def check_server() -> None:
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/readyz", timeout=4) as r:
            payload = json.loads(r.read().decode())
        line(OK, f"servidor respondiendo ({payload.get('status')}, {payload.get('user')})")
    except Exception:
        line(WARN, "servidor no esta corriendo (normal antes de 3-run.bat)")


def main() -> int:
    print()
    print("  DeepSeek Bridge — diagnostico")
    print("  " + "-" * 46)
    check_python()
    check_deps()
    check_chromium()
    check_wasm()
    check_session()
    check_port()
    check_server()
    print()

    if warnings:
        print("  Avisos:")
        for w in warnings:
            print(f"   - {w}")
        print()

    if problems:
        print("  Problemas que hay que resolver:")
        for p in problems:
            print(f"   - {p}")
        print()
        return 1

    print("  Todo listo.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
