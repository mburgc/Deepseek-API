"""Agrega el proveedor `deepseek-api` a la configuracion de opencode.

- Hace backup antes de escribir.
- Fusiona: no borra los otros proveedores ni el modelo por defecto.
- Si la config tiene comentarios (JSONC) o esta rota, no la toca: avisa y
  sugiere pegar el bloque de opencode.jsonc a mano.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HOME = Path.home()
CONFIG_DIR = HOME / ".config" / "opencode"

CANDIDATES = [
    CONFIG_DIR / "opencode.json",
    CONFIG_DIR / "opencode.jsonc",
    Path(os.environ.get("APPDATA", "")) / "opencode" / "opencode.json",
]

PROVIDER = {
    "npm": "@ai-sdk/openai-compatible",
    "name": "DeepSeek API (local)",
    "options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local"},
    "models": {
        # El modelo rapido (equivalente a "flash"): el default de DeepSeek.
        # `reasoning` solo aparece cuando escribis `@think`.
        "deepseek": {
            "name": "DeepSeek (rapido)",
            "tool_call": True,
            "reasoning": True,
            "limit": {"context": 800000, "output": 32768},
        },
        # El modelo fuerte, para tareas dificiles.
        "deepseek-expert": {
            "name": "DeepSeek Expert",
            "tool_call": True,
            "reasoning": True,
            "limit": {"context": 800000, "output": 32768},
        },
    },
}


def _is_bridge_provider(cfg: object) -> bool:
    """Proveedor que apunta al bridge local (posible duplicado viejo)."""
    if not isinstance(cfg, dict):
        return False
    options = cfg.get("options") or {}
    url = str(options.get("baseURL") or "").lower()
    return "127.0.0.1:8000" in url or "localhost:8000" in url


def find_config() -> Path:
    for path in CANDIDATES:
        if path.is_file():
            return path
    return CANDIDATES[0]


def has_comments(text: str) -> bool:
    """Detecta // fuera de strings (suficiente para descartar JSONC)."""
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            return True
    return False


def main() -> int:
    path = find_config()
    print(f"  Config de opencode: {path}")

    if path.exists():
        raw = path.read_text(encoding="utf-8")
        if has_comments(raw):
            print("  [AVISO] El archivo tiene comentarios (JSONC).")
            print("          No lo toco para no romperlo.")
            print("          Abri opencode.jsonc y pega el bloque a mano.")
            return 2
        try:
            data = json.loads(raw)
        except Exception as exc:
            print(f"  [AVISO] No pude parsear la config: {exc}")
            print("          Revisala o pega el bloque a mano.")
            return 2
        if not isinstance(data, dict):
            print("  [AVISO] La config no es un objeto JSON. No la toco.")
            return 2
    else:
        data = {}
        print("  No existia: la creo desde cero.")

    providers = data.get("provider")
    if not isinstance(providers, dict):
        providers = {}

    # Proveedores viejos que tambien apuntan al bridge: se eliminan para que no
    # queden modelos duplicados ni ids que ya no existen.
    stale = [name for name in list(providers) if name != "deepseek-api" and _is_bridge_provider(providers[name])]
    for name in stale:
        providers.pop(name)

    already = providers.get("deepseek-api") == PROVIDER
    if already and not stale:
        print("  [OK] Ya estaba todo configurado. Nada que hacer.")
        return 0

    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"  Backup: {backup.name}")

    if stale:
        print("  Eliminados (apuntaban al bridge y ya no se usan): " + ", ".join(stale))

    providers["deepseek-api"] = PROVIDER
    data["provider"] = providers

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("  [OK] Proveedor 'deepseek-api' configurado con 2 modelos.")
    print()
    print("  Modelos disponibles en opencode:")
    print("    deepseek-api/deepseek         (rapido, estilo flash)")
    print("    deepseek-api/deepseek-expert  (el fuerte)")
    print()
    print("  En el chat:")
    print("    @think <mensaje>    activa el razonamiento profundo")
    print("    @search <mensaje>   busca en la web con DeepSeek")
    print("    @edit <n> <texto>   reescribe la enesima respuesta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
