# DeepSeek Bridge

Convierte **tu cuenta gratuita de chat.deepseek.com** en una API compatible con
OpenAI, para usarla desde opencode, Cline, la SDK de OpenAI o `curl`.

No usa la API oficial ni claves de pago: maneja tu sesión de navegador y resuelve
el proof-of-work por vos. Es un proyecto no oficial — usalo con moderación.

---

## Requisitos

- **Windows** (los scripts son `.bat`; en Linux/macOS se usan los comandos de abajo)
- **Python 3.9 o superior** → https://www.python.org/downloads/
  Al instalar, marcá **"Add python.exe to PATH"**.
- Una cuenta gratuita de DeepSeek.

---

## Instalación (una sola vez)

1. Descomprimí el zip en una carpeta tuya, por ejemplo `C:\DeepSeek-Bridge`.
2. Doble clic en **`1-setup.bat`**.
   Crea el entorno virtual, instala dependencias y descarga Chromium (~150 MB).
   Al final corre un diagnóstico.
3. Doble clic en **`2-login.bat`**.
   Se abre un navegador: iniciá sesión con tu cuenta y resolvé el
   *"verify you are human"*. Cerralo cuando veas el chat.
4. Doble clic en **`4-opencode.bat`** (solo si usás opencode).
   Agrega el proveedor a tu configuración con backup previo. No borra nada.

---

## Uso

Doble clic en **`3-run.bat`**. Dejá esa ventana abierta.

```
http://127.0.0.1:8000/v1
```

Comprobar que responde:

```bash
curl http://127.0.0.1:8000/readyz
# {"status":"ready","user":"tu@correo"}
```

### Con opencode

La forma fácil: **`4-opencode.bat`**. Fusiona el proveedor en tu config y deja un
backup. Si tu config tiene comentarios (JSONC), el script no la toca y te avisa:
en ese caso copiá el bloque de **`opencode.jsonc`** a mano.

Después elegí el modelo `deepseek-api/deepseek` (rápido) o
`deepseek-api/deepseek-expert` (el fuerte). Los dos tienen tool calling, adjuntos
y edición.

### Comandos en el chat

Se escriben al principio del mensaje:

| Comando | Qué hace |
|---|---|
| `@think explica esto a fondo` | activa el razonamiento profundo (DeepThink) |
| `@search qué pasó hoy en Chile` | busca en la web con DeepSeek |
| `@think @search ...` | las dos cosas |
| `@edit 1 más breve` | reescribe la última respuesta (o la enésima) |

También valen `!think`, `/think`, `#think`, `@piensa`, `@busca` y `@web`:
opencode usa `@` para mencionar archivos y puede interceptarlo.

El razonamiento y la búsqueda se activan **por turno**. El modelo queda igual;
no hace falta cambiar de modelo ni abrir un chat nuevo.

> La búsqueda web de opencode es una tool del cliente y el bridge no puede
> interceptarla. Con `@search` usás la de DeepSeek, que corre del lado del
> servidor y devuelve las fuentes en `annotations`.

### Con la SDK de OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hola"}],
)
print(r.choices[0].message.content)
```

---

## Qué tiene de particular

**Hilos reales, no reenvíos.** Cada conversación se reanuda en el servidor de
DeepSeek con `parent_message_id`; solo viaja el turno nuevo.

**Editar no abre un chat nuevo.** Si cambiás un mensaje, el bridge rebobina al
punto de divergencia y sigue desde ahí. En la web de DeepSeek eso aparece como
una **versión nueva del mensaje** (las flechas `‹ 1/2 ›`), que es exactamente lo
que hace la interfaz cuando editás.

**Marcador `@edit`.** Como último mensaje:

```text
@edit más breve y con la fuente
@edit 2 reescribí esa respuesta en inglés
```

Rebobina a la enésima respuesta desde el final (1 por defecto) y aplica la
instrucción.

**Herramientas, archivos y visión.** `tools` se emula con un protocolo de texto;
`POST /v1/files` sube adjuntos; los `image_url` cambian el hilo al modelo de
visión.

**DeepThink y búsqueda web.** Se eligen por modelo:
`deepseek-chat-think`, `deepseek-chat-search`, `deepseek-chat-think-search`.

---

## Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `Python no esta en el PATH` | Reinstalá Python marcando "Add python.exe to PATH". |
| Falla `pip install` | Ejecutá `1-setup.bat` de nuevo; revisá que tengas internet. |
| El login no abre el navegador | `venv\Scripts\python -m playwright install chromium` y reintentá. |
| `login_required` al usar la API | La sesión expiró: corré `2-login.bat`. |
| `429` | Límite de DeepSeek. Esperá unos segundos; no paralelices. |
| Puerto ocupado | Cerrá otra instancia o cambiá `PORT` (ver `.env.example`). |
| Respuestas raras con tools | Los modelos gratuitos a veces ignoran el formato; reintentá. |

Diagnóstico completo en cualquier momento:

```bash
venv\Scripts\python doctor.py
```

---

## Límites y advertencias

- **Una cuenta, una generación a la vez.** Las llamadas se serializan; no es un
  servidor para varias personas.
- **Tu sesión es tuya.** La carpeta `session/` guarda cookies y un perfil de
  Chrome. **No la compartas ni la subas a ningún lado.** Si la borrás, hay que
  volver a hacer login.
- **Es emulación de la web**, no la API oficial: los límites de DeepSeek aplican
  y el formato puede cambiar si ellos cambian el sitio.
- Uso personal y moderado, respetando los términos de DeepSeek.

---

## Estructura

```
app.py              servidor
deepseek/           cliente (auth, PoW, streaming)
server/             capa OpenAI: formato, hilos, tools, archivos
examples/           ejemplos ejecutables
tests/              68 tests (pytest)
doctor.py           diagnóstico
1-setup.bat         2-login.bat         3-run.bat
```

Licencia MIT — ver `LICENSE`.
