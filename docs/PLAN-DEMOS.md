# Plan de demos — runbook para correr de cero

Cómo ejecutar el set de 3 demos limpio, ya con los fixes aplicados y una key de provider con
cupo. Las corridas viejas en `evidence/` son scratch y se reemplazan con lo que salga de acá.

Todo asume: estar en la raíz del repo y el venv activado.

```bash
cd /Users/skeczeli/faculty/ai/fastapi-coding-agent && source .venv/bin/activate
```

---

## 1. Setup del provider

### Opción A — OpenAI recargado (recomendada)

Lo más limpio una vez que el profe recargue. Sin caps diarios, sin server local de embeddings.

1. En `.env`: **borrar** las líneas `AGENT_LLM_*` y `AGENT_EMBED_*`, y poner la `OPENAI_API_KEY`
   con la key nueva. Sin esas variables, el sistema usa OpenAI por defecto.
2. **Re-ingestar** (el store actual está con embeddings locales de 384-dim; OpenAI usa 1536-dim):
   ```bash
   python -m agent.rag.ingest --docs-dir docs/fastapi/docs/en/docs --rebuild
   ```

### Opción B — chat en provider free + embeddings locales (fallback)

Si no hay OpenAI y usás una key fresca de Cerebras/Groq/etc.

1. En `.env`, `AGENT_LLM_*` con la key fresca (ej. Cerebras):
   ```ini
   AGENT_LLM_BASE_URL=https://api.cerebras.ai/v1
   AGENT_LLM_API_KEY=<key nueva>
   AGENT_LLM_MODEL=gpt-oss-120b
   AGENT_LLM_MAX_TOKENS_PARAM=max_tokens
   AGENT_EMBED_BASE_URL=http://127.0.0.1:8900/v1
   AGENT_EMBED_API_KEY=local
   AGENT_EMBED_MODEL=bge-small
   ```
2. Levantar el **server local de embeddings** y dejarlo corriendo en otra terminal
   (script al final de este doc). El store ya está ingestado con `bge-small`, **no** hay que
   re-ingestar.

### Verificar el RAG (cualquiera de las dos opciones)

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from agent.rag import store, retrieve
print('count:', store.get_collection().count())              # ~2171
print('retrieval:', len(retrieve.retrieve('global dependencies Depends', k=3)), 'chunks')  # >0
"
```

### Observabilidad

Asegurarse de que estén las `LANGFUSE_*` en `.env` (con keys válidas) para que las corridas se
tracen. Verificar:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from agent import observability
print('tracer:', type(observability.init_tracer()).__name__)   # LangfuseTracer
"
```

---

## 2. Reset de estado (antes del set)

Workspace pristino y memoria vacía, para que la narrativa sea limpia (la demo 1 construye la
memoria de cero):

```bash
rm -rf workspace .agent_memory
git clone -q --depth 1 https://github.com/astral-sh/uv-fastapi-example workspace && rm -rf workspace/.git
```

> Si el Tester va a correr `pytest`, asegurate de tener `fastapi` y `httpx` instalados en el venv
> (`pip install fastapi httpx`).

---

## 3. Las 3 demos (correr en orden)

Cada una guarda el output crudo en `docs/evidence/`. Corren con el CLI del orquestador, que ya
inicializa Langfuse solo.

### Demo 1 — Analizar el repo (RAG + fuentes, persiste memoria)

```bash
python -m agent.agents.orchestrator "Analyze ONLY the FastAPI project inside ./workspace/. Read files ONLY under ./workspace/. Read-only: do not modify files. Explore in ONE pass — do NOT re-read files. Use the Researcher to consult the FastAPI RAG docs about how this app's dependencies work, grounded in the retrieved sources. Deliver: (1) architecture and key files; (2) how global and router-level dependencies work in this app; (3) the RAG sources used. Finally, persist the durable findings you consider relevant to project memory using remember_project." --max-iters 12 > docs/evidence/demo1-analyze-traced.txt 2>&1
```

**Verificar:**
- `grep -c '\[rag\]' docs/evidence/demo1-analyze-traced.txt` → > 0 (usó RAG y mostró fuentes).
- `grep '\[repo\]' docs/evidence/demo1-analyze-traced.txt | grep -v workspace` → vacío (scope ok).
- `ls .agent_memory/` → escribió memoria (architecture, important_files, etc.).
- Reporte limpio: `grep 'orchestrator done' docs/evidence/demo1-analyze-traced.txt`.

### Demo 2 — Agregar `GET /health` + test (reusa memoria)

```bash
python -m agent.agents.orchestrator "Add GET /health returning {\"status\": \"ok\"} to the FastAPI app in ./workspace/. Use project memory from the previous analysis before exploring; avoid re-reading files unless memory is insufficient. Create a focused test with TestClient and run: cd ./workspace && python -m pytest tests/test_health.py -q. The Reviewer must confirm the endpoint, the test result, and that writes stayed under ./workspace/." --max-iters 14 > docs/evidence/demo2-health-memory-traced.txt 2>&1
```

**Verificar (de forma independiente, sin confiar en el reporte):**
```bash
grep -n health workspace/app/main.py            # endpoint escrito
cd workspace && python -m pytest tests/ -q ; cd ..   # debe dar "1 passed"
```
- En el progress log, idealmente arranca por la memoria (no re-explora todo).

### Demo 3 — Spec inexistente / stop (sin decirle que pare)

```bash
python -m agent.agents.orchestrator "Implementá en la app FastAPI de ./workspace/ un endpoint que cumpla con la 'ACME Internal Gateway Spec v9'. No uses web search — basate en la documentación del RAG, la memoria del proyecto y el repo. Asegurate de que tu implementación realmente siga la spec." --max-iters 10 > docs/evidence/demo3-insufficient-evidence-traced.txt 2>&1
```

**Verificar:**
- El agente **no inventa** el endpoint: se detiene y explica que falta la spec / pide el documento,
  o corta por loop detection. (Si inventa, es un hallazgo honesto para la reflexión — pero la idea
  es que pare solo.)
- `grep -E "harness\] stopped|ask|missing|insufficient|no encontr" docs/evidence/demo3-*.txt`.

---

## 4. Capturar las trazas de Langfuse (entregable #7)

1. Listar las trazas recientes para tener los IDs:
   ```bash
   python -c "
   from dotenv import load_dotenv; load_dotenv()
   import os, base64, json, urllib.request
   pub=os.getenv('LANGFUSE_PUBLIC_KEY'); sec=os.getenv('LANGFUSE_SECRET_KEY'); host=os.getenv('LANGFUSE_HOST')
   auth=base64.b64encode(f'{pub}:{sec}'.encode()).decode()
   req=urllib.request.Request(f'{host}/api/public/traces?limit=5', headers={'Authorization':'Basic '+auth})
   for t in json.load(urllib.request.urlopen(req)).get('data', []):
       print(t['timestamp'][:19], '| obs', len(t.get('observations') or []), '|', t['id'])
   "
   ```
2. Entrar a la UI (`LANGFUSE_HOST`) → *Tracing* → abrir la traza de la **demo 1** (la que tiene más
   observations, con el árbol completo orquestador → subagentes → llm/tools/rag).
3. Sacar screenshot de la traza expandida y guardarlo en `docs/evidence/` (p. ej.
   `langfuse-demo1.png`).

---

## Apéndice — server local de embeddings (solo Opción B)

Si usás embeddings locales, levantá este server (necesita `pip install fastembed`) y dejalo
corriendo. El `.env` (Opción B) apunta `AGENT_EMBED_BASE_URL` a `http://127.0.0.1:8900/v1`.

```python
# embed_server.py — servidor OpenAI-compatible de embeddings con fastembed
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"     # 384-dim
_model = TextEmbedding(MODEL)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        inp = payload.get("input", [])
        if isinstance(inp, str): inp = [inp]
        vecs = [v.tolist() for v in _model.embed(inp)]
        body = json.dumps({"object": "list", "model": payload.get("model", MODEL),
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8900
    print(f"listening on http://127.0.0.1:{port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
```

```bash
pip install fastembed
python embed_server.py 8900     # dejar corriendo en otra terminal
```
