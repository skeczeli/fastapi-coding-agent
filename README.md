# FastAPI Coding Agent

Un coding agent multi-agente de IA **especializado en FastAPI**, construido sobre el
harness del TP en clase — **sin frameworks de orquestación** (sin LangChain / LangGraph /
CrewAI / AutoGen). El harness y la coordinación entre agentes están escritos a mano.

El TP en clase era un solo agente: un harness loop (LLM ↔ tools) con plan mode,
supervisión y guardrails. Este TP final lo evoluciona a un **equipo de agentes
especializados** coordinados por un orquestador principal, grounded en documentación real
de FastAPI vía RAG, con memoria persistente, políticas de seguridad por configuración y
observabilidad completa.

---

## Arquitectura

```
                         tarea del usuario
                            │
                  ┌─────────▼──────────┐
                  │    Orchestrator    │  dueño del TaskState compartido,
                  │  (agente principal)│  coordina subagentes como tools
                  └─────────┬──────────┘
        ┌───────────┬───────┼────────┬───────────┐
   ┌────▼───┐  ┌────▼────┐ ┌▼──────┐ ┌▼─────┐ ┌──▼─────┐
   │Explorer│  │Researcher│ │Implem.│ │Tester│ │Reviewer│
   └────────┘  └─────────┘ └───────┘ └──────┘ └────────┘
     lee        RAG+web     escribe  run cmd   lee+run
        └───────────┴────── TaskState compartido ─────┴──────┘
```

- **Orchestrator** (`agent/agents/orchestrator.py`) — el agente principal. Recibe la
  tarea del usuario, es dueño del `TaskState` compartido, y coordina los subagentes con el
  patrón **agents-as-tools**: cada subagente se expone al LLM del orquestador como un tool
  invocable. El LLM decide el orden real (flujo previsto: explore → research → implement →
  test → review, adaptándose según haga falta). También puede persistir hallazgos en la
  memoria del proyecto vía un tool `remember_project`.

- **Subagentes** — cada uno es su propio harness loop con un prompt específico y un
  **subconjunto de tools/permisos**:

  | Subagente | Responsabilidad | Tools permitidos |
  |-----------|-----------------|------------------|
  | **Explorer** | Entiende el repo: estructura, arquitectura, deps, convenciones, archivos relevantes | `read_file`, `list_files` |
  | **Researcher** | Responde preguntas de FastAPI — **RAG primero**, web search como fallback; etiqueta cada hallazgo por origen | `rag_search`, `web_search` |
  | **Implementer** | Aplica cambios de código a partir de los hallazgos del explorer/researcher | `read_file`, `write_file`, `list_files` |
  | **Tester** | Corre checks (pytest, lint, levantar server y pegarle a un endpoint) y reporta pass/fail | `run_command`, `read_file` |
  | **Reviewer** | Revisa el diff/cambios contra el pedido; aprueba o devuelve | `read_file`, `run_command` |

- **El harness loop es la primitiva compartida** (`agent/harness.py`): un único
  `run_loop(system_prompt, tools, state, …)` reutilizado por el orquestador *y* cada
  subagente. Le pregunta al LLM, valida cada tool call contra la política, lo ejecuta,
  devuelve el resultado, y corta al completar, al detectar un loop sin progreso, o al
  llegar a `max_iters`.

### Estado compartido (`TaskState`)

El único objeto del que es dueño el orquestador y que cada subagente lee/escribe
(`agent/state.py`). Registra, como mínimo:

| Campo | Significado |
|-------|-------------|
| `request` | el pedido original del usuario, textual |
| `plan` / `progress` | pasos previstos / log legible de lo que se hizo |
| `subagent_results` | último resultado de cada subagente, por nombre |
| `sources` | cada fuente consultada, etiquetada con su **origen** (`repo` / `rag` / `web` / `memory` / `inference`) |
| `files_modified` | rutas creadas o editadas |
| `observations` | notas libres (errores, decisiones, callejones sin salida) |

### Memoria persistente del proyecto

Más allá de una sola conversación, el agente mantiene **memoria por proyecto** en disco
(`agent/memory.py`, por defecto `.agent_memory/`, un archivo JSON por categoría):
arquitectura detectada, archivos importantes, dependencias, comandos útiles, convenciones,
decisiones, bugs investigados y resúmenes de sesiones. Se carga al iniciar y se reusa entre
corridas, así una corrida posterior no re-explora lo que ya se sabe.

### Manejo de contexto y cognición

`agent/context.py` mantiene chico el contexto de trabajo: **resume** el historial viejo
cuando se excede un presupuesto de tokens (conservando decisiones clave), y **detecta loops
sin progreso** (la misma llamada devolviendo el mismo resultado, o un ciclo que se repite)
para que el agente pueda cambiar de estrategia, detenerse o pedir ayuda. El prompt del
orquestador además lo hace detenerse y explicarse ante **evidencia insuficiente** (pedido
ambiguo, sin hits en RAG, bloqueo de política).

### Seguridad: políticas por configuración

Cada tool call se valida contra `config/agent.config.yaml` **antes de ejecutarse**
(`agent/policy.py`):

```yaml
workspace: "./workspace"          # writes/comandos confinados acá
read:
  deny: [".env", ".env.*", "**/secrets/**"]
write:
  deny: [".git/**", "**/*.lock"]
commands:
  deny: ["rm -rf*", "sudo*"]
  require_approval: ["git push*", "pip install*", "rm *"]
```

Las rutas de lectura/escritura se matchean contra los globs de `deny`; las escrituras
además se **confinan al workspace** (así no se puede bypassear el guardrail con, p. ej.,
`shutil.rmtree`); los comandos peligrosos se bloquean, y los de `require_approval` piden
confirmación al usuario (en el REPL interactivo).

### Observabilidad

`agent/observability.py` envuelve las llamadas al LLM, los tools y los retrievals en spans
a través de una interfaz de tracer. Con las keys `LANGFUSE_*` seteadas, un tracer de
**Langfuse** registra prompts, modelo, tool calls, documentos recuperados, iteraciones,
errores, latencia, tokens y el resultado final de una corrida completa. Sin keys degrada a
no-op (el tracing nunca rompe el agente).

---

## RAG sobre la documentación de FastAPI

El Researcher consulta un vector store armado con la **documentación oficial** de FastAPI
antes de caer a web search.

- **Corpus fuente:** los docs en markdown de `tiangolo/fastapi/docs` (en inglés),
  ingestados bajo `docs/`.
- **Chunking** (`agent/rag/ingest.py`): cada `.md` se parte **por headings de markdown**,
  registrando el breadcrumb de headings (p. ej. `Tutorial > Request Body`) como metadata.
  Las secciones más largas que `MAX_TOKENS` (≈800 tokens) se sub-parten en ventanas de
  tokens con un pequeño `OVERLAP` (100 tokens), así ningún chunk se pasa del punto óptimo
  del modelo de embeddings.
- **Embeddings:** OpenAI `text-embedding-3-small` (1536-dim) por defecto — el backend de
  embeddings es **seleccionable por env** (ver *Providers* abajo).
- **Vector store:** **Chroma** (`PersistentClient`, distancia coseno), persistido en
  `chroma_db/`. Se usa el mismo `embed_texts` para documentos (ingest) y queries
  (retrieval) para que los vectores sean comparables.
- **Atribución de fuentes:** `retrieve(query, k)` devuelve los top-k chunks como objetos
  `Source` con `origin="rag"`, el breadcrump archivo/sección, el snippet y un score de
  similitud coseno — que se muestran en el reporte de la corrida para ver *qué* docs se
  usaron.

Armar/refrescar el store:

```bash
python -m agent.rag.ingest             # baja docs de FastAPI, chunkea, embebe, guarda
python -m agent.rag.ingest --rebuild   # borra y repuebla
python -m agent.rag.ingest --dry-run   # solo chunking + stats (sin API/store)
```

---

## Caso de uso

**Repo target:** [`astral-sh/uv-fastapi-example`](https://github.com/astral-sh/uv-fastapi-example)
(el tutorial oficial *"Bigger Applications"* de FastAPI — moderno, sin base de datos),
clonado en `./workspace/` como el proyecto sobre el que opera el agente.

Dos objetivos concretos y verificables ejercitan el sistema:

1. **Analizar el repo** → producir un reporte de arquitectura (routers, la auth basada en
   dependencias global/router-level) grounded en los docs RAG de FastAPI, con fuentes
   citadas. *Éxito = reporte correcto citando fuentes repo + RAG, read-only.*
2. **Agregar una funcionalidad** → agregar un endpoint `GET /health` que devuelva
   `{"status": "ok"}` más un test. *Éxito = `pytest` pasa y el endpoint se comporta como se
   especificó.*

Ver `docs/evidence/` para corridas capturadas (output, fuentes recuperadas y una traza de
Langfuse).

---

## Stack

- **Lenguaje:** Python · **LLM/embeddings:** OpenAI SDK (provider-pluggable)
- **Vector store:** Chroma · **Embeddings:** OpenAI `text-embedding-3-small`
- **Web search:** Tavily · **Observabilidad:** Langfuse

---

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[rag,web,obs]"     # core + RAG + web search + observabilidad
#   o más liviano: pip install -e ".[dev]"   (core + pytest)
```

Los grupos opcionales de dependencias mantienen liviana la instalación: `rag` (Chroma +
tiktoken), `web` (Tavily), `obs` (Langfuse), `dev` (pytest).

## Configuración

Copiá `.env.example` a `.env` y completá tus keys:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-...            # LLM + embeddings (provider por defecto)
TAVILY_API_KEY=...               # fallback de web_search (opcional)
LANGFUSE_PUBLIC_KEY=pk-lf-...    # observabilidad (opcional)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Providers (seleccionables por env)

El chat y los embeddings pasan cada uno por `agent/providers.py`, así podés apuntarlos a
cualquier endpoint OpenAI-compatible sin tocar código. Sin variables `AGENT_*`, el default
es OpenAI (`gpt-5-nano` + `text-embedding-3-small`). Para usar otro backend (p. ej. un
endpoint OpenAI-compatible de Gemini/Groq/Cerebras), seteá:

```ini
AGENT_LLM_BASE_URL=...           # endpoint de chat OpenAI-compatible
AGENT_LLM_API_KEY=...
AGENT_LLM_MODEL=...
AGENT_LLM_MAX_TOKENS_PARAM=max_tokens   # OpenAI usa max_completion_tokens
AGENT_EMBED_BASE_URL=...         # (si cambiás el modelo de embeddings → re-ingestar el RAG)
AGENT_EMBED_API_KEY=...
AGENT_EMBED_MODEL=...
```

Editá `config/agent.config.yaml` para apuntar `workspace` al repo sobre el que el agente
debe trabajar y para ajustar las políticas de seguridad.

## Ejecución

```bash
# Orquestador multi-agente (una tarea entra, resultado coordinado sale):
python -m agent.agents.orchestrator "Agregá un endpoint GET /health a ./workspace y un test"

# REPL interactivo de un solo agente (el harness portado del TP en clase):
python -m agent

# Offline, sin API key (respuestas de LLM enlatadas — para probar el loop / los tests):
AGENT_LLM_MOCK=1 python -m agent.agents.orchestrator "..."
```

El orquestador imprime un reporte final: la respuesta, las fuentes consultadas (con sus
labels de origen), los archivos modificados, los resultados por subagente y el progress log.

## Tests

```bash
AGENT_LLM_MOCK=1 pytest -q        # suite completa, offline (sin API key)
```

---

## Entrega / Evidencias

La documentación completa de entrega (caso de uso detallado, arquitectura, base RAG,
demos, observabilidad y reflexión) está en [`docs/INFORME.md`](./docs/INFORME.md).

Las corridas raw y las capturas de observabilidad están en
[`docs/evidence/`](./docs/evidence/).

## Estructura del proyecto

```
agent/
  llm.py            # único lugar que llama al LLM (+ mock mode)
  providers.py      # backend de chat / embeddings seleccionable por env
  harness.py        # run_loop compartido (inner) + converse (outer REPL)
  state.py          # schema del TaskState compartido
  config.py         # carga de configuración
  policy.py         # policy engine (validado antes de cada tool call)
  memory.py         # memoria persistente por proyecto
  context.py        # resumen de historial + detección de loops
  observability.py  # interfaz de tracer + tracer de Langfuse
  modes.py          # plan / supervision modes
  tools/            # interfaz Tool + tools base + rag_search
  agents/           # orquestador + 5 subagentes
  rag/              # ingest + retrieve + store
config/agent.config.yaml
docs/               # corpus de FastAPI ingestado + evidencia de corridas
tests/
```

Ver [`BACKLOG.md`](./BACKLOG.md) para el desglose completo de trabajo y la asignación de
issues.
