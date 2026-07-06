# Informe de entrega — FastAPI Coding Agent

El README cubre instalación, configuración y ejecución. Este informe cubre el resto de los
entregables: caso de uso, arquitectura, base RAG, demos, observabilidad y reflexión.

---

## Caso de uso

Elegimos como repo target [`astral-sh/uv-fastapi-example`](https://github.com/astral-sh/uv-fastapi-example),
el tutorial oficial *"Bigger Applications"* de FastAPI: un proyecto FastAPI real pero
autocontenido (sus tests corren sin base de datos ni servicios), cuya temática —dependencias
globales y a nivel router— está bien cubierta por la documentación de FastAPI que alimenta el
RAG. Lo clonamos en `./workspace/`, que es además la frontera de seguridad del agente: las
escrituras y comandos quedan confinados ahí.

Definimos dos objetivos concretos y verificables:

| Objetivo | Criterio de éxito |
|----------|-------------------|
| **Analizar el repo** y explicar su auth basada en dependencias (`get_query_token` / `get_token_header`), grounded en los docs de FastAPI | Reporte correcto que cita fuentes **repo + RAG**, diferenciando el origen; read-only |
| **Agregar una funcionalidad**: endpoint `GET /health` → `{"status": "ok"}` + un test | **`pytest` pasa** y el endpoint responde como se especificó |

---

## Arquitectura

El sistema tiene un **orquestador** (agente principal) que recibe la tarea del usuario, es
dueño del estado compartido y coordina cinco subagentes especializados con el patrón
**agents-as-tools**: cada subagente se le expone al LLM del orquestador como un tool invocable,
y el LLM decide a quién delegar y en qué orden (el flujo previsto es explore → research →
implement → test → review).

Cada subagente es su propio harness loop, con un prompt específico y un subconjunto acotado de
tools y permisos:

- **Explorer** — entiende el repo (estructura, archivos, convenciones); solo lectura.
- **Researcher** — responde preguntas de FastAPI consultando primero el RAG y, si no alcanza,
  web search; etiqueta cada hallazgo por origen.
- **Implementer** — escribe los cambios de código a partir de los hallazgos.
- **Tester** — corre los checks (pytest) y reporta pass/fail.
- **Reviewer** — revisa el diff contra el pedido y aprueba o devuelve.

La pieza reusada por todos es una única primitiva, `harness.run_loop(...)`: le pregunta al LLM,
valida cada tool call contra la política, lo ejecuta, devuelve el resultado y corta al
completar, al detectar un loop sin progreso o al llegar a `max_iters`.

El **estado compartido** (`TaskState`) es el objeto que el orquestador posee y todos
leen/escriben: guarda el pedido original, el plan y el progreso, los resultados de cada
subagente, las fuentes consultadas (cada una con su origen: `repo` / `rag` / `web` / `memory` /
`inference`), los archivos modificados y observaciones libres.

Sobre esa base se montan: **memoria persistente por proyecto** (en `.agent_memory/`, que
sobrevive entre corridas), **manejo de contexto** (resumen de historial + detección de loops),
**políticas de seguridad** validadas antes de cada tool call, y **observabilidad** con Langfuse.

---

## Base RAG

Construimos el RAG sobre la **documentación oficial de FastAPI** (`tiangolo/fastapi/docs`, en
inglés). El pipeline de ingest parte cada archivo markdown **por headings**, conservando el
breadcrumb de headings como metadata (p. ej. `Tutorial > Request Body`); las secciones más
largas que ~800 tokens se sub-parten en ventanas con 100 tokens de overlap. El corpus queda en
**2171 chunks** de 154 archivos.

Embebemos con `text-embedding-3-small` de OpenAI (1536-dim) y guardamos en **Chroma**
(`PersistentClient`, distancia coseno) en `chroma_db/`. Usamos la misma función de embedding para
los documentos y para las queries, así los vectores son comparables. En retrieval, cada chunk
vuelve como un `Source` con su origen (`rag`), el breadcrumb archivo > sección, el snippet y un
score de similitud — que el agente muestra en el reporte para dejar a la vista *qué* documentos
usó.

El backend de embeddings (y el del LLM) pasa por una capa de providers seleccionable por
variables de entorno, así que el sistema puede correr sobre cualquier endpoint OpenAI-compatible
sin tocar el código del agente.

---

## Demos

El set de demostración consta de cuatro tareas sobre el caso de uso que, en conjunto, ejercitan las
capacidades que pide la consigna. Se corren en orden (la 2 reusa la memoria que escribe la 1) y
todas quedan registradas en Langfuse:

| # | Demo | Qué demuestra |
|---|------|---------------|
| 1 | **Analizar el repo** | coordinación Explorer + Researcher, **RAG con fuentes**, diferenciación de origen (repo / RAG / inferencia); persiste memoria |
| 2 | **Agregar `GET /health` + test** | **reusa la memoria** de la 1, cadena Implementer → Tester → Reviewer, **resultado verificable** (`pytest` pasa) |
| 3 | **Spec inexistente (stop)** | el agente reconoce **evidencia insuficiente** y se detiene / pide ayuda por su cuenta |
| 4 | **Fuentes web inalcanzables (loop)** | el harness **detecta un loop sin progreso** y el agente cambia de estrategia / pide ayuda |

El runbook ejecutable (setup, prompts finales, verificación y captura de trazas) está en
[`PLAN-DEMOS.md`](./PLAN-DEMOS.md). El output completo de cada corrida queda en
[`evidence/`](./evidence/); acá documentamos qué se observó en cada una.

### Análisis por demo

**Demo 1 — Analizar el repo** ([`evidence/demo1-analyze-traced.txt`](./evidence/demo1-analyze-traced.txt)).
El Explorer mapeó `./workspace/` en una pasada y el Researcher trajo 12 fuentes `[rag]` sobre
dependencies/`APIRouter` (scores 0.54–0.65). El reporte final combina ambos hallazgos —arquitectura,
cómo funcionan las dependencies globales y a nivel router, y las fuentes usadas—, se mantuvo
read-only (`Files modified: (none)`) y persistió memoria (`architecture.json`, `decisions.json`)
para que la reuse la Demo 2.

**Demo 2 — Agregar `GET /health` + test** ([`evidence/demo2-health-memory-traced.txt`](./evidence/demo2-health-memory-traced.txt)).
El progress log nunca invoca al Explorer — el orquestador arrancó directo desde
`[memory] architecture` y `[memory] decisions`, confirmando el reuso de memoria que pide la
consigna. La cadena Implementer → Tester → Reviewer corrió completa y el Reviewer aprobó; lo
verificamos de forma independiente (sin confiar solo en el reporte) corriendo `pytest` nosotros
mismos: `1 passed`. Escrituras confinadas a `workspace/app/main.py` y `workspace/tests/test_health.py`.

**Demo 3 — Spec inexistente (stop)** ([`evidence/demo3-insufficient-evidence-traced.txt`](./evidence/demo3-insufficient-evidence-traced.txt)).
El Explorer se invocó una única vez para confirmar que no hay ningún rastro de la "ACME Internal
Gateway Spec v9" en el repo; combinado con la memoria (que tampoco la menciona) y sin poder usar web
search, el orquestador concluyó explícitamente que no tiene evidencia suficiente para implementar
nada y se detuvo, explicando qué verificó y qué necesitaría (el texto real de la spec) para
continuar — sin modificar ningún archivo (`Files modified: (none)`).

**Demo 4 — Fuentes web inalcanzables (loop)** ([`evidence/demo4-loop-detection-traced.txt`](./evidence/demo4-loop-detection-traced.txt)).
La tarea pide citar fuentes web recientes que la documentación del RAG no cubre. El Researcher
respeta el orden RAG-primero (recupera 6 fuentes `[rag]` sobre benchmarks/performance) y después va
a la web, pero `web_search` no está disponible (sin `TAVILY_API_KEY`) y devuelve el mismo error en
cada intento. Tras repetir la búsqueda sin obtener nada nuevo, la **detección de loops** del harness
se dispara sola —`No progress: web_search returned the same result 4 times`— y en vez de seguir
insistiendo, el agente **cambia de estrategia**: le explica al usuario que no tiene acceso web y le
ofrece alternativas concretas (habilitar la key, aportar las URLs a mano, o aplazar). Cierra las tres
conductas que pide la consigna en una sola corrida —detectar la repetición sin progreso, cambiar de
estrategia y pedir ayuda—, sin modificar ningún archivo (`Files modified: (none)`).

---

## Observabilidad

Integramos **Langfuse**. Con las keys `LANGFUSE_*` configuradas, cada corrida del orquestador
genera una traza con un span raíz `agent.turn`, bajo el cual cuelgan las generaciones del LLM
(modelo, prompt, tokens, latencia), los tool calls y los retrievals del RAG, anidados según la
delegación a subagentes. Una corrida multi-agente completa produce una traza con decenas de
observations (todo el árbol orquestador → subagentes → llm / tools / rag).

Las trazas de las demos y sus capturas de pantalla quedan en [`evidence/`](./evidence/) (cómo
capturarlas: ver `PLAN-DEMOS.md`). Capturas de una corrida completa de la Demo 1, en
[`evidence/langfuse/`](./evidence/langfuse/):

- [`langfuse-full-trace-tree.jpeg`](./evidence/langfuse/langfuse-full-trace-tree.jpeg) — el árbol
  completo: orquestador → `tool:researcher`/`tool:implementer` (cada uno con su propio `agent.turn`
  anidado) → `llm.complete` / `tool:rag_search` / `tool:web_search`.
- [`langfuse-dashboard-overview.jpeg`](./evidence/langfuse/langfuse-dashboard-overview.jpeg) —
  agregados del proyecto: traces totales, costo y tokens por modelo (`gpt-5-nano`).
- [`langfuse-tool-usage-breakdown.jpeg`](./evidence/langfuse/langfuse-tool-usage-breakdown.jpeg) —
  desglose de observations por tipo de tool (`read_file`, `rag_search`, `web_search`, etc.).
- [`langfuse-reviewer-verdict.jpeg`](./evidence/langfuse/langfuse-reviewer-verdict.jpeg) — el
  razonamiento del Reviewer sobre el plan propuesto, no solo el veredicto final.
- [`langfuse-implementer-prompt-metadata.jpeg`](./evidence/langfuse/langfuse-implementer-prompt-metadata.jpeg) —
  system prompt completo del Implementer junto con metadata (`tool_calls`, `latency_ms`).

---

## Reflexión

**Qué funcionó bien.** La arquitectura multi-agente cumplió: la coordinación orquestador →
subagentes vía agents-as-tools produce resultados correctos y verificables de punta a punta. El
RAG recupera las secciones on-target de los docs (scores 0.8–0.9) y el agente diferencia el origen
de cada parte de la información (repo / RAG / memoria / inferencia). La detección de loops se
dispara cuando el agente se repite y lo frena con una sugerencia útil. Y el manejo de errores
resultó robusto: cuando una llamada al LLM falla, degrada a un mensaje claro en lugar de un
traceback.

**Qué costó.** El principal aprendizaje fue la **sensibilidad al prompt**: con instrucciones
vagas, el Explorer tendía a **releer los mismos archivos** y agotar `max_iters` antes de
sintetizar; hizo falta un prompt directivo ("explorá una vez, no releas, después sintetizá") para
que convergiera. También tuvimos que cuidar la **higiene del vector store**: tras varios
`--rebuild`, Chroma acumuló segmentos huérfanos y la colección llegó a reportar 0 documentos; lo
resolvimos borrando el store y re-ingestando de cero. Como detalle de color, la capa de providers
nos dejó correr el sistema sobre distintos backends OpenAI-compatible cuando hizo falta, sin tocar
una línea de código.

**Falta de evidencia / cuándo se detiene.** El orquestador está instruido para detenerse y
explicar qué le falta ante evidencia insuficiente (pedido ambiguo, sin hits en RAG, bloqueo de
política), y la detección de loops cubre el caso de repetición sin progreso.

**Mejoras que aplicamos a partir de los hallazgos.** (1) Se **atribuye la memoria como fuente**
(`origin="memory"`): una respuesta basada en memoria ya no aparece como "sin evidencia".
(2) Se **deduplican las fuentes** en el reporte: el Researcher reformula la query y recuperaba el
mismo chunk muchas veces, así que ahora cada fuente distinta se muestra una vez (con su mejor
score). (3) Se cableó la **aprobación de comandos** en el flujo multi-agente: los comandos
`require_approval` (`pip install`, `git commit`, …) ahora **piden confirmación** en vez de
denegarse, vía un handler a nivel proceso que el CLI instala y que alcanza también a los
subagentes.

**Qué mejoraríamos.** Una **detección de loops más fina**: hoy solo caza repeticiones idénticas, y
releer archivos *distintos* sin aportar información nueva no se detecta; convendría una señal de
progreso semántica. También un **backoff propio ante rate limits** para hacer las corridas
multi-agente más resilientes a la congestión de la API, y poder **acotar las iteraciones de los
subagentes** (no solo las del orquestador), que es donde más se va el tiempo cuando un subagente
da vueltas.
