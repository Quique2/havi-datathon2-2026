# Havi — Asistente Virtual de Hey Banco · Datathon 2026

API de inteligencia conversacional personalizada para Hey Banco. Havi es un agente que combina segmentación de clientes, RAG sobre conversaciones reales y un filtro de ruido heurístico para brindar atención financiera adaptada al perfil de cada usuario.

---

## Arquitectura general

```
dataset_50k_anonymized.csv
hey_clientes.csv              → hey_pipeline_v4.py → outputs/perfiles_usuarios.csv
hey_productos.csv                                     outputs/metricas_propension.csv
                                                      Supabase (usuarios, clasificaciones)

                              → hey_databases.py  → outputs/noise_detector.pkl
                                                    outputs/rag_index.pkl

                              → havi_api.py (FastAPI) → expuesto en Railway
```

### Componentes

| Archivo | Responsabilidad |
|---|---|
| `hey_pipeline_v4.py` | Pipeline de ML: segmentación KMeans + propensión GBM por producto + flags de riesgo |
| `hey_databases.py` | Construye el detector de ruido semántico y el índice RAG (embeddings multilingues) |
| `havi_api.py` | **API principal (FastAPI)**. Noise gate heurístico, RAG TF-IDF, sesiones en Supabase, streaming SSE |
| `hey_server.py` | API alternativa (Flask) con embeddings sentence-transformers (sin Supabase) |
| `hey_agent_havi_v2.py` | Agente de referencia (lógica de conversación aislada) |

---

## Pipeline de ML (`hey_pipeline_v4.py`)

Dado el historial de un usuario produce:

1. **Segmento de comportamiento** — KMeans (6 clusters)
2. **Propensión de adopción** — GradientBoosting calibrado por producto
3. **Perfil de comunicación** — tono, longitud de respuesta, uso de emojis, horario de mayor actividad
4. **Flags de riesgo** — `flag_riesgo_churn`, `flag_credito_estresado`, `flag_uso_atipico`
5. **Score de ruido conversacional** — noise gate pasivo por usuario

### Segmentos

| Segmento | Descripción |
|---|---|
| Caótico / Explorador de alto riesgo | Alta variabilidad, mensajes fuera de tema |
| No bancarizado / Básico | Uso mínimo de productos financieros |
| Premium Inversionista Digital | Perfil de alto ingreso, productos de inversión |
| Empresario / PYME Estable | Persona moral o dueño de negocio |
| Sobreendeudado / Uso Intensivo | Crédito estresado, alto riesgo de churn |
| Usuario de Crédito Moderado | Perfil promedio, uso regular de tarjeta |

---

## API principal (`havi_api.py`)

Construida con **FastAPI** y desplegada en **Railway** con workers Uvicorn.

### Variables de entorno requeridas

| Variable | Descripción |
|---|---|
| `ANTHROPIC_API_KEY` | Clave de la API de Anthropic (Claude Sonnet) |
| `SUPABASE_URL` | URL del proyecto en Supabase |
| `SUPABASE_SERVICE_KEY` | Service role key de Supabase |

### Endpoints

#### `GET /health`
Estado del servidor. Railway lo usa como health check.

```json
{
  "status": "ok",
  "usuarios": 1200,
  "segmentos": 6,
  "rag_docs": 48500,
  "rag_tipo": "TF-IDF (sin PyTorch)",
  "api_key": "present",
  "active_sessions": 3
}
```

---

#### `GET /users`
Devuelve 2 usuarios por segmento (para demo).

```json
{
  "users": [
    {
      "user_id": "USR-00042",
      "segmento": "Premium Inversionista Digital",
      "tono": "profesional y conciso",
      "producto_top_1": "Cuenta de inversión CETES"
    }
  ],
  "total": 1200
}
```

---

#### `GET /profile/{user_id}`
Perfil completo del usuario (segmento, flags de riesgo, productos top, datos demográficos).

```bash
GET /profile/USR-00042
```

---

#### `POST /chat/sync`
Respuesta completa sin streaming.

**Request:**
```json
{
  "user_id": "USR-00042",
  "message": "¿Cuánto me dan de rendimiento en CETES?",
  "reset_session": false
}
```

**Response:**
```json
{
  "response": "Los CETES actualmente ofrecen un rendimiento anual ...",
  "noise_gate": false,
  "noise_type": "on_topic",
  "noise_score": 0.0,
  "n_rag_docs": 2,
  "turno": 1
}
```

---

#### `POST /chat`
Respuesta en streaming (Server-Sent Events). Efecto de escritura palabra por palabra.

**Request:** igual que `/chat/sync`

**Eventos SSE:**

| Evento | Payload |
|---|---|
| `meta` | `{"noise_gate": false, "noise_score": 0.0, "n_rag_docs": 2, "turno": 1}` |
| `token` | `{"token": "Los "}` — una palabra a la vez |
| `done` | `{}` |
| `error` | `{"error": "descripción"}` |

**Ejemplo con curl:**
```bash
curl -X POST https://tu-app.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"USR-00001","message":"¿Qué beneficios tiene Hey Pro?"}' \
  --no-buffer
```

---

#### `POST /reset/{user_id}`
Cierra la sesión activa del usuario y reinicia el historial.

```bash
POST /reset/USR-00042
```

---

#### `GET /clusters`
Resumen de los 6 segmentos: número de usuarios, porcentaje, tono, horario y noise promedio.

---

#### `GET /metricas`
AUC de propensión por producto del último `pipeline_run` registrado en Supabase.

---

## Noise Gate

Filtra mensajes fuera de tema antes de llamar a Claude, ahorrando tokens y mejorando la experiencia.

### Tipos de ruido detectados

| Tipo | Score | Ejemplo |
|---|---|---|
| `simbolos` | 0.90 | `!!!???` |
| `broma` | 0.90 | `"te amo"`, `"cuéntame un chiste"` |
| `exploracion` | 0.65 | `"hola"` (sin contexto financiero) |
| `menu` | 0.55 | `"a"`, `"1"` |
| `afirm` | 0.40 | `"ok"`, `"gracias"` |
| `on_topic` | 0.00 | `"¿cuánto debo de mi tarjeta?"` |

- Umbral por mensaje: `≥ 0.55` activa el gate  
- Umbral por usuario: `≥ 0.40` activa alertas en el system prompt  
- Si el mensaje contiene palabras financieras (`tarjeta`, `saldo`, `spei`, etc.), el score se reduce un 55%

---

## RAG (Retrieval-Augmented Generation)

Antes de llamar a Claude se recuperan hasta 3 conversaciones reales del corpus `dataset_50k_anonymized.csv` con mayor similitud semántica al mensaje del usuario.

- **Motor:** TF-IDF con `scikit-learn` (sin PyTorch, sin sentence-transformers)  
- **Índice:** 15,000 features, n-gramas (1,2), TF sublineal  
- **Similitud mínima:** coseno ≥ 0.15 para incluirse  
- Los ejemplos recuperados se inyectan en el system prompt como contexto real de cómo Havi respondió en el pasado

---

## System Prompt dinámico

Cada mensaje construye un system prompt personalizado con:

- Segmento y datos demográficos del usuario
- Instrucciones de longitud, emojis y tono del segmento
- Productos recomendados (top 3)
- Alertas activas (churn, crédito estresado, uso atípico, frustración)
- Ejemplos RAG de conversaciones reales

---

## Base de datos (Supabase)

Tablas principales:

| Tabla | Contenido |
|---|---|
| `usuarios` | Perfil base con `datos_demograficos` (JSONB) |
| `clasificaciones` | Segmento activo, tono, productos top, flags de riesgo |
| `sesiones_chat` | Sesiones activas/cerradas por usuario |
| `mensajes` | Historial de conversaciones |
| `pipeline_runs` | Registro de ejecuciones del pipeline con métricas |

---

## Despliegue en Railway

```bash
# Procfile
web: gunicorn -k uvicorn.workers.UvicornWorker havi_api:app --bind 0.0.0.0:$PORT
```

`railway.json` configura:
- Builder: Nixpacks
- Health check: `GET /health` con timeout de 300 s
- Restart on failure (máx. 3 reintentos)

---

## Instalación local

```bash
pip install -r requirements.txt
```

Crear `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

Ejecutar:
```bash
uvicorn havi_api:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva disponible en `http://localhost:8000/docs`.

---

## Modelo LLM

- **Modelo:** `claude-sonnet-4-5` (Anthropic)  
- **Max tokens por respuesta:** 600  
- **Contexto:** system prompt dinámico + historial de la sesión  
- El historial se persiste en Supabase y se recarga al retomar una sesión

---

## Equipo

Proyecto desarrollado para el **Hey Banco Datathon 2026** por el equipo **Havii**.
