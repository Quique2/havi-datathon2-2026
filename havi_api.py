"""
=============================================================
HEY BANCO — DATATHON 2026
havi_api.py — API FastAPI para el Agente Havi v2

Sin sentence-transformers ni PyTorch.
Noise gate: heurístico | RAG: TF-IDF sklearn (~400MB build)

Endpoints:
  GET  /health              health check
  GET  /users               2 usuarios por segmento (demo)
  GET  /profile/{user_id}   perfil completo del usuario
  POST /chat/sync           respuesta completa
  POST /chat                respuesta streamed (SSE, word by word)
  POST /reset/{user_id}     reiniciar sesión

Variables de entorno:
  ANTHROPIC_API_KEY   — requerida en Railway → Variables
=============================================================
"""

import os, json, math, re, logging, asyncio, urllib.request, urllib.error
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def _out(fn): return os.path.join(OUTPUTS_DIR, fn)
def _data(fn): return os.path.join(BASE_DIR, fn)


# ─────────────────────────────────────────────
# ESTADO GLOBAL
# ─────────────────────────────────────────────
_supabase:    Optional[Client]        = None
_rag_corpus:  Optional[list]         = None
_rag_tfidf:   Optional[TfidfVectorizer] = None
_rag_matrix                           = None
_sessions:    dict                    = {}


# ═══════════════════════════════════════════════════════════
# LIFESPAN — carga todo una sola vez al arrancar
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _supabase, _rag_corpus, _rag_tfidf, _rag_matrix

    log.info("=" * 55)
    log.info("  HAVI API — Iniciando (sin PyTorch)")
    log.info("=" * 55)

    # 1. API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("⚠ ANTHROPIC_API_KEY no configurada — las respuestas de Claude fallarán")

    # 2. Supabase
    _supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    count_res  = _supabase.table("usuarios").select("id", count="exact").execute()
    n_usuarios = count_res.count or 0
    seg_res    = (_supabase.table("clasificaciones")
                  .select("segmento_nombre").eq("activo", True).execute())
    n_segs     = len({r["segmento_nombre"] for r in seg_res.data})
    log.info(f"  Supabase: {n_usuarios:,} usuarios | {n_segs} segmentos")

    # 3. RAG con TF-IDF
    convs_path = _data("dataset_50k_anonymized.csv")
    if os.path.exists(convs_path):
        log.info("  Construyendo índice RAG con TF-IDF...")
        convs = pd.read_csv(convs_path).dropna(subset=["input", "output"])
        convs = convs[convs["output"].str.len() > 30]
        corpus = (convs.sort_values("date", na_position="last")
                       .groupby("conv_id").first().reset_index())
        _rag_corpus = corpus[["input", "output"]].to_dict(orient="records")
        _rag_tfidf  = TfidfVectorizer(max_features=15000,
                                      ngram_range=(1, 2),
                                      sublinear_tf=True)
        _rag_matrix = _rag_tfidf.fit_transform([r["input"] for r in _rag_corpus])
        log.info(f"  RAG TF-IDF: {len(_rag_corpus):,} documentos indexados")
    else:
        log.warning("  dataset_50k_anonymized.csv no encontrado — RAG desactivado")

    log.info("  [OK] Servidor listo")
    log.info("=" * 55)
    yield
    log.info("  Servidor detenido")


# ═══════════════════════════════════════════════════════════
# NOISE GATE HEURÍSTICO
# ═══════════════════════════════════════════════════════════

_AFIRM_NEG = frozenset([
    "si","sí","no","ok","vale","gracias","perfecto","listo",
    "entendido","claro","de acuerdo","adelante","continuar","np",
])
_MENU_NAV  = frozenset(["a","b","c","d","1","2","3","4"])
_BROMAS    = ["chiste","te amo","quien eres","quién eres","eres humano",
              "eres ia","gaming","cántame","cantame","me ayudas a ligar",
              "cuéntame un chiste","cuentame un chiste"]
_FIN_WORDS = ["tarjeta","cuenta","saldo","transferencia","credito","crédito",
              "pago","banco","inversión","inversion","spei","cajero","cashback",
              "havi","hey","oxxo","dinero","cobro","token","clabe","préstamo",
              "prestamo","límite","limite","cargo","movimiento","gat","msi"]

NOISE_UMBRAL_MSG  = 0.55
NOISE_UMBRAL_USER = 0.40

_RESP_RUIDO = {
    "broma":      "¡Jaja! 😄 Eso está fuera de mis conocimientos financieros. ¿En qué te puedo ayudar con tu cuenta?",
    "simbolos":   "¿Me puedes escribir en texto lo que necesitas? ¡Estoy aquí!",
    "exploracion":"¿Hay algo en lo que te pueda ayudar? Puedo orientarte con tu cuenta, tarjetas, transferencias, créditos o inversiones.",
}


def _evaluar_noise(texto: str):
    """
    Evalúa si el mensaje es ruido.
    Returns: (gate_activo, noise_type, noise_score, respuesta_sugerida)
    """
    if not texto or not texto.strip():
        return True, "vacio", 1.0, _RESP_RUIDO["exploracion"]

    t   = str(texto).lower().strip()
    tok = re.findall(r"\w+", t)
    n   = len(tok)

    is_sym   = len(t) > 0 and not re.search(r"[a-záéíóúñ0-9]", t, re.I)
    is_broma = any(b in t for b in _BROMAS)
    is_menu  = t in _MENU_NAV
    is_afirm = t in _AFIRM_NEG
    has_fin  = any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in _FIN_WORDS)
    is_expl  = n <= 2 and not has_fin and not is_afirm and not is_menu

    if   is_sym:   tipo, score = "simbolos",    0.90
    elif is_broma: tipo, score = "broma",        0.90
    elif is_menu:  tipo, score = "menu",          0.55
    elif is_afirm: tipo, score = "afirm",         0.40
    elif is_expl:  tipo, score = "exploracion",   0.65
    else:          tipo, score = "on_topic",       0.0

    if has_fin and score > 0:
        score *= 0.45
    score = round(min(1.0, score), 4)

    gate = score >= NOISE_UMBRAL_MSG
    resp = _RESP_RUIDO.get(tipo) if gate else None
    return gate, tipo, score, resp


# ═══════════════════════════════════════════════════════════
# RAG TF-IDF
# ═══════════════════════════════════════════════════════════

def _recuperar_rag(texto: str, top_k: int = 3, min_sim: float = 0.15) -> list:
    if _rag_tfidf is None or _rag_matrix is None:
        return []
    vec  = _rag_tfidf.transform([texto])
    sims = cosine_similarity(vec, _rag_matrix).flatten()
    top  = np.argsort(sims)[::-1][:top_k * 3]
    res, vistos = [], set()
    for i in top:
        sim = float(sims[i])
        if sim < min_sim:
            break
        r   = _rag_corpus[i]
        key = r["output"][:60]
        if key in vistos:
            continue
        vistos.add(key)
        res.append({
            "pregunta_similar": r["input"],
            "respuesta_havi":   r["output"],
            "similitud":        round(sim, 3),
        })
        if len(res) >= top_k:
            break
    return res


# ═══════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════

def _construir_system_prompt(perfil: dict, rag_docs: list = None) -> str:
    segmento = perfil.get("segmento_nombre", "Usuario de Crédito Moderado")
    tono     = perfil.get("tono", "amigable y concreto")
    longitud = str(perfil.get("longitud", "media"))
    emojis   = str(perfil.get("emojis", "ocasional"))
    horario  = perfil.get("horario", "")
    estilo   = perfil.get("estilo_comunicacion", "claro y directo")

    prods = [p for p in [perfil.get("producto_top_1", ""),
                          perfil.get("producto_top_2", ""),
                          perfil.get("producto_top_3", "")]
             if p and str(p) not in ("nan", "", "None")]

    def _safe_bool(v): return bool(int(float(v or 0)))
    def _safe_float(v):
        try: return float(v or 0)
        except: return 0.0
    def _safe_int(v):
        try: return int(float(v or 0))
        except: return 0

    flag_churn = _safe_bool(perfil.get("flag_riesgo_churn", 0))
    flag_est   = _safe_bool(perfil.get("flag_credito_estresado", 0))
    flag_at    = _safe_bool(perfil.get("flag_uso_atipico", 0))
    noise_pct  = _safe_float(perfil.get("noise_score_usuario", 0))
    frust      = _safe_int(perfil.get("conv_frustracion", 0))

    lk = longitud.split("(")[0].strip().split(" ")[0].lower()
    ek = emojis.split("(")[0].strip().split(" ")[0].lower()

    long_str = {
        "corta": "Máximo 2 oraciones. Ve directo al punto.",
        "media": "Máximo 4 oraciones. Completo pero sin rodeos.",
        "larga": "Puedes desarrollar con detalle.",
    }.get(lk, "Máximo 4 oraciones.")

    emoj_str = {
        "nunca":     "NO uses emojis bajo ninguna circunstancia.",
        "ocasional": "Puedes usar máximo un emoji si da calidez.",
        "frecuente": "Puedes usar emojis para ser más amigable.",
    }.get(ek, "Máximo un emoji si da calidez.")

    alertas = []
    if noise_pct >= NOISE_UMBRAL_USER:
        alertas.append("⚠ NOISE GATE: NO ofrezcas productos. Solo responde educativamente.")
    if flag_est:
        alertas.append("⚠ CRÉDITO ESTRESADO: PROHIBIDO ofrecer más crédito.")
    if flag_churn:
        alertas.append("⚠ RIESGO ABANDONO: Resuelve completamente antes de cualquier oferta.")
    if flag_at:
        alertas.append("⚠ PATRÓN ATÍPICO: Si menciona cargos desconocidos, ofrece aclaración inmediata.")
    if frust >= 2:
        alertas.append("⚠ FRUSTRACIÓN HISTÓRICA: Resuelve primero, vende después.")
    if not alertas:
        alertas.append("✓ Sin alertas. Puedes responder con normalidad.")

    rag_sec = ""
    if rag_docs:
        rag_sec = "\n\n## EJEMPLOS REALES DE CONVERSACIONES HAVI\n"
        for i, r in enumerate(rag_docs):
            rag_sec += (f"Ejemplo {i+1}:\n"
                        f"Usuario: {r['pregunta_similar']}\n"
                        f"Havi: {r['respuesta_havi']}\n\n")

    prods_txt = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(prods)) \
                or "  (sin recomendaciones activas)"

    try:
        ingreso_fmt = f"${float(perfil.get('ingreso_mensual_mxn', 0) or 0):,.0f}"
    except Exception:
        ingreso_fmt = str(perfil.get("ingreso_mensual_mxn", ""))

    return f"""Eres Havi, el asistente virtual de Hey Banco. Brinda atención personalizada.

## PERFIL DEL CLIENTE ({perfil.get('user_id', '')})
- Segmento: {segmento}
- {perfil.get('ocupacion', '')} | Ingreso: {ingreso_fmt} MXN/mes | Score buró: {perfil.get('score_buro', '')}
- Hey Pro: {'Sí' if perfil.get('es_hey_pro') else 'No'} | NPS: {perfil.get('satisfaccion_1_10', '')}/10

## CÓMO COMUNICARTE CON ESTE CLIENTE ({segmento})
- Tono: {tono}
- Longitud: {long_str}
- Emojis: {emoj_str}
- Horario de mayor actividad: {horario}
- Estilo: {estilo}

## PRODUCTOS RECOMENDADOS
{prods_txt}
Menciona estos solo si emergen naturalmente del contexto.

## ALERTAS ACTIVAS
{chr(10).join(alertas)}
{rag_sec}
## REGLAS INVARIABLES
1. Responde siempre en español mexicano natural.
2. Nunca inventes saldos, fechas ni montos que no conozcas.
3. Aplica SIEMPRE las instrucciones de longitud y emojis del segmento.
4. Si el cliente pide un humano: 800-123-4567, disponible 24/7.
5. Nunca solicites ni repitas datos sensibles del cliente.
"""


# ═══════════════════════════════════════════════════════════
# CLAUDE API
# ═══════════════════════════════════════════════════════════

def _llamar_claude(system_prompt: str, historial: list, mensaje: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[Error: ANTHROPIC_API_KEY no configurada en Railway → Variables]"

    messages = historial + [{"role": "user", "content": mensaje}]
    payload  = json.dumps({
        "model":      "claude-sonnet-4-5",
        "max_tokens": 600,
        "system":     system_prompt,
        "messages":   messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return "[Sin respuesta del modelo]"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[Error HTTP {e.code}: {body[:300]}]"
    except Exception as e:
        return f"[Error API: {e}]"


# ═══════════════════════════════════════════════════════════
# SESIÓN DE CONVERSACIÓN
# ═══════════════════════════════════════════════════════════

class HaviSession:
    def __init__(self, user_id: str, perfil: dict):
        self.user_id   = user_id
        self.perfil    = perfil
        self.sesion_id = self._init_sesion()
        self.historial = self._cargar_historial()
        self.turno     = len(self.historial) // 2

    def _init_sesion(self) -> str:
        existing = (_supabase.table("sesiones_chat")
                    .select("id").eq("user_id", self.user_id).eq("activa", True)
                    .limit(1).execute())
        if existing.data:
            return existing.data[0]["id"]
        row = (_supabase.table("sesiones_chat")
               .insert({"user_id": self.user_id, "activa": True}).execute())
        return row.data[0]["id"]

    def _cargar_historial(self) -> list:
        msgs = (_supabase.table("mensajes")
                .select("rol,contenido")
                .eq("sesion_id", self.sesion_id)
                .order("created_at").execute())
        return [{"role": m["rol"], "content": m["contenido"]} for m in msgs.data]

    def _guardar_mensajes(self, user_msg: str, asst_msg: str):
        _supabase.table("mensajes").insert([
            {"sesion_id": self.sesion_id, "user_id": self.user_id,
             "rol": "user",      "tipo": "texto", "contenido": user_msg},
            {"sesion_id": self.sesion_id, "user_id": self.user_id,
             "rol": "assistant", "tipo": "texto", "contenido": asst_msg},
        ]).execute()

    def chat(self, mensaje: str) -> tuple[str, dict]:
        self.turno += 1

        # Noise gate
        gate, tipo, score, resp_pred = _evaluar_noise(mensaje)
        if gate and resp_pred:
            self._guardar_mensajes(mensaje, resp_pred)
            self.historial.append({"role": "user",      "content": mensaje})
            self.historial.append({"role": "assistant", "content": resp_pred})
            return resp_pred, {
                "noise_gate":  True,
                "noise_tipo":  tipo,
                "noise_score": score,
                "n_rag_docs":  0,
                "turno":       self.turno,
            }

        # RAG + Claude
        rag_docs      = _recuperar_rag(mensaje)
        system_prompt = _construir_system_prompt(self.perfil, rag_docs)
        respuesta     = _llamar_claude(system_prompt, self.historial, mensaje)

        self._guardar_mensajes(mensaje, respuesta)
        self.historial.append({"role": "user",      "content": mensaje})
        self.historial.append({"role": "assistant", "content": respuesta})

        return respuesta, {
            "noise_gate":  False,
            "noise_tipo":  tipo,
            "noise_score": score,
            "n_rag_docs":  len(rag_docs),
            "rag_sims":    [r["similitud"] for r in rag_docs],
            "turno":       self.turno,
        }


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _get_perfil(user_id: str) -> dict:
    u = _supabase.table("usuarios").select("*").eq("user_id", user_id).execute()
    if not u.data:
        raise HTTPException(status_code=404,
                            detail=f"Usuario '{user_id}' no encontrado")
    usuario = u.data[0]

    c = (_supabase.table("clasificaciones")
         .select("*").eq("user_id", user_id).eq("activo", True)
         .order("created_at", desc=True).limit(1).execute())

    demo = usuario.get("datos_demograficos") or {}
    perfil: dict = {"user_id": user_id}
    perfil.update(demo)
    if c.data:
        perfil.update(c.data[0])
    return perfil

def _get_or_create_session(user_id: str, reset: bool = False) -> HaviSession:
    if reset:
        if user_id in _sessions:
            del _sessions[user_id]
        _supabase.table("sesiones_chat").update(
            {"activa": False, "cerrada_at": datetime.now(timezone.utc).isoformat()}
        ).eq("user_id", user_id).eq("activa", True).execute()
    if user_id not in _sessions:
        perfil = _get_perfil(user_id)
        _sessions[user_id] = HaviSession(user_id, perfil)
    return _sessions[user_id]

def _clean_perfil(perfil: dict) -> dict:
    clean = {}
    for k, v in perfil.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        clean[k] = v
    return clean


# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Havi Agent API",
    version="2.1",
    description="Motor de inteligencia personalizada — Hey Banco Datathon 2026",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id:       str
    message:       str
    reset_session: Optional[bool] = False

class ChatResponse(BaseModel):
    response:    str
    noise_gate:  bool
    noise_type:  str
    noise_score: float
    n_rag_docs:  int
    turno:       int


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — Railway lo usa para saber si el servidor arrancó."""
    key_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    n_usuarios, n_segs = 0, 0
    if _supabase:
        try:
            cr = _supabase.table("usuarios").select("id", count="exact").execute()
            n_usuarios = cr.count or 0
            sr = (_supabase.table("clasificaciones")
                  .select("segmento_nombre").eq("activo", True).execute())
            n_segs = len({r["segmento_nombre"] for r in sr.data})
        except Exception:
            pass
    return {
        "status":          "ok",
        "usuarios":        n_usuarios,
        "segmentos":       n_segs,
        "rag_docs":        len(_rag_corpus) if _rag_corpus else 0,
        "rag_tipo":        "TF-IDF (sin PyTorch)",
        "api_key":         "present" if key_ok else "missing",
        "active_sessions": len(_sessions),
    }


@app.get("/users")
def list_users():
    """Devuelve 2 usuarios por segmento en el orden canónico (para demo)."""
    SEGMENT_ORDER = [
        "Caótico / Explorador de alto riesgo",
        "No bancarizado / Básico",
        "Premium Inversionista Digital",
        "Empresario / PYME Estable",
        "Sobreendeudado / Uso Intensivo",
        "Usuario de Crédito Moderado",
    ]
    rows = (_supabase.table("clasificaciones")
            .select("user_id,segmento_nombre,tono,producto_top_1")
            .eq("activo", True).execute()).data

    rank = {s: i for i, s in enumerate(SEGMENT_ORDER)}
    rows.sort(key=lambda r: (rank.get(r.get("segmento_nombre", ""), 99),
                              r.get("user_id", "")))

    seen: dict[str, int] = {}
    users = []
    for r in rows:
        seg = r.get("segmento_nombre", "")
        if seen.get(seg, 0) >= 2:
            continue
        seen[seg] = seen.get(seg, 0) + 1
        users.append({
            "user_id":        r.get("user_id", ""),
            "segmento":       seg,
            "tono":           r.get("tono", ""),
            "producto_top_1": r.get("producto_top_1", ""),
        })

    total = (_supabase.table("usuarios").select("id", count="exact").execute()).count or 0
    return {"users": users, "total": total}


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    """Perfil completo del usuario."""
    return {"profile": _clean_perfil(_get_perfil(user_id))}


@app.post("/chat/sync", response_model=ChatResponse)
def chat_sync(req: ChatRequest):
    """Respuesta completa — sin streaming."""
    session  = _get_or_create_session(req.user_id, req.reset_session)
    resp, info = session.chat(req.message)
    return ChatResponse(
        response    = resp,
        noise_gate  = info["noise_gate"],
        noise_type  = info["noise_tipo"],
        noise_score = info["noise_score"],
        n_rag_docs  = info["n_rag_docs"],
        turno       = info["turno"],
    )


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """
    Respuesta streamed (SSE) — efecto de escritura palabra por palabra.

    Eventos:
      event: meta    data: {"noise_gate":..., "noise_score":..., "n_rag_docs":...}
      event: token   data: {"token": "palabra "}
      event: done    data: {}
      event: error   data: {"error": "..."}
    """
    session    = _get_or_create_session(req.user_id, req.reset_session)
    resp, info = session.chat(req.message)

    async def generate():
        # 1. Metadata
        meta = {
            "noise_gate":  info["noise_gate"],
            "noise_type":  info["noise_tipo"],
            "noise_score": info["noise_score"],
            "n_rag_docs":  info["n_rag_docs"],
            "turno":       info["turno"],
        }
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 2. Tokens palabra por palabra
        words = resp.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.03)

        # 3. Done
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering":"no",
        },
    )


@app.post("/reset/{user_id}")
def reset_session(user_id: str):
    """Reinicia la sesión de un usuario."""
    if user_id in _sessions:
        del _sessions[user_id]
    _supabase.table("sesiones_chat").update(
        {"activa": False, "cerrada_at": datetime.now(timezone.utc).isoformat()}
    ).eq("user_id", user_id).eq("activa", True).execute()
    return {"status": "ok", "user_id": user_id}


@app.get("/clusters")
def get_clusters():
    """Resumen de los 6 segmentos."""
    rows = (_supabase.table("clasificaciones")
            .select("segmento_nombre,tono,horario,noise_score_usuario")
            .eq("activo", True).execute()).data

    groups: dict[str, list] = {}
    for r in rows:
        seg = r.get("segmento_nombre", "Sin segmento")
        groups.setdefault(seg, []).append(r)

    total = len(rows) or 1
    res = []
    for seg, grp in groups.items():
        noise_vals = [r["noise_score_usuario"] for r in grp
                      if r.get("noise_score_usuario") is not None]
        res.append({
            "segmento":        seg,
            "n_usuarios":      len(grp),
            "pct":             round(len(grp) / total, 3),
            "tono":            grp[0].get("tono", ""),
            "horario":         grp[0].get("horario", ""),
            "noise_pct_medio": round(sum(noise_vals) / len(noise_vals), 3) if noise_vals else 0.0,
        })
    return sorted(res, key=lambda x: x["n_usuarios"], reverse=True)


@app.get("/metricas")
def get_metricas():
    """Métricas AUC de propensión por producto (último pipeline_run)."""
    run = (_supabase.table("pipeline_runs")
           .select("metricas").order("ejecutado_at", desc=True).limit(1).execute())
    if not run.data or not run.data[0].get("metricas"):
        raise HTTPException(status_code=404, detail="Métricas no disponibles")
    metricas = run.data[0]["metricas"]
    return metricas if isinstance(metricas, list) else [metricas]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("havi_api:app", host="0.0.0.0", port=port, reload=False)