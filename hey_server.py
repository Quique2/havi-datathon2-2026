"""
=============================================================
HEY BANCO — DATATHON 2026
hey_server.py — API Flask para Railway

Endpoints:
  GET  /                        health check
  GET  /perfil/<user_id>        perfil completo del usuario
  POST /chat                    turno de conversación con Havi
  GET  /clusters                resumen de los 6 segmentos
  GET  /metricas                métricas de propensión por producto

Variables de entorno requeridas en Railway:
  ANTHROPIC_API_KEY   — clave de la API de Anthropic

Startup automático:
  Si no existen outputs/noise_detector.pkl o outputs/rag_index.pkl,
  el servidor los construye al arrancar (tarda ~3-5 min la primera vez).
  Si no existe outputs/perfiles_usuarios.csv, también corre el pipeline.
=============================================================
"""

import os
import sys
import json
import pickle
import logging
import threading

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# PATHS — Railway despliega en el directorio raíz del repo
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def _out(fn):
    return os.path.join(OUTPUTS_DIR, fn)

def _data(fn):
    return os.path.join(BASE_DIR, fn)


# ─────────────────────────────────────────────
# ESTADO GLOBAL — cargado una sola vez al arrancar
# ─────────────────────────────────────────────
_model     = None   # sentence-transformers
_detector  = None   # noise_detector.pkl
_rag_index = None   # rag_index.pkl
_perfiles  = None   # DataFrame de perfiles_usuarios.csv
_sesiones  = {}     # user_id → HaviSession
_ready     = False  # True cuando todo está cargado
_init_error = None  # mensaje de error si el startup falla


# ─────────────────────────────────────────────
# STARTUP — corre en hilo separado para no bloquear Flask
# ─────────────────────────────────────────────

def _startup():
    global _model, _detector, _rag_index, _perfiles, _ready, _init_error
    try:
        log.info("=" * 55)
        log.info("  HAVI — INICIANDO SERVIDOR")
        log.info("=" * 55)

        # 1. Verificar API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise EnvironmentError(
                "Variable ANTHROPIC_API_KEY no configurada. "
                "Agrégala en Railway → Variables."
            )

        # 2. Cargar o construir perfiles
        perfiles_path = _out('perfiles_usuarios.csv')
        if not os.path.exists(perfiles_path):
            log.info("Perfiles no encontrados — corriendo pipeline v4...")
            _correr_pipeline()
        else:
            log.info(f"Perfiles cargados desde {perfiles_path}")
        _perfiles = pd.read_csv(perfiles_path)
        log.info(f"  {len(_perfiles):,} usuarios | "
                 f"{_perfiles['segmento_nombre'].nunique()} segmentos")

        # 3. Cargar modelo de embeddings
        log.info("Cargando modelo de embeddings...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        log.info("  Modelo cargado (384 dims)")

        # 4. Cargar o construir noise detector
        detector_path = _out('noise_detector.pkl')
        if not os.path.exists(detector_path):
            log.info("Noise detector no encontrado — construyendo...")
            _construir_bases_de_datos()
        _detector  = pickle.load(open(_out('noise_detector.pkl'), 'rb'))
        _rag_index = pickle.load(open(_out('rag_index.pkl'), 'rb'))
        log.info(f"  Noise detector: F1={_detector.get('f1_corpus',0):.3f} "
                 f"| umbral={_detector['umbral']:.3f}")
        log.info(f"  RAG index: {_rag_index['n_docs']:,} documentos")

        _ready = True
        log.info("=" * 55)
        log.info("  SERVIDOR LISTO")
        log.info("=" * 55)

    except Exception as e:
        _init_error = str(e)
        log.error(f"ERROR en startup: {e}")


def _correr_pipeline():
    """Corre hey_pipeline_v4.py como módulo."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'pipeline_v4', _data('hey_pipeline_v4.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _construir_bases_de_datos():
    """Corre hey_databases.py como módulo."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'databases', _data('hey_databases.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# ─────────────────────────────────────────────
# NOISE GATE (replicado del agente para no importar el módulo completo)
# ─────────────────────────────────────────────
import re

_AFIRM_NEG = frozenset([
    'si','sí','no','ok','vale','gracias','perfecto','listo',
    'entendido','claro','de acuerdo','adelante','continuar',
])
_MENU_NAV  = frozenset(['a','b','c','d','1','2','3','4'])
_BROMAS    = ['chiste','te amo','quien eres','quién eres','eres humano',
              'eres ia','gaming','cántame','cantame','me ayudas a ligar',
              'cuéntame un chiste','cuentame un chiste']
_FIN_WORDS = ['tarjeta','cuenta','saldo','transferencia','credito','crédito',
              'pago','banco','inversión','inversion','spei','cajero','cashback',
              'havi','hey','oxxo','dinero','cobro','token','clabe','préstamo']
_W_NOISE   = {
    'artefacto_voz':0.20,'menu_navegacion':0.55,
    'afirmacion_negacion':0.40,'exploracion_corta':0.65,
    'broma_offtopic':0.85,'solo_simbolos':0.90,'on_topic':0.00,
}
NOISE_GATE_UMBRAL_MENSAJE  = 0.55
NOISE_GATE_UMBRAL_USUARIO  = 0.40

_RESP_RUIDO = {
    'broma_offtopic':  ('¡Jaja! 😄 Eso está fuera de mis conocimientos financieros, '
                        'pero con gusto te ayudo con tu cuenta de Hey Banco. ¿Qué necesitas?'),
    'solo_simbolos':   '¿Me puedes escribir en texto lo que necesitas? ¡Estoy aquí!',
    'exploracion_corta':('¿Hay algo en lo que te pueda ayudar? Puedo orientarte con '
                         'tu cuenta, tarjetas, transferencias, créditos o inversiones.'),
}


def _evaluar_noise_semantico(texto):
    """Evalúa ruido usando el detector semántico."""
    # Pre-filtro rápido
    t = str(texto).lower().strip()
    if not t: return True, 'vacio', 1.0, _RESP_RUIDO['exploracion_corta']

    if len(t) > 0 and not re.search(r'[a-záéíóúñ0-9]', t, re.I):
        return True, 'solo_simbolos', 0.90, _RESP_RUIDO['solo_simbolos']
    if any(b in t for b in _BROMAS):
        return True, 'broma_offtopic', 0.90, _RESP_RUIDO['broma_offtopic']
    if t in _AFIRM_NEG:
        return False, 'afirmacion_negacion', 0.40, None
    if t in _MENU_NAV:
        return False, 'menu_navegacion', 0.40, None

    # Embedding + nearest centroid
    vec    = _model.encode([texto], normalize_embeddings=True,
                           show_progress_bar=False)[0]
    sim_n  = float(vec @ _detector['proto_noise'])
    sim_o  = float(vec @ _detector['proto_ontopic'])
    score  = sim_n / (sim_n + sim_o + 1e-9)
    gate   = score >= NOISE_GATE_UMBRAL_MENSAJE
    n_tok  = len(re.findall(r'\w+', t))
    tipo   = 'on_topic' if not gate else ('exploracion_corta' if n_tok <= 3 else 'alta_similitud_noise')
    resp   = _RESP_RUIDO.get('exploracion_corta', None) if gate else None
    return gate, tipo, round(score, 4), resp


def _recuperar_rag(texto, top_k=3, min_sim=0.65):
    """Recupera conversaciones similares del índice."""
    vec  = _model.encode([texto], normalize_embeddings=True,
                         show_progress_bar=False)[0]
    sims = _rag_index['embeddings'] @ vec
    top  = np.argsort(sims)[::-1][:top_k * 3]
    vistos, resultado = set(), []
    for i in top:
        sim = float(sims[i])
        if sim < min_sim: break
        meta = _rag_index['metadata'][i]
        key  = meta['output'][:60]
        if key in vistos: continue
        vistos.add(key)
        resultado.append({
            'pregunta_similar': meta['input'],
            'respuesta_havi':   meta['output'],
            'similitud':        round(sim, 3),
        })
        if len(resultado) >= top_k: break
    return resultado


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

def _construir_system_prompt(perfil, rag_docs=None):
    """Construye el system prompt con las 6 dimensiones del cluster."""
    segmento  = perfil.get('segmento_nombre', 'Usuario de Crédito Moderado')
    tono      = perfil.get('tono', 'amigable y concreto')
    longitud  = str(perfil.get('longitud', 'media'))
    emojis    = str(perfil.get('emojis', 'ocasional'))
    urgencia  = str(perfil.get('urgencia', 'empática'))
    horario   = perfil.get('horario', '')
    estilo    = perfil.get('estilo_comunicacion', 'claro y directo')

    prod_1 = perfil.get('producto_top_1', '')
    prod_2 = perfil.get('producto_top_2', '')
    prod_3 = perfil.get('producto_top_3', '')
    prods  = [p for p in [prod_1, prod_2, prod_3] if p and str(p) != 'nan']

    # Flags
    flag_churn     = bool(int(float(perfil.get('flag_riesgo_churn', 0))))
    flag_estresado = bool(int(float(perfil.get('flag_credito_estresado', 0))))
    flag_atipico   = bool(int(float(perfil.get('flag_uso_atipico', 0))))
    noise_pct      = float(perfil.get('noise_score_usuario', 0))
    es_explorador  = noise_pct >= NOISE_GATE_UMBRAL_USUARIO
    frustracion    = int(float(perfil.get('conv_frustracion', 0)))

    # Traducir dimensiones a instrucciones
    long_map = {
        'corta':  'Máximo 2 oraciones. Ve directo al punto.',
        'media':  'Máximo 4 oraciones. Completo pero sin rodeos.',
        'larga':  'Puedes desarrollar con detalle si el tema lo requiere.',
    }
    emoji_map = {
        'nunca':     'NO uses emojis bajo ninguna circunstancia.',
        'ocasional': 'Puedes usar máximo un emoji si da calidez.',
        'frecuente': 'Puedes usar emojis para hacer la respuesta más amigable.',
    }
    urgencia_map = {
        'inmediata': 'Si el cliente reporta un problema, ofrece solución en el mismo mensaje.',
        'empática':  'Valida brevemente cómo se siente el cliente antes de resolver.',
        'formal':    'Mantén tono profesional incluso ante quejas. Responde con datos y pasos.',
    }

    lk = longitud.split('(')[0].strip().split(' ')[0].lower()
    ek = emojis.split('(')[0].strip().split(' ')[0].lower()
    uk = urgencia.split('(')[0].strip().lower()

    instr_long   = long_map.get(lk, long_map['media'])
    instr_emoji  = emoji_map.get(ek, emoji_map['ocasional'])
    instr_urgenc = urgencia_map.get(uk, urgencia_map['empática'])

    # Alertas
    alertas = []
    if es_explorador:
        alertas.append('⚠ NOISE GATE ACTIVO: NO ofrezcas productos. Solo responde de forma educativa.')
    if flag_estresado:
        alertas.append('⚠ CRÉDITO ESTRESADO: PROHIBIDO ofrecer más crédito. Orienta a diferimiento o MSI.')
    if flag_churn:
        alertas.append('⚠ RIESGO DE ABANDONO: Resuelve completamente antes de cualquier oferta.')
    if flag_atipico:
        alertas.append('⚠ PATRÓN ATÍPICO: Si menciona cargos desconocidos, ofrece aclaración inmediata.')
    if frustracion >= 2:
        alertas.append('⚠ FRUSTRACIÓN HISTÓRICA: Empieza siempre resolviendo, no vendiendo.')
    if not alertas:
        alertas.append('✓ Sin alertas. Puedes responder con normalidad.')

    # RAG
    rag_section = ''
    if rag_docs:
        rag_section = '\n\n## EJEMPLOS REALES DE CONVERSACIONES HAVI\n'
        for i, r in enumerate(rag_docs):
            rag_section += (f"Ejemplo {i+1}:\n"
                            f"Usuario: {r['pregunta_similar']}\n"
                            f"Havi: {r['respuesta_havi']}\n\n")

    prods_txt = ('\n'.join(f"  {i+1}. {p}" for i, p in enumerate(prods))
                 if prods else '  (sin recomendaciones activas)')

    return f"""Eres Havi, el asistente virtual de Hey Banco. Brinda atención personalizada según el perfil de este cliente.

## PERFIL DEL CLIENTE
- Segmento: {segmento}
- Ocupación: {perfil.get('ocupacion','')} | Ingreso: ${float(perfil.get('ingreso_mensual_mxn',0)):,.0f} MXN/mes
- Score buró: {perfil.get('score_buro','')} | Hey Pro: {'Sí' if perfil.get('es_hey_pro') else 'No'}
- NPS: {perfil.get('satisfaccion_1_10','')} / 10

## CÓMO COMUNICARTE CON ESTE CLIENTE (perfil del segmento {segmento})
- Tono: {tono}
- Longitud: {instr_long}
- Emojis: {instr_emoji}
- Ante urgencia: {instr_urgenc}
- Horario de mayor actividad: {horario}
- Estilo: {estilo}

## PRODUCTOS RECOMENDADOS
{prods_txt}
Menciona estos productos solo si emergen naturalmente del contexto.

## ALERTAS ACTIVAS
{chr(10).join(alertas)}
{rag_section}
## REGLAS INVARIABLES
1. Responde siempre en español mexicano natural.
2. Nunca inventes saldos, fechas, montos ni datos que no conozcas.
3. Aplica SIEMPRE las instrucciones de longitud y emojis.
4. Si el cliente pide un humano: 800-123-4567, disponible 24/7.
5. Nunca solicites ni repitas datos sensibles del cliente.
"""


# ─────────────────────────────────────────────
# LLAMADA A CLAUDE API
# ─────────────────────────────────────────────

def _llamar_claude(system_prompt, historial, mensaje):
    import urllib.request
    import urllib.error

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return '[Error: ANTHROPIC_API_KEY no configurada en Railway → Variables]'

    messages = historial + [{'role': 'user', 'content': mensaje}]
    payload  = json.dumps({
        'model':      'claude-sonnet-4-5',
        'max_tokens': 600,
        'system':     system_prompt,
        'messages':   messages,
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={
            'Content-Type':      'application/json',
            'x-api-key':         api_key,
            'anthropic-version': '2023-06-01',
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for block in data.get('content', []):
                if block.get('type') == 'text':
                    return block['text']
            return '[Sin respuesta del modelo]'
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        return f'[Error HTTP {e.code}: {body[:300]}]'
    except Exception as e:
        return f'[Error API: {e}]'


# ─────────────────────────────────────────────
# SESIONES EN MEMORIA
# ─────────────────────────────────────────────

class HaviSession:
    def __init__(self, user_id, perfil_row):
        self.user_id   = user_id
        self.perfil    = perfil_row.to_dict()
        self.historial = []
        self.turno_log = []

    def chat(self, mensaje):
        # Noise gate semántico
        gate, tipo, score, resp_pred = _evaluar_noise_semantico(mensaje)

        if gate and resp_pred:
            respuesta = resp_pred
            self.historial.append({'role': 'user',      'content': mensaje})
            self.historial.append({'role': 'assistant', 'content': respuesta})
            return respuesta, {'gate': True, 'tipo': tipo, 'score': score, 'rag': 0}

        # RAG
        rag_docs = _recuperar_rag(mensaje, top_k=3, min_sim=0.65)

        # System prompt dinámico
        system_prompt = _construir_system_prompt(self.perfil, rag_docs)

        # Claude
        respuesta = _llamar_claude(system_prompt, self.historial, mensaje)

        self.historial.append({'role': 'user',      'content': mensaje})
        self.historial.append({'role': 'assistant', 'content': respuesta})

        return respuesta, {
            'gate': False, 'tipo': tipo, 'score': score,
            'rag': len(rag_docs),
            'rag_sims': [r['similitud'] for r in rag_docs],
        }


# ─────────────────────────────────────────────
# MIDDLEWARE — verificar que el servidor esté listo
# ─────────────────────────────────────────────

def _check_ready():
    if _init_error:
        return jsonify({'error': f'Error de inicialización: {_init_error}'}), 503
    if not _ready:
        return jsonify({'status': 'iniciando', 'mensaje':
                        'El servidor está construyendo las bases de datos. '
                        'Esto tarda ~3-5 minutos la primera vez. '
                        'Vuelve a intentarlo en unos momentos.'}), 503
    return None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    """Health check — Railway lo usa para saber si el servidor arrancó."""
    if _init_error:
        return jsonify({'status': 'error', 'mensaje': _init_error}), 503
    if not _ready:
        return jsonify({'status': 'iniciando',
                        'mensaje': 'Construyendo bases de datos...'}), 200
    return jsonify({
        'status':    'ok',
        'usuarios':  len(_perfiles) if _perfiles is not None else 0,
        'segmentos': int(_perfiles['segmento_nombre'].nunique()) if _perfiles is not None else 0,
        'rag_docs':  _rag_index['n_docs'] if _rag_index else 0,
    })


@app.route('/perfil/<user_id>', methods=['GET'])
def get_perfil(user_id):
    """Devuelve el perfil completo del usuario desde el pipeline."""
    err = _check_ready()
    if err: return err

    row = _perfiles[_perfiles['user_id'] == user_id]
    if row.empty:
        return jsonify({'error': f"Usuario '{user_id}' no encontrado"}), 404

    # Convertir NaN a None para JSON
    perfil = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
              for k, v in row.iloc[0].to_dict().items()}
    return jsonify(perfil)


@app.route('/chat', methods=['POST'])
def chat():
    """
    Turno de conversación con Havi.

    Body JSON:
      user_id   str  — ID del usuario (ej: USR-00001)
      mensaje   str  — mensaje del usuario
      reset     bool — True para reiniciar la sesión (opcional)

    Response JSON:
      respuesta   str   — texto de Havi
      info        dict  — metadata del turno (noise gate, RAG, etc.)
    """
    err = _check_ready()
    if err: return err

    data    = request.json or {}
    user_id = data.get('user_id', '').strip()
    mensaje = data.get('mensaje', '').strip()
    reset   = bool(data.get('reset', False))

    if not user_id:
        return jsonify({'error': 'Falta user_id'}), 400
    if not mensaje:
        return jsonify({'error': 'Falta mensaje'}), 400

    row = _perfiles[_perfiles['user_id'] == user_id]
    if row.empty:
        return jsonify({'error': f"Usuario '{user_id}' no encontrado"}), 404

    # Crear o resetear sesión
    if user_id not in _sesiones or reset:
        _sesiones[user_id] = HaviSession(user_id, row.iloc[0])

    respuesta, info = _sesiones[user_id].chat(mensaje)

    return jsonify({
        'respuesta': respuesta,
        'user_id':   user_id,
        'segmento':  row.iloc[0].get('segmento_nombre', ''),
        'info':      info,
    })


@app.route('/clusters', methods=['GET'])
def get_clusters():
    """Resumen de los 6 segmentos con sus características."""
    err = _check_ready()
    if err: return err

    resumen = []
    for seg, grp in _perfiles.groupby('segmento_nombre'):
        resumen.append({
            'segmento':    seg,
            'n_usuarios':  len(grp),
            'pct':         round(len(grp) / len(_perfiles), 3),
            'tono':        grp['tono'].iloc[0] if 'tono' in grp.columns else '',
            'longitud':    grp['longitud'].iloc[0] if 'longitud' in grp.columns else '',
            'horario':     grp['horario'].iloc[0] if 'horario' in grp.columns else '',
            'noise_pct_medio': round(grp['noise_score_usuario'].mean(), 3),
            'flag_churn_pct':  round(grp['flag_riesgo_churn'].mean(), 3),
        })
    return jsonify(sorted(resumen, key=lambda x: x['n_usuarios'], reverse=True))


@app.route('/metricas', methods=['GET'])
def get_metricas():
    """Métricas de propensión por producto (del pipeline)."""
    metricas_path = _out('metricas_propension.csv')
    if not os.path.exists(metricas_path):
        return jsonify({'error': 'Métricas no disponibles aún'}), 404
    df = pd.read_csv(metricas_path)
    return jsonify(df.to_dict(orient='records'))


@app.route('/usuarios', methods=['GET'])
def get_usuarios():
    """Lista de user_ids disponibles (para testing)."""
    err = _check_ready()
    if err: return err

    segmento = request.args.get('segmento', '')
    n        = int(request.args.get('n', 10))

    df = _perfiles.copy()
    if segmento:
        df = df[df['segmento_nombre'].str.contains(segmento, case=False)]

    cols = ['user_id', 'segmento_nombre', 'tono',
            'producto_top_1', 'noise_score_usuario']
    cols = [c for c in cols if c in df.columns]
    return jsonify(df[cols].head(n).to_dict(orient='records'))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Arrancar el startup en background para no bloquear Flask
    thread = threading.Thread(target=_startup, daemon=True)
    thread.start()

    port = int(os.environ.get('PORT', 5000))
    log.info(f"Flask arrancando en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
