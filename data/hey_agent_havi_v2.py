"""
=============================================================
HEY BANCO — DATATHON 2026
hey_agent_havi.py v2 — Agente con bases de datos semánticas

CAMBIOS RESPECTO A v1
─────────────────────
Noise gate:
  ANTES — heurístico (listas de palabras, regex)
  AHORA — semántico: embeddings + nearest centroid
           entrenado sobre el corpus real de Havi
           Captura variantes no listadas en las reglas

Respuestas:
  ANTES — Claude genera todo desde cero
  AHORA — RAG: recupera top-3 intercambios similares del
           corpus real antes de llamar a Claude.
           Claude recibe los ejemplos como contexto y
           los usa para anclar su respuesta al estilo
           y políticas reales de Hey Banco.

PREREQUISITOS:
  1. pip install sentence-transformers
  2. python hey_databases.py   (construye los índices)
  3. python hey_pipeline_v4.py (genera perfiles)

USO:
  python hey_agent_havi.py --user USR-00001 --demo
  python hey_agent_havi.py --user USR-00001
=============================================================
"""

import os, re, json, pickle, argparse, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

def _resolve(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base in [os.path.join(script_dir, 'outputs'), script_dir]:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No se encontró '{filename}'.\n"
        f"Ejecuta primero:\n"
        f"  python hey_databases.py\n"
        f"  python hey_pipeline_v4.py"
    )


# ═══════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA DE BASES DE DATOS
# ═══════════════════════════════════════════════════════════

def cargar_bases_de_datos():
    """
    Carga los tres recursos que el agente necesita:
      1. Modelo de embeddings (sentence-transformers)
      2. Noise detector (prototipos entrenados sobre corpus Havi)
      3. RAG index (conversaciones reales indexadas)
    """
    print("  Cargando bases de datos...")

    # Modelo de embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("  [OK] Modelo de embeddings")
    except ImportError:
        raise ImportError(
            "Instala sentence-transformers:\n"
            "  pip install sentence-transformers"
        )

    # Noise detector
    detector  = pickle.load(open(_resolve('noise_detector.pkl'), 'rb'))
    umbral    = detector['umbral']
    f1_corpus = detector.get('f1_corpus', 0)
    print(f"  [OK] Noise detector  "
          f"(umbral={umbral:.3f}, F1_corpus={f1_corpus:.3f}, "
          f"n={detector['n_train']:,})")

    # RAG index
    rag_index = pickle.load(open(_resolve('rag_index.pkl'), 'rb'))
    print(f"  [OK] RAG index       ({rag_index['n_docs']:,} conversaciones indexadas)")

    return model, detector, rag_index


# ═══════════════════════════════════════════════════════════
# SECCIÓN 2 — NOISE GATE SEMÁNTICO
# ═══════════════════════════════════════════════════════════

# Umbral de mensaje en tiempo real
# (separado del umbral del detector, que es sobre el corpus)
NOISE_GATE_UMBRAL_MENSAJE  = 0.55
NOISE_GATE_UMBRAL_USUARIO  = 0.40

# Respuestas predefinidas por tipo de ruido
# (evitan gastar tokens del LLM en mensajes off-topic)
_RESPUESTAS_RUIDO = {
    'alta_similitud_noise':   ("¡Hola! Soy Havi, tu asistente de Hey Banco. "
                               "¿En qué te puedo ayudar con tu cuenta hoy? 😊"),
    'exploracion':            ("¿Hay algo específico en lo que te pueda ayudar? "
                               "Puedo orientarte con tu cuenta, tarjetas, "
                               "transferencias, créditos o inversiones."),
    'broma_detectada':        ("¡Jaja! 😄 Eso está fuera de mis conocimientos financieros, "
                               "pero con gusto te ayudo con tu cuenta de Hey Banco. "
                               "¿Qué necesitas?"),
    'simbolos':               ("¿Me puedes escribir en texto lo que necesitas? "
                               "¡Estoy aquí para ayudarte!"),
    'afirmacion':             None,  # respuesta mid-conversación → pasar al LLM
    'navegacion_menu':        None,  # el LLM maneja la navegación con contexto
}

# Patrones rápidos (pre-filtro antes de llamar al modelo de embeddings)
# Evita el costo de embedding para casos trivialmente claros
_SOLO_SIMBOLOS_RE = re.compile(r'^[^\w\sáéíóúñ]+$', re.I)
_BROMAS_RAPIDAS   = ['cuéntame un chiste','cuentame un chiste','te amo',
                     'eres humano','eres ia','me ayudas a ligar']
_AFIRM_NEG_SET    = frozenset(['si','sí','no','ok','vale','gracias',
                               'perfecto','listo','entendido','claro'])
_MENU_SET         = frozenset(['a','b','c','d','1','2','3','4'])


def _prefiltro_rapido(texto):
    """
    Verifica casos triviales sin usar el modelo de embeddings.
    Devuelve (tipo_ruido | None) si detecta algo obvio.
    """
    t = str(texto).lower().strip()
    if not t: return 'exploracion'
    if _SOLO_SIMBOLOS_RE.match(t): return 'simbolos'
    if any(b in t for b in _BROMAS_RAPIDAS): return 'broma_detectada'
    return None


def evaluar_noise_gate(texto, model, detector):
    """
    Evalúa el mensaje actual usando el detector semántico.

    Flujo:
      1. Pre-filtro rápido (regex/strings) — sin embeddings
      2. Si no es trivialmente obvio → embed + nearest centroid
      3. Combinar score con umbral calibrado

    Returns:
        gate_activo  bool
        tipo         str  — categoría del ruido
        score        float [0,1]
        respuesta    str|None — respuesta predefinida si aplica
    """
    # 1. Pre-filtro (0ms)
    tipo_rapido = _prefiltro_rapido(texto)
    if tipo_rapido == 'broma_detectada':
        return True, 'broma_detectada', 0.90, _RESPUESTAS_RUIDO['broma_detectada']
    if tipo_rapido == 'simbolos':
        return True, 'simbolos', 0.95, _RESPUESTAS_RUIDO['simbolos']

    t_low   = str(texto).lower().strip()
    n_tokens = len(re.findall(r'\w+', t_low))

    # Afirmaciones y menú → no gastar embedding
    if t_low in _AFIRM_NEG_SET:
        return False, 'afirmacion', 0.40, None
    if t_low in _MENU_SET:
        return False, 'navegacion_menu', 0.40, None

    # 2. Detector semántico
    vec     = model.encode([texto], normalize_embeddings=True,
                           show_progress_bar=False)[0]
    sim_n   = float(vec @ detector['proto_noise'])
    sim_o   = float(vec @ detector['proto_ontopic'])
    score   = sim_n / (sim_n + sim_o + 1e-9)

    # 3. Umbral de tiempo real (puede diferir del calibrado en corpus)
    gate = score >= NOISE_GATE_UMBRAL_MENSAJE

    if gate:
        if n_tokens <= 3:
            tipo = 'exploracion'
            resp = _RESPUESTAS_RUIDO['exploracion']
        else:
            tipo = 'alta_similitud_noise'
            resp = _RESPUESTAS_RUIDO['alta_similitud_noise']
    else:
        tipo = 'on_topic'
        resp = None

    return gate, tipo, round(score, 4), resp


# ═══════════════════════════════════════════════════════════
# SECCIÓN 3 — RAG: RECUPERACIÓN DE CONTEXTO
# ═══════════════════════════════════════════════════════════

def recuperar_contexto_rag(texto, model, rag_index, top_k=3, min_sim=0.65):
    """
    Busca en el corpus real de Havi los intercambios más similares
    al mensaje actual del usuario.

    Args:
        top_k    int   — máximo de resultados a recuperar
        min_sim  float — similitud coseno mínima para incluir resultado
                         0.65 = semánticamente similar
                         0.80 = casi idéntico

    Returns:
        lista de dicts {pregunta_similar, respuesta_havi, similitud}
    """
    vec  = model.encode([texto], normalize_embeddings=True,
                        show_progress_bar=False)[0]
    sims = rag_index['embeddings'] @ vec

    top_idx = np.argsort(sims)[::-1][:top_k * 3]  # candidatos extra

    resultados   = []
    vistos       = set()
    for i in top_idx:
        sim = float(sims[i])
        if sim < min_sim: break

        meta = rag_index['metadata'][i]
        key  = meta['output'][:60]
        if key in vistos: continue
        vistos.add(key)

        resultados.append({
            'pregunta_similar': meta['input'],
            'respuesta_havi':   meta['output'],
            'similitud':        round(sim, 3),
        })
        if len(resultados) >= top_k:
            break

    return resultados


def formatear_contexto_rag(recuperados):
    """
    Formatea los ejemplos recuperados para incluirlos en el system prompt.
    """
    if not recuperados:
        return ""

    lines = [
        "\n## EJEMPLOS REALES DE CONVERSACIONES HAVI\n",
        "Estos son intercambios reales del banco que pueden orientar tu respuesta.",
        "Úsalos como referencia de tono, terminología y políticas — no los copies literalmente.\n",
    ]
    for i, r in enumerate(recuperados):
        lines.append(f"**Ejemplo {i+1}** (similitud semántica: {r['similitud']:.0%})")
        lines.append(f"Usuario: {r['pregunta_similar']}")
        lines.append(f"Havi:    {r['respuesta_havi']}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# SECCIÓN 4 — CARGA DE PERFIL Y SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════

def cargar_perfil(user_id):
    perfiles = pd.read_csv(_resolve('perfiles_usuarios.csv'))
    row = perfiles[perfiles['user_id'] == user_id]
    if row.empty:
        raise ValueError(f"Usuario '{user_id}' no encontrado.")
    return row.iloc[0].to_dict()


def construir_system_prompt(perfil, contexto_rag=""):
    """
    Genera el system prompt personalizado para Claude.
    Incluye el perfil del usuario y, opcionalmente, ejemplos del RAG.
    """
    user_id       = perfil.get('user_id', '')
    segmento      = perfil.get('segmento_nombre', 'Digital Básico')
    tono          = perfil.get('tono', 'informal')
    estilo        = perfil.get('estilo_comunicacion', 'directo y claro')
    horario       = perfil.get('horario', '')
    prod_1        = perfil.get('producto_top_1', '')
    prod_2        = perfil.get('producto_top_2', '')
    prod_3        = perfil.get('producto_top_3', '')
    productos_rec = [p for p in [prod_1, prod_2, prod_3] if p]

    flag_churn      = bool(int(perfil.get('flag_riesgo_churn', 0)))
    flag_estresado  = bool(int(perfil.get('flag_credito_estresado', 0)))
    flag_atipico    = bool(int(perfil.get('flag_uso_atipico', 0)))
    noise_usuario   = float(perfil.get('noise_score_usuario', 0))
    es_explorador   = noise_usuario >= NOISE_GATE_UMBRAL_USUARIO
    frustracion     = int(perfil.get('conv_frustracion', 0))

    ingreso    = perfil.get('ingreso_mensual_mxn', 0)
    score_buro = perfil.get('score_buro', 0)
    antiguedad = int(perfil.get('antiguedad_dias', 0)) // 365
    ocupacion  = perfil.get('ocupacion', '')
    es_pro     = bool(perfil.get('es_hey_pro', False))
    nps        = perfil.get('satisfaccion_1_10', 0)
    dias_login = int(perfil.get('dias_desde_ultimo_login', 0))

    prompt = f"""Eres Havi, el asistente virtual de Hey Banco. Eres profesional, empático y estás diseñado para ayudar al cliente de forma personalizada.

## PERFIL DEL CLIENTE ({user_id})

- **Segmento:** {segmento}
- **Perfil:** {ocupacion}, ingreso ~${ingreso:,.0f} MXN/mes, {antiguedad} año(s) como cliente
- **Score crediticio:** {score_buro} / 850
- **Hey Pro:** {'Sí' if es_pro else 'No'}
- **Satisfacción (NPS):** {nps}/10
- **Último login:** hace {dias_login} días

## ESTILO DE COMUNICACIÓN

- **Tono:** {tono}
- **Estilo:** {estilo}
- **Horario de mayor actividad:** {horario}

## PRODUCTOS RECOMENDADOS

Productos con mayor probabilidad de ser útiles para este cliente:
{chr(10).join(f"  {i+1}. {p}" for i, p in enumerate(productos_rec)) if productos_rec else "  (sin recomendaciones activas)"}

Menciona estos productos solo si son relevantes al contexto. No los ofrezcas en cada respuesta.
"""

    # Instrucciones condicionales
    instrucciones = []

    if es_explorador:
        instrucciones.append(
            "- Este cliente usa el chat de forma exploratoria. "
            "NO ofrezcas productos. Responde de forma educativa y orientadora."
        )
    if flag_estresado:
        instrucciones.append(
            "- Utilización de crédito >85%. NO ofrezcas más crédito. "
            "Orienta hacia diferimiento o plan de pagos si el tema surge."
        )
    if flag_churn:
        instrucciones.append(
            "- Cliente con riesgo de abandono (inactivo + NPS bajo). "
            "Prioriza resolver su problema. Tono empático."
        )
    if flag_atipico:
        instrucciones.append(
            "- Patrones de uso atípicos detectados. "
            "Si menciona movimientos desconocidos, ofrece iniciar aclaración."
        )
    if frustracion >= 2:
        instrucciones.append(
            "- Ha expresado frustración en conversaciones previas. "
            "Resuelve primero, vende después."
        )
    if not instrucciones:
        instrucciones.append(
            "- Cliente activo y sin alertas. "
            "Puedes mencionar los productos recomendados si el contexto lo permite."
        )

    prompt += "\n## INSTRUCCIONES ESPECÍFICAS\n" + "\n".join(instrucciones)

    # Contexto RAG
    if contexto_rag:
        prompt += "\n" + contexto_rag

    prompt += """

## REGLAS GENERALES

1. Responde siempre en español mexicano.
2. Nunca inventes saldos, fechas, montos o datos que no conozcas.
3. Si no sabes algo, dilo claramente y ofrece alternativas.
4. Respuestas concisas — máximo 3 párrafos salvo que el cliente pida más.
5. Si el cliente pide hablar con un humano: indica que puede llamar al 800-123-4567.
6. Nunca compartas datos sensibles del cliente en la respuesta.
"""
    return prompt


# ═══════════════════════════════════════════════════════════
# SECCIÓN 5 — LLAMADA A CLAUDE API
# ═══════════════════════════════════════════════════════════

def llamar_claude(system_prompt, historial, mensaje_usuario):
    """
    Llama a la API de Anthropic con el system prompt y el historial.
    Usa urllib para no requerir la librería anthropic instalada.

    API KEY — dos formas de configurarla (usa solo una):

    Opción A (recomendada): variable de entorno — no guardes la clave en el código.
      Windows PowerShell:
        $env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
      Windows CMD:
        set ANTHROPIC_API_KEY=sk-ant-api03-...
      Mac/Linux:
        export ANTHROPIC_API_KEY="sk-ant-api03-..."

    Opción B (solo pruebas locales): escribe tu clave directamente abajo.
      Cambia "" por "sk-ant-api03-..." en la línea API_KEY_DIRECTO.
    """
    import os, urllib.request, urllib.error

    # ── Opción A: variable de entorno ──────────────────────
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Opción B: clave directa (solo para pruebas locales) ─
    # API_KEY = "sk-ant-api03-PEGA-TU-CLAVE-AQUI"

    if not API_KEY:
        return (
            "[Error: falta la API key de Anthropic]\n"
            "Configúrala así en PowerShell antes de correr el agente:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-api03-...'"
        )

    messages = historial + [{"role": "user", "content": mensaje_usuario}]

    payload = json.dumps({
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system":     system_prompt,
        "messages":   messages,
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for block in data.get('content', []):
                if block.get('type') == 'text':
                    return block['text']
            return "[Sin respuesta del modelo]"
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        return f"[Error HTTP {e.code}: {body[:200]}]"
    except Exception as e:
        return f"[Error API: {e}]"


# ═══════════════════════════════════════════════════════════
# SECCIÓN 6 — SESIÓN DE CONVERSACIÓN
# ═══════════════════════════════════════════════════════════

class HaviSession:
    """
    Gestiona una sesión completa de conversación.

    Mejoras v2:
      - Noise gate semántico (paraphrase-multilingual-MiniLM-L12-v2)
      - RAG: cada turno on_topic recupera contexto del corpus real
      - El system prompt se actualiza dinámicamente con el contexto RAG
      - Registro de métricas por turno para análisis

    Flujo por turno:
      1. Pre-filtro rápido (sin embedding)
      2. Si no es trivial → embedding + noise gate semántico
      3. Si es noise → respuesta predefinida (sin LLM)
      4. Si es on_topic → RAG + Claude API
    """

    def __init__(self, user_id, model, detector, rag_index):
        self.user_id    = user_id
        self.model      = model
        self.detector   = detector
        self.rag_index  = rag_index
        self.perfil     = cargar_perfil(user_id)
        self.historial  = []
        self.turno_log  = []
        self.turno      = 0

        self._imprimir_resumen_perfil()

    def _imprimir_resumen_perfil(self):
        p = self.perfil
        print(f"\n  {'─'*50}")
        print(f"  Perfil cargado: {self.user_id}")
        print(f"  Segmento:       {p.get('segmento_nombre','?')}")
        print(f"  Tono:           {p.get('tono','?')}")
        print(f"  Prod. top-1:    {p.get('producto_top_1','(ninguno)')}")
        print(f"  Noise hist.:    {float(p.get('noise_score_usuario',0)):.0%}")
        flags = []
        if int(p.get('flag_riesgo_churn',0)):      flags.append('churn')
        if int(p.get('flag_credito_estresado',0)):  flags.append('crédito_estresado')
        if int(p.get('flag_uso_atipico',0)):        flags.append('uso_atípico')
        print(f"  Flags:          {', '.join(flags) if flags else 'ninguno'}")
        print(f"  {'─'*50}")

    def chat(self, mensaje_usuario):
        """
        Procesa un turno de conversación.

        Returns:
            respuesta    str
            turno_info   dict — métricas del turno
        """
        self.turno += 1
        turno_info = {
            'turno':        self.turno,
            'mensaje':      mensaje_usuario[:80],
            'noise_gate':   False,
            'noise_tipo':   '',
            'noise_score':  0.0,
            'n_rag_docs':   0,
            'rag_sims':     [],
        }

        # ── NOISE GATE SEMÁNTICO ───────────────────────────
        gate, tipo, score, respuesta_pred = evaluar_noise_gate(
            mensaje_usuario, self.model, self.detector
        )

        turno_info['noise_gate']  = gate
        turno_info['noise_tipo']  = tipo
        turno_info['noise_score'] = score

        if gate and respuesta_pred is not None:
            # Respuesta predefinida — sin llamar al LLM
            respuesta = respuesta_pred
            self.historial.append({"role": "user",      "content": mensaje_usuario})
            self.historial.append({"role": "assistant",  "content": respuesta})
            self.turno_log.append(turno_info)
            return respuesta, turno_info

        # ── RAG: recuperar contexto ────────────────────────
        recuperados = recuperar_contexto_rag(
            mensaje_usuario, self.model, self.rag_index,
            top_k=3, min_sim=0.65
        )
        turno_info['n_rag_docs'] = len(recuperados)
        turno_info['rag_sims']   = [r['similitud'] for r in recuperados]

        # ── SYSTEM PROMPT DINÁMICO ─────────────────────────
        contexto_rag  = formatear_contexto_rag(recuperados)
        system_prompt = construir_system_prompt(self.perfil, contexto_rag)

        # ── LLAMADA A CLAUDE ───────────────────────────────
        respuesta = llamar_claude(system_prompt, self.historial, mensaje_usuario)

        self.historial.append({"role": "user",      "content": mensaje_usuario})
        self.historial.append({"role": "assistant",  "content": respuesta})
        self.turno_log.append(turno_info)

        return respuesta, turno_info

    def resumen_sesion(self):
        n = self.turno
        if n == 0:
            return {}
        n_noise    = sum(1 for t in self.turno_log if t['noise_gate'])
        noise_med  = sum(t['noise_score'] for t in self.turno_log) / n
        n_rag      = sum(t['n_rag_docs'] for t in self.turno_log)
        sim_med    = (sum(s for t in self.turno_log for s in t['rag_sims']) /
                      max(1, sum(len(t['rag_sims']) for t in self.turno_log)))
        return {
            'user_id':           self.user_id,
            'segmento':          self.perfil.get('segmento_nombre'),
            'total_turnos':      n,
            'turnos_noise':      n_noise,
            'pct_noise':         round(n_noise / n, 3),
            'noise_score_medio': round(noise_med, 3),
            'docs_rag_total':    n_rag,
            'rag_sim_media':     round(sim_med, 3),
        }


# ═══════════════════════════════════════════════════════════
# SECCIÓN 7 — DEMO Y CLI
# ═══════════════════════════════════════════════════════════

DEMO_CONVERSACION = [
    ("Hola",                                              True),
    ("¿Cuánto debo de mi tarjeta de crédito?",            False),
    ("B",                                                  True),
    ("¿Qué pasa si no pago el mínimo esta quincena?",     False),
    ("jaja ok cuéntame un chiste",                         True),
    ("¿Cómo funciona la inversión Hey?",                   False),
    ("No me deja entrar a la app, me da error de token",  False),
    ("gracias",                                            False),
]


def run_demo(user_id, model, detector, rag_index):
    print("\n" + "═"*60)
    print(f"  DEMO — AGENTE HAVI v2 PARA {user_id}")
    print(f"  Noise gate: SEMÁNTICO  |  RAG: corpus real Havi")
    print("═"*60)

    session = HaviSession(user_id, model, detector, rag_index)

    for msg, es_noise_esperado in DEMO_CONVERSACION:
        respuesta, info = session.chat(msg)

        gate_str = "[GATE]" if info['noise_gate'] else "      "
        rag_str  = f"RAG:{info['n_rag_docs']}" if info['n_rag_docs'] > 0 else "RAG:0"
        print(f"\n  {gate_str} score={info['noise_score']:.2f} | "
              f"{info['noise_tipo']:<22} | {rag_str}")
        print(f"  Usuario: {msg}")
        print(f"  Havi:    {respuesta[:130]}{'...' if len(respuesta)>130 else ''}")

    r = session.resumen_sesion()
    print("\n" + "─"*60)
    print(f"  RESUMEN DE SESIÓN")
    print(f"  Turnos:          {r['total_turnos']}")
    print(f"  Noise gate:      {r['turnos_noise']} ({r['pct_noise']:.0%})")
    print(f"  Docs RAG usados: {r['docs_rag_total']}")
    print(f"  Sim. RAG media:  {r['rag_sim_media']:.3f}")
    print("─"*60)


def run_interactivo(user_id, model, detector, rag_index):
    print("\n" + "═"*60)
    print(f"  HAVI v2 — MODO INTERACTIVO ({user_id})")
    print("  'salir' para terminar | 'resumen' para ver métricas")
    print("═"*60)

    session = HaviSession(user_id, model, detector, rag_index)
    print("\n  Havi: ¡Hola! Soy Havi, tu asistente de Hey Banco. "
          "¿En qué te puedo ayudar?\n")

    while True:
        try:
            msg = input("  Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not msg: continue
        if msg.lower() == 'salir': break
        if msg.lower() == 'resumen':
            r = session.resumen_sesion()
            print(f"\n  Sesión | turnos={r['total_turnos']} | "
                  f"noise={r['pct_noise']:.0%} | "
                  f"RAG docs={r['docs_rag_total']} | "
                  f"sim_media={r['rag_sim_media']:.3f}\n")
            continue

        respuesta, info = session.chat(msg)
        gate_str = f" [noise:{info['noise_tipo']}]" if info['noise_gate'] else ""
        rag_str  = f" [RAG:{info['n_rag_docs']}]" if info['n_rag_docs'] > 0 else ""
        print(f"\n  Havi{gate_str}{rag_str}: {respuesta}\n")

    r = session.resumen_sesion()
    print(f"\n  Sesión terminada. {r.get('total_turnos',0)} turnos | "
          f"noise={r.get('pct_noise',0):.0%}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agente Havi v2 — Hey Banco')
    parser.add_argument('--user', default='USR-00001',
                        help='ID del usuario (ej: USR-00001)')
    parser.add_argument('--demo', action='store_true',
                        help='Ejecutar conversación demo predefinida')
    args = parser.parse_args()

    # Cargar bases de datos (una sola vez por proceso)
    model, detector, rag_index = cargar_bases_de_datos()

    if args.demo:
        run_demo(args.user, model, detector, rag_index)
    else:
        run_interactivo(args.user, model, detector, rag_index)
