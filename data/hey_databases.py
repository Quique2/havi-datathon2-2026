"""
=============================================================
HEY BANCO — DATATHON 2026
hey_databases.py — Bases de datos semánticas para Havi

Construye y serializa dos bases de datos que el agente usa:

BASE 1 — NOISE DETECTOR (semántico)
  Modelo: paraphrase-multilingual-MiniLM-L12-v2
  Fuente: dataset_50k_anonymized.csv (corpus real de Havi)
  Método: prototipos de embedding por clase (noise / on_topic)
  Mejora vs heurístico: captura matices semánticos que las
    reglas no detectan, ej: "¿tú cobras?" puede ser ruido
    o una pregunta legítima sobre comisiones.

BASE 2 — RAG INDEX (recuperación de conversaciones)
  Modelo: paraphrase-multilingual-MiniLM-L12-v2
  Fuente: dataset_50k_anonymized.csv (conversaciones on_topic)
  Método: embeddings de inputs → índice numpy con cosine sim
  Uso: dado el mensaje actual del usuario, recuperar las
    top-k respuestas más similares que Havi dio en el pasado.
    Esto fundamenta las respuestas del agente en patrones
    REALES de Hey Banco en lugar de respuestas genéricas.

INSTALAR:
  pip install sentence-transformers numpy pandas scikit-learn

EJECUTAR:
  python hey_databases.py
  → outputs/noise_detector.pkl
  → outputs/rag_index.pkl
=============================================================
"""

import os, pickle, re, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

def _resolve(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No se encontró '{filename}' en {script_dir}")

def _out(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ─────────────────────────────────────────────
# MODELO DE EMBEDDINGS (compartido por ambas BDs)
# ─────────────────────────────────────────────
#
# paraphrase-multilingual-MiniLM-L12-v2:
#   - 50+ idiomas incluyendo español mexicano
#   - 384 dimensiones, 118M parámetros
#   - Optimizado para similitud semántica
#   - Ligero: corre bien en CPU (~300ms/batch)

EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


def cargar_modelo():
    """Carga el modelo de embeddings. Se descarga la primera vez (~450 MB)."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Cargando {EMBED_MODEL_NAME}...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"  Modelo cargado OK (384 dimensiones)")
        return model
    except ImportError:
        raise ImportError(
            "Instala sentence-transformers:\n"
            "  pip install sentence-transformers"
        )


def embed_batch(model, textos, batch_size=256, desc=""):
    """Embeddings en batch con progress bar simple."""
    textos  = [str(t) for t in textos]
    total   = len(textos)
    result  = []
    for i in range(0, total, batch_size):
        batch = textos[i:i+batch_size]
        vecs  = model.encode(batch, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
        result.append(vecs)
        if desc:
            print(f"    {desc}: {min(i+batch_size, total)}/{total}", end='\r')
    if desc: print()
    return np.vstack(result)


# ═══════════════════════════════════════════════════════════
# SECCIÓN 1 — ETIQUETADOR DE RUIDO (genera labels de entrenamiento)
# ═══════════════════════════════════════════════════════════
#
# Usamos nuestras heurísticas validadas para etiquetar el corpus.
# Luego los embeddings aprenden representaciones semánticas
# de esos patrones — capturan casos que las reglas no ven.

_AFIRM_NEG = frozenset([
    'si','sí','no','ok','vale','gracias','perfecto','listo',
    'entendido','claro','de acuerdo','adelante','continuar',
    'np','claro que sí','por supuesto',
])
_MENU_NAV = frozenset(['a','b','c','d','1','2','3','4'])
_BROMAS   = [
    'chiste','te amo','te quiero','quien eres','quién eres',
    'eres humano','eres ia','eres real','cántame','cantame',
    'gaming','me ayudas a ligar','cuéntame un chiste',
    'cuentame un chiste','me gustas','eres bonito','eres bonita',
]
_FIN_WORDS = [
    'tarjeta','cuenta','saldo','transferencia','credito','crédito',
    'pago','banco','inversión','inversion','spei','cajero','cashback',
    'havi','hey','oxxo','dinero','cobro','token','clabe','préstamo',
    'prestamo','límite','limite','cargo','movimiento','estado de cuenta',
    'bloquear','reposición','nip','pin','interes','interés','plazo',
]


def etiquetar_ruido(texto):
    """
    Devuelve 'noise' u 'on_topic'.
    Estas etiquetas son ground truth para entrenar el detector semántico.
    """
    if pd.isna(texto): return 'noise'
    t_raw = str(texto)
    t_low = t_raw.lower().strip()
    tokens = re.findall(r'\w+', t_low)
    n_tok  = len(tokens)

    # Artefacto STT
    if re.search(r'\w{3,}\s{2,}\w{3,}\s{2,}\w{3,}', t_raw):
        return 'on_topic'   # ruido técnico pero financieramente relevante

    # Solo símbolos
    if len(t_low) > 0 and not re.search(r'[a-záéíóúñ0-9]', t_low, re.I):
        return 'noise'

    # Broma / off-topic explícito
    if any(b in t_low for b in _BROMAS):
        return 'noise'

    # Menú / afirmación sin contexto financiero
    has_fin = any(re.search(r'\b' + re.escape(w) + r'\b', t_low) for w in _FIN_WORDS)

    if t_low in _MENU_NAV and not has_fin:
        return 'noise'

    if t_low in _AFIRM_NEG and not has_fin:
        return 'noise'

    # Exploración corta sin contenido financiero
    if n_tok <= 2 and not has_fin:
        return 'noise'

    return 'on_topic'


# ═══════════════════════════════════════════════════════════
# BASE DE DATOS 1 — NOISE DETECTOR SEMÁNTICO
# ═══════════════════════════════════════════════════════════

def construir_noise_detector(model, convs):
    """
    Entrena un clasificador semántico noise / on_topic.

    Método: Nearest Centroid con embeddings normalizados.
      1. Etiqueta cada turno del corpus con heurísticas.
      2. Calcula el embedding promedio (prototipo) de cada clase.
      3. En inferencia: cosine sim del mensaje vs cada prototipo.
         La clase con mayor similitud gana.
         Si sim(noise) > UMBRAL → gate activado.

    Ventajas sobre reglas puras:
      - Captura variantes semánticas no listadas en las reglas
      - Aprende del contexto real de Hey Banco
      - Generaliza a nuevas formas de off-topic

    El modelo guarda:
      - prototipo_noise    (384,)  — embedding medio del ruido
      - prototipo_ontopic  (384,)  — embedding medio del on-topic
      - umbral             float   — threshold calibrado en el corpus
      - metricas           dict    — reporte de clasificación
    """
    print("\n" + "═"*60)
    print("  BASE 1 — NOISE DETECTOR SEMÁNTICO")
    print("═"*60)

    # Etiquetar corpus
    convs = convs.copy()
    convs['label'] = convs['input'].apply(etiquetar_ruido)

    dist = convs['label'].value_counts()
    print(f"\n  Distribución de etiquetas:")
    print(f"    on_topic : {dist.get('on_topic', 0):>6,} ({dist.get('on_topic',0)/len(convs):.1%})")
    print(f"    noise    : {dist.get('noise', 0):>6,} ({dist.get('noise',0)/len(convs):.1%})")

    # Embeddings
    print(f"\n  Generando embeddings ({len(convs):,} turnos)...")
    embeddings = embed_batch(model, convs['input'].tolist(), desc="embedding")

    # Prototipos por clase (media de embeddings normalizados)
    mask_noise    = (convs['label'] == 'noise').values
    mask_ontopic  = (convs['label'] == 'on_topic').values

    proto_noise   = embeddings[mask_noise].mean(axis=0)
    proto_ontopic = embeddings[mask_ontopic].mean(axis=0)

    # Normalizar prototipos
    proto_noise   = proto_noise   / (np.linalg.norm(proto_noise)   + 1e-9)
    proto_ontopic = proto_ontopic / (np.linalg.norm(proto_ontopic) + 1e-9)

    # Calcular scores en el corpus completo (validación)
    sim_noise   = embeddings @ proto_noise
    sim_ontopic = embeddings @ proto_ontopic

    # Score de ruido: similitud relativa
    noise_score = sim_noise / (sim_noise + sim_ontopic + 1e-9)

    # Calibrar umbral: maximizar F1 en el corpus (aproximación)
    best_thr, best_f1 = 0.5, 0.0
    from sklearn.metrics import f1_score
    y_true = mask_noise.astype(int)
    for thr in np.linspace(0.35, 0.75, 41):
        y_pred = (noise_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred_final = (noise_score >= best_thr).astype(int)
    print(f"\n  Umbral óptimo: {best_thr:.3f} (F1={best_f1:.4f})")
    print("\n  Reporte de clasificación en corpus:")
    print(classification_report(y_true, y_pred_final,
                                target_names=['on_topic', 'noise'], zero_division=0))

    # Ejemplos de errores para análisis
    fp_idx = np.where((y_pred_final == 1) & (y_true == 0))[0][:5]
    fn_idx = np.where((y_pred_final == 0) & (y_true == 1))[0][:5]
    print("  Falsos positivos (on_topic clasificado como noise):")
    for i in fp_idx:
        print(f"    '{convs.iloc[i]['input'][:70]}'")
    print("  Falsos negativos (noise clasificado como on_topic):")
    for i in fn_idx:
        print(f"    '{convs.iloc[i]['input'][:70]}'")

    detector = {
        'tipo':           'nearest_centroid',
        'modelo':         EMBED_MODEL_NAME,
        'proto_noise':    proto_noise,
        'proto_ontopic':  proto_ontopic,
        'umbral':         best_thr,
        'f1_corpus':      round(best_f1, 4),
        'n_train':        len(convs),
        'dist_noise_pct': round(mask_noise.mean(), 4),
    }

    path = _out('noise_detector.pkl')
    pickle.dump(detector, open(path, 'wb'))
    print(f"\n  → Guardado: {path}")
    return detector


# ═══════════════════════════════════════════════════════════
# BASE DE DATOS 2 — RAG INDEX
# ═══════════════════════════════════════════════════════════

def construir_rag_index(model, convs):
    """
    Construye el índice de recuperación sobre conversaciones on_topic.

    Qué se indexa:
      Cada turno on_topic del corpus: (input del usuario, output de Havi)
      Se embeden los inputs del usuario para buscar por similitud semántica.

    En inferencia:
      Dado el mensaje actual del usuario, se recuperan los top-k pares
      (pregunta_similar, respuesta_havi) que sirvieron bien en el pasado.
      Estos se incluyen como contexto en el system prompt de Claude.

    Esto logra:
      1. Respuestas fundamentadas en patrones REALES de Hey Banco
      2. Consistencia en políticas (qué prometió Havi antes)
      3. Terminología y tono propios del banco

    Estructura del índice:
      - embeddings  (N, 384)  numpy array normalizado
      - metadata    list[dict] con input, output, user_id, date, conv_id
    """
    print("\n" + "═"*60)
    print("  BASE 2 — RAG INDEX")
    print("═"*60)

    # Filtrar solo conversaciones on_topic con output significativo
    convs = convs.copy()
    convs['label'] = convs['input'].apply(etiquetar_ruido)
    convs_ok = convs[
        (convs['label'] == 'on_topic') &
        (convs['output'].str.len() > 30) &
        (convs['input'].str.len() > 5)
    ].copy()

    # Agregar por conversación: usar solo el primer turno de cada conv
    # para tener entradas más autocontenidas
    conv_first = convs_ok.sort_values('date').groupby('conv_id').first().reset_index()

    print(f"\n  Pares input→output seleccionados: {len(conv_first):,}")
    print(f"  (de {len(convs):,} turnos totales, filtrando ruido y outputs cortos)")

    # Embeddings de los inputs
    print(f"\n  Generando embeddings de {len(conv_first):,} inputs...")
    embeddings = embed_batch(model, conv_first['input'].tolist(),
                             batch_size=256, desc="RAG index")

    # Metadata para recuperación
    metadata = []
    for _, row in conv_first.iterrows():
        metadata.append({
            'input':   str(row['input'])[:200],
            'output':  str(row['output'])[:400],
            'user_id': row.get('user_id', ''),
            'conv_id': row.get('conv_id', ''),
        })

    # Estadísticas del índice
    inp_lens  = conv_first['input'].str.len()
    out_lens  = conv_first['output'].str.len()
    print(f"\n  Longitud media input:  {inp_lens.mean():.0f} chars")
    print(f"  Longitud media output: {out_lens.mean():.0f} chars")

    rag_index = {
        'tipo':       'cosine_similarity',
        'modelo':     EMBED_MODEL_NAME,
        'embeddings': embeddings,   # (N, 384) float32
        'metadata':   metadata,
        'n_docs':     len(metadata),
    }

    path = _out('rag_index.pkl')
    pickle.dump(rag_index, open(path, 'wb'))
    print(f"\n  → Guardado: {path}  ({len(metadata):,} documentos)")
    return rag_index


# ═══════════════════════════════════════════════════════════
# FUNCIONES DE INFERENCIA (importadas por el agente)
# ═══════════════════════════════════════════════════════════

def clasificar_noise(texto, detector, model):
    """
    Clasifica un mensaje como noise / on_topic usando el detector semántico.

    Returns:
        es_noise    bool
        score       float [0,1] — probabilidad de ser ruido
        similitudes dict — {'noise': sim_n, 'on_topic': sim_o}
    """
    vec = model.encode([str(texto)], normalize_embeddings=True,
                       show_progress_bar=False)[0]

    sim_n = float(vec @ detector['proto_noise'])
    sim_o = float(vec @ detector['proto_ontopic'])
    score = sim_n / (sim_n + sim_o + 1e-9)

    return score >= detector['umbral'], round(score, 4), {'noise': sim_n, 'on_topic': sim_o}


def recuperar_contexto(texto, rag_index, model, top_k=3, min_sim=0.65):
    """
    Recupera los top_k intercambios más similares del corpus real de Havi.

    Args:
        texto    str   — mensaje actual del usuario
        top_k    int   — número de resultados a recuperar
        min_sim  float — similitud mínima para incluir un resultado

    Returns:
        lista de dicts con 'input', 'output', 'similitud'
    """
    vec = model.encode([str(texto)], normalize_embeddings=True,
                       show_progress_bar=False)[0]

    sims = rag_index['embeddings'] @ vec   # cosine sim (ya normalizados)
    top_idx = np.argsort(sims)[::-1][:top_k * 2]  # candidatos extra para filtrar

    resultados = []
    vistos_outputs = set()
    for i in top_idx:
        sim = float(sims[i])
        if sim < min_sim:
            break
        meta = rag_index['metadata'][i]
        # Evitar duplicar outputs idénticos
        key = meta['output'][:80]
        if key in vistos_outputs:
            continue
        vistos_outputs.add(key)
        resultados.append({
            'pregunta_similar': meta['input'],
            'respuesta_havi':   meta['output'],
            'similitud':        round(sim, 3),
        })
        if len(resultados) >= top_k:
            break

    return resultados


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  HEY BANCO — CONSTRUCCIÓN DE BASES DE DATOS")
    print("═"*60)

    # 1. Cargar conversaciones
    convs = pd.read_csv(_resolve('dataset_50k_anonymized.csv')).dropna(subset=['input'])
    convs['date'] = pd.to_datetime(convs['date'], errors='coerce')
    print(f"\n  Corpus cargado: {len(convs):,} turnos | "
          f"{convs['conv_id'].nunique():,} conversaciones")

    # 2. Cargar modelo de embeddings
    model = cargar_modelo()

    # 3. Noise detector
    detector = construir_noise_detector(model, convs)

    # 4. RAG index
    rag_index = construir_rag_index(model, convs)

    # 5. Test rápido de ambas bases
    print("\n" + "═"*60)
    print("  TEST DE BASES DE DATOS")
    print("═"*60)

    test_msgs = [
        ("¿Cómo bloqueo mi tarjeta si la perdí?",       False),
        ("jaja hola que pedo",                           True),
        ("¿Cuánto es el límite de mi crédito?",          False),
        ("ok",                                           True),
        ("Quiero hacer una transferencia SPEI urgente",  False),
        ("cuéntame un chiste",                           True),
        ("No me procesó el pago en OXXO",                False),
    ]

    print(f"\n  {'Mensaje':<45} {'¿Noise?':^8} {'Score':^7} {'Esperado':^9}")
    print("  " + "─"*75)
    for msg, esperado in test_msgs:
        es_noise, score, _ = clasificar_noise(msg, detector, model)
        correcto = "OK" if es_noise == esperado else "FAIL"
        print(f"  {msg[:44]:<45} {'SI' if es_noise else 'NO':^8} {score:^7.3f} {correcto:^9}")

    print(f"\n  Test RAG — recuperando contexto para mensaje de prueba:")
    recuperados = recuperar_contexto(
        "¿Cuánto tiempo tarda una transferencia SPEI?",
        rag_index, model, top_k=3
    )
    for i, r in enumerate(recuperados):
        print(f"\n  [{i+1}] Similitud={r['similitud']} | "
              f"Pregunta: '{r['pregunta_similar'][:60]}'")
        print(f"       Respuesta: '{r['respuesta_havi'][:100]}...'")

    print("\n" + "═"*60)
    print("  BASES DE DATOS CONSTRUIDAS")
    print("  → outputs/noise_detector.pkl")
    print("  → outputs/rag_index.pkl")
    print("  Siguiente paso: ejecutar hey_agent_havi.py")
    print("═"*60)
