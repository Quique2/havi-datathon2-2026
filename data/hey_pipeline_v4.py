"""
=============================================================
HEY BANCO — DATATHON 2026
Pipeline v4 — Perfil de usuario + Propensión por producto

RESPONSABILIDAD ÚNICA:
  Dado el historial de un usuario, produce:
    1. Segmento de comportamiento (KMeans)
    2. Probabilidad de adopción por producto (GBM por producto)
    3. Perfil de comunicación (tono, canal, horario)
    4. Flags de riesgo (churn, atípico)
    5. Score de ruido conversacional (noise gate pasivo)

OUTPUT:
  perfiles_usuarios.csv  — 1 fila por usuario, listo para el agente
  modelos/               — modelos serializados (.pkl)

Este pipeline NO decide qué decirle al cliente.
Esa responsabilidad es del Agente Havi (hey_agent_havi.py).
=============================================================
"""

import os, re, pickle, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score, davies_bouldin_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

def _resolve(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"\n  No se encontró '{filename}'"
        f"\n  Buscado en: {script_dir}"
    )

def _out(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

def _models_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(script_dir, 'outputs', 'modelos')
    os.makedirs(d, exist_ok=True)
    return d


PATH_CLIENTES = _resolve('hey_clientes.csv')
PATH_PRODUCTOS = _resolve('hey_productos.csv')
PATH_TRANSACC  = _resolve('hey_transacciones.csv')
PATH_CONVS     = _resolve('dataset_50k_anonymized.csv')


# ═══════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA Y LIMPIEZA
# ═══════════════════════════════════════════════════════════

def cargar_y_limpiar():
    print("\n" + "═"*60)
    print("  CARGA Y LIMPIEZA")
    print("═"*60)

    # Clientes
    cl = pd.read_csv(PATH_CLIENTES)
    if 'sexo' in cl.columns:
        cl.rename(columns={'sexo': 'genero'}, inplace=True)
    cl['estado'] = cl['estado'].fillna('Desconocido')
    cl['ciudad'] = cl['ciudad'].fillna('Desconocido')
    med = cl.groupby('ocupacion')['satisfaccion_1_10'].transform('median')
    cl['satisfaccion_1_10'] = cl['satisfaccion_1_10'].fillna(med).fillna(7.5)
    for col in ['es_hey_pro','nomina_domiciliada','recibe_remesas',
                'usa_hey_shop','tiene_seguro','patron_uso_atipico']:
        if cl[col].dtype == object:
            cl[col] = cl[col].map({'True': True, 'False': False})
    print(f"  [clientes]       {len(cl):,} usuarios")

    # Productos
    pr = pd.read_csv(PATH_PRODUCTOS)
    pr['fecha_apertura']          = pd.to_datetime(pr['fecha_apertura'], errors='coerce')
    pr['fecha_ultimo_movimiento'] = pd.to_datetime(pr['fecha_ultimo_movimiento'], errors='coerce')
    mask = pr['utilizacion_pct'].notna() & ((pr['utilizacion_pct'] < 0) | (pr['utilizacion_pct'] > 1))
    pr.loc[mask, 'utilizacion_pct'] = np.nan
    print(f"  [productos]      {len(pr):,} registros")

    # Transacciones
    tx = pd.read_csv(PATH_TRANSACC)
    tx['fecha_hora'] = pd.to_datetime(tx['fecha_hora'], errors='coerce')
    n = len(tx)
    tx = tx.sort_values('fecha_hora').drop_duplicates(subset='transaccion_id', keep='first')
    tx['ciudad_transaccion'] = tx['ciudad_transaccion'].fillna('Desconocida')
    tx['descripcion_libre']  = tx['descripcion_libre'].fillna('')
    for col in ['es_internacional', 'patron_uso_atipico']:
        if tx[col].dtype == object:
            tx[col] = tx[col].map({'True': True, 'False': False})
    print(f"  [transacciones]  {len(tx):,} ({n-len(tx)} duplicados eliminados)")

    # Conversaciones
    cv = pd.read_csv(PATH_CONVS).dropna(subset=['input'])
    cv['date']      = pd.to_datetime(cv['date'], errors='coerce')
    cv['es_voz']    = (cv['channel_source'] == 2).astype(int)
    cv['input_len'] = cv['input'].str.len()
    print(f"  [conversaciones] {len(cv):,} turnos")

    return cl, pr, tx, cv


# ═══════════════════════════════════════════════════════════
# SECCIÓN 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

# ── Intents ──────────────────────────────────────────────
_INTENT_RULES = {
    'credito':       ['crédito','credito','préstamo','prestamo','financiamiento'],
    'transferencia': ['transferencia','spei','clabe','enviar','envío'],
    'tarjeta':       ['tarjeta','bloquear','robo','cancelar'],
    'inversion':     ['inversión','inversion','rendimiento','gat','ahorro'],
    'aclaracion':    ['cobro','cargo','disputa','reembolso','no reconozco'],
    'seguridad':     ['contraseña','nip','pin','token','bloqueado'],
    'negocio':       ['negocio','empresa','terminal','pos','factura'],
    'seguro':        ['seguro','vida','protección'],
}

def _intent(t):
    if pd.isna(t) or not str(t).strip(): return 'otro'
    t = str(t).lower()
    for k, ws in _INTENT_RULES.items():
        if any(w in t for w in ws): return k
    return 'otro'

# ── Noise ─────────────────────────────────────────────────
_AFIRM_NEG = frozenset([
    'si','sí','no','ok','vale','gracias','perfecto','listo',
    'entendido','claro','de acuerdo','adelante','continuar',
])
_MENU_NAV  = frozenset(['a','b','c','d','1','2','3','4'])
_BROMAS    = ['chiste','te amo','quien eres','eres humano','eres ia',
              'cántame','gaming','me ayudas a ligar']
_PALABRAS_FIN = ['tarjeta','cuenta','saldo','transferencia','credito','crédito',
                 'pago','banco','inversión','inversion','spei','cajero','cashback',
                 'havi','hey','oxxo','dinero','cobro','token','clabe']
_W_NOISE = {
    'artefacto_voz': 0.20, 'menu_navegacion': 0.55,
    'afirmacion_negacion': 0.40, 'exploracion_corta': 0.65,
    'broma_offtopic': 0.85, 'solo_simbolos': 0.90, 'on_topic': 0.0,
}

def _noise_score(texto):
    """Devuelve (noise_type, noise_score) para un turno."""
    if pd.isna(texto): return 'nulo', 1.0
    t_raw = str(texto); t_low = t_raw.lower().strip()
    tokens = re.findall(r'\w+', t_low)
    n_tok  = len(tokens)
    is_voz = int(bool(re.search(r'\w{3,}\s{2,}\w{3,}\s{2,}\w{3,}', t_raw)))
    is_sym = int(len(t_low) > 0 and not re.search(r'[a-záéíóúñ0-9]', t_low, re.I))
    is_broma = int(any(b in t_low for b in _BROMAS))
    is_menu  = int(t_low in _MENU_NAV)
    is_afirm = int(t_low in _AFIRM_NEG)
    has_fin  = int(any(re.search(r'\b' + re.escape(w) + r'\b', t_low) for w in _PALABRAS_FIN))
    is_expl  = int(n_tok <= 2 and not has_fin and not is_afirm and not is_menu)

    if   is_sym:   nt = 'solo_simbolos'
    elif is_broma: nt = 'broma_offtopic'
    elif is_voz:   nt = 'artefacto_voz'
    elif is_menu:  nt = 'menu_navegacion'
    elif is_afirm: nt = 'afirmacion_negacion'
    elif is_expl:  nt = 'exploracion_corta'
    else:          nt = 'on_topic'

    raw = _W_NOISE[nt]
    if has_fin and raw > 0: raw *= 0.45
    return nt, round(min(1.0, raw), 4)


def features_transacciones(tx):
    print("  [features] transacciones...")
    agg = tx.groupby('user_id').agg(
        txn_total          = ('transaccion_id', 'count'),
        txn_monto_prom     = ('monto', 'mean'),
        txn_monto_std      = ('monto', 'std'),
        txn_monto_total    = ('monto', 'sum'),
        txn_n_categorias   = ('categoria_mcc', 'nunique'),
        txn_n_dias_activo  = ('fecha_hora', lambda x: x.dt.date.nunique()),
        txn_pct_digital    = ('canal', lambda x: x.isin(['app_ios','app_android','app_huawei','codi']).mean()),
        txn_pct_efectivo   = ('canal', lambda x: x.isin(['cajero_banregio','cajero_externo','oxxo','farmacia_ahorro']).mean()),
        txn_pct_inter      = ('es_internacional', 'mean'),
        txn_n_inter        = ('es_internacional', 'sum'),
        txn_pct_atipico    = ('patron_uso_atipico', 'mean'),
        txn_pct_fallida    = ('estatus', lambda x: (x == 'no_procesada').mean()),
        txn_n_intentos_ext = ('intento_numero', lambda x: (x > 1).sum()),
        txn_n_viajes       = ('categoria_mcc', lambda x: (x == 'viajes').sum()),
        txn_n_servicios_dig= ('categoria_mcc', lambda x: (x == 'servicios_digitales').sum()),
        txn_n_supermercado = ('categoria_mcc', lambda x: (x == 'supermercado').sum()),
        txn_n_restaurante  = ('categoria_mcc', lambda x: (x == 'restaurante').sum()),
        txn_n_msi          = ('meses_diferidos', lambda x: x.notna().sum()),
        txn_cashback_total = ('cashback_generado', 'sum'),
        txn_hora_media     = ('hora_del_dia', 'mean'),
        txn_pct_finde      = ('dia_semana', lambda x: x.isin(['Saturday','Sunday']).mean()),
        txn_pct_nocturno   = ('hora_del_dia', lambda x: ((x >= 22) | (x <= 5)).mean()),
    ).reset_index()
    agg['txn_volatilidad']  = agg['txn_monto_std'] / (agg['txn_monto_prom'] + 1)
    agg['txn_frec_por_dia'] = agg['txn_total'] / (agg['txn_n_dias_activo'] + 1)
    print(f"    → {agg.shape[1]-1} features | {len(agg):,} usuarios")
    return agg


def features_productos(pr):
    print("  [features] productos...")
    act = pr[pr['estatus'] == 'activo']
    agg = pr.groupby('user_id').agg(
        prod_n_total      = ('producto_id', 'count'),
        prod_n_activos    = ('estatus', lambda x: (x == 'activo').sum()),
        prod_n_cancelados = ('estatus', lambda x: (x == 'cancelado').sum()),
        prod_n_revision   = ('estatus', lambda x: (x == 'revision_de_pagos').sum()),
        prod_util_media   = ('utilizacion_pct', 'mean'),
        prod_util_max     = ('utilizacion_pct', 'max'),
        prod_saldo_total  = ('saldo_actual', 'sum'),
        prod_limite_total = ('limite_credito', 'sum'),
    ).reset_index()
    agg['prod_util_media'] = agg['prod_util_media'].fillna(0)
    agg['prod_util_max']   = agg['prod_util_max'].fillna(0)
    agg['prod_ratio_problematicos'] = (
        agg['prod_n_cancelados'] + agg['prod_n_revision']
    ) / (agg['prod_n_total'] + 1)

    # Flags de productos activos (para excluirlos de propensión)
    for prod in ['tarjeta_credito_hey','tarjeta_credito_garantizada',
                 'credito_personal','credito_nomina','credito_auto',
                 'inversion_hey','seguro_vida','seguro_compras',
                 'cuenta_negocios','tarjeta_credito_negocios']:
        col = f'tiene_{prod}'
        tiene = act[act['tipo_producto'] == prod]['user_id'].unique()
        agg[col] = agg['user_id'].isin(tiene).astype(int)

    print(f"    → {agg.shape[1]-1} features | {len(agg):,} usuarios")
    return agg


def features_conversaciones(cv):
    print("  [features] conversaciones...")
    cv = cv.copy()
    cv['intent'] = cv['input'].apply(_intent)
    cv[['noise_type','noise_score']] = cv['input'].apply(
        lambda t: pd.Series(_noise_score(t))
    )

    # Turnos por conversación
    tc    = cv.groupby('conv_id')['input'].count().reset_index(name='turnos_conv')
    cv    = cv.merge(tc, on='conv_id', how='left')

    agg = cv.groupby('user_id').agg(
        conv_n_total        = ('conv_id', 'nunique'),
        conv_turnos_prom    = ('turnos_conv', 'mean'),
        conv_turnos_max     = ('turnos_conv', 'max'),
        conv_pct_voz        = ('es_voz', 'mean'),
        conv_input_len_med  = ('input_len', 'mean'),
        conv_urgencia       = ('input', lambda x: x.str.lower().str.contains(
                                '|'.join(['urgente','urge','inmediato','rápido']),
                                na=False, regex=True).sum()),
        conv_frustracion    = ('input', lambda x: x.str.lower().str.contains(
                                '|'.join(['error','no funciona','no me deja','problema','falla']),
                                na=False, regex=True).sum()),
        conv_noise_pct      = ('noise_score', lambda x: (x > 0.35).mean()),
        conv_noise_score    = ('noise_score', 'mean'),
        conv_noise_max      = ('noise_score', 'max'),
        intent_credito      = ('intent', lambda x: (x == 'credito').sum()),
        intent_inversion    = ('intent', lambda x: (x == 'inversion').sum()),
        intent_tarjeta      = ('intent', lambda x: (x == 'tarjeta').sum()),
        intent_aclaracion   = ('intent', lambda x: (x == 'aclaracion').sum()),
        intent_seguridad    = ('intent', lambda x: (x == 'seguridad').sum()),
        intent_negocio      = ('intent', lambda x: (x == 'negocio').sum()),
    ).reset_index()

    # Reintentos < 48h (conversaciones no resueltas)
    cv_s = cv.sort_values(['user_id','date'])
    cv_s['prev_date'] = cv_s.groupby('user_id')['date'].shift(1)
    cv_s['hrs_gap']   = (cv_s['date'] - cv_s['prev_date']).dt.total_seconds() / 3600
    reintentos = cv_s[cv_s['hrs_gap'] < 48].groupby('user_id').size()
    agg = agg.merge(reintentos.rename('conv_reintentos_48h'), on='user_id', how='left')
    agg['conv_reintentos_48h'] = agg['conv_reintentos_48h'].fillna(0)

    # Noise trend (más ruidoso al final = explorador persistente)
    def _trend(grp):
        n = len(grp)
        if n < 4: return 0.0
        mid = n // 2
        return round(grp.iloc[mid:]['noise_score'].mean() -
                     grp.iloc[:mid]['noise_score'].mean(), 4)

    trend = cv.sort_values(['user_id','date']).groupby('user_id').apply(_trend).reset_index()
    trend.columns = ['user_id','conv_noise_trend']
    agg = agg.merge(trend, on='user_id', how='left')
    agg['conv_noise_trend'] = agg['conv_noise_trend'].fillna(0)

    print(f"    → {agg.shape[1]-1} features | {len(agg):,} usuarios")
    return agg


def construir_master(cl, pr, tx, cv):
    print("\n  [master] uniendo tablas...")
    f_tx = features_transacciones(tx)
    f_pr = features_productos(pr)
    f_cv = features_conversaciones(cv)

    m = cl.copy()
    m = m.merge(f_tx, on='user_id', how='left')
    m = m.merge(f_pr, on='user_id', how='left')
    m = m.merge(f_cv, on='user_id', how='left')
    m = m.fillna(0)

    # ── NUEVO: codificar columnas categóricas ──────────────
    for col in ['nivel_educativo', 'ocupacion', 'genero',
                'estado', 'ciudad', 'preferencia_canal',
                'canal_apertura', 'idioma_preferido']:
        if col in m.columns:
            m[col] = m[col].astype('category').cat.codes

    print(f"  → Master: {m.shape[0]:,} usuarios × {m.shape[1]} columnas")
    return m


# ═══════════════════════════════════════════════════════════
# SECCIÓN 3 — SEGMENTACIÓN
# ═══════════════════════════════════════════════════════════

FEATURES_CLUSTERING = [
    'edad','ingreso_mensual_mxn','score_buro','antiguedad_dias',
    'txn_monto_prom','txn_pct_digital','txn_pct_efectivo','txn_pct_inter',
    'txn_n_viajes','txn_volatilidad','txn_frec_por_dia','txn_pct_nocturno',
    'txn_pct_atipico','txn_n_msi',
    'prod_n_activos','prod_util_media','prod_util_max','prod_ratio_problematicos',
    'conv_n_total','conv_pct_voz','conv_input_len_med',
    'conv_urgencia','conv_frustracion','conv_noise_pct','conv_noise_score',
    'intent_credito','intent_inversion','intent_negocio','intent_aclaracion',
]

NOMBRES_SEGMENTO = {
    0: 'PYME Emprendedor',
    1: 'Digital Básico',
    2: 'Premium Digital',
    3: 'Familia Establecida',
    4: 'Ahorrador Cauteloso',
}


def segmentar(master, n_clusters=5, random_state=42):
    print("\n" + "═"*60)
    print("  SEGMENTACIÓN")
    print("═"*60)
    cols  = [c for c in FEATURES_CLUSTERING if c in master.columns]
    X     = master[cols].copy()
    scaler = StandardScaler()
    Xs    = scaler.fit_transform(X)

    print("\n  Evaluando k óptimo...")
    for k in range(2, 8):
        km  = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbs = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lbs, sample_size=3000)
        print(f"    k={k}  silhouette={sil:.4f}")

    km5 = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    master['segmento_id'] = km5.fit_predict(Xs)
    sil5 = silhouette_score(Xs, master['segmento_id'], sample_size=5000)
    db   = davies_bouldin_score(Xs, master['segmento_id'])
    print(f"\n  k={n_clusters} → Silhouette={sil5:.4f} | Davies-Bouldin={db:.4f}")

    pca = PCA(n_components=2, random_state=random_state)
    Xp  = pca.fit_transform(Xs)
    master['pca_x'] = Xp[:, 0]
    master['pca_y'] = Xp[:, 1]

    master['segmento_nombre'] = master['segmento_id'].map(NOMBRES_SEGMENTO)

    print("\n  Perfil de segmentos:")
    cols_show = [c for c in ['ingreso_mensual_mxn','score_buro','txn_pct_digital',
                              'txn_n_viajes','prod_util_media','conv_noise_pct']
                 if c in master.columns]
    print(master.groupby('segmento_nombre')[cols_show].mean().round(2).to_string())

    # Guardar scaler y modelo
    pickle.dump(scaler, open(os.path.join(_models_dir(), 'scaler_segmentos.pkl'), 'wb'))
    pickle.dump(km5,    open(os.path.join(_models_dir(), 'kmeans.pkl'), 'wb'))
    pickle.dump(cols,   open(os.path.join(_models_dir(), 'cols_clustering.pkl'), 'wb'))

    return master


# ═══════════════════════════════════════════════════════════
# SECCIÓN 4 — PROPENSIÓN POR PRODUCTO
# ═══════════════════════════════════════════════════════════

# Productos a modelar: (columna_target_en_master, nombre_legible)
PRODUCTOS_OBJETIVO = {
    'tiene_tarjeta_credito_hey':        'Tarjeta de Crédito Hey',
    'tiene_inversion_hey':              'Inversión Hey',
    'tiene_seguro_vida':                'Seguro de Vida',
    'tiene_seguro_compras':             'Seguro de Compras',
    'tiene_credito_personal':           'Crédito Personal',
    'tiene_credito_nomina':             'Crédito Nómina',
    'tiene_credito_auto':               'Crédito Auto',
    'tiene_tarjeta_credito_garantizada':'Tarjeta Garantizada',
    'tiene_cuenta_negocios':            'Cuenta Negocios',
}

# Features para propensión — excluyen flags de "tiene_X" para evitar
# fuga de información entre productos hermanos
FEATURES_PROPENSION = [
    # Demográficas
    'edad', 'ingreso_mensual_mxn', 'score_buro', 'antiguedad_dias',
    'nivel_educativo', 'ocupacion',
    'nomina_domiciliada', 'recibe_remesas', 'es_hey_pro', 'usa_hey_shop',
    'satisfaccion_1_10', 'dias_desde_ultimo_login',
    # Transaccional
    'txn_total', 'txn_monto_prom', 'txn_monto_total',
    'txn_pct_digital', 'txn_pct_efectivo', 'txn_pct_inter',
    'txn_n_viajes', 'txn_n_servicios_dig', 'txn_n_supermercado',
    'txn_n_restaurante', 'txn_pct_atipico', 'txn_pct_fallida',
    'txn_n_msi', 'txn_cashback_total', 'txn_volatilidad',
    'txn_frec_por_dia', 'txn_hora_media', 'txn_pct_finde', 'txn_pct_nocturno',
    # Portafolio base
    'prod_n_activos', 'prod_util_media', 'prod_util_max',
    'prod_saldo_total', 'prod_limite_total', 'prod_ratio_problematicos',
    # Conversacional
    'conv_n_total', 'conv_pct_voz', 'conv_turnos_prom',
    'conv_urgencia', 'conv_frustracion', 'conv_reintentos_48h',
    'conv_noise_pct', 'conv_noise_score',
    'intent_credito', 'intent_inversion', 'intent_tarjeta',
    'intent_aclaracion', 'intent_negocio',
    # Segmento
    'segmento_id',
]


def _preparar_X(df, cols):
    """Selecciona columnas, deduplica y convierte tipos."""
    X = df[[c for c in cols if c in df.columns]].copy()
    X = X.loc[:, ~X.columns.duplicated()]

    cols_a_eliminar = []
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            try:
                X[c] = X[c].astype('category').cat.codes
            except Exception:
                cols_a_eliminar.append(c)

    # Eliminar fuera del loop para no modificar mientras iteramos
    if cols_a_eliminar:
        X = X.drop(columns=cols_a_eliminar)

    return X


def entrenar_propension(master):
    """
    Entrena un GBM calibrado por cada producto objetivo.
    Target: ¿el usuario YA tiene el producto activo?

    Por qué este enfoque es válido:
      - Aprende el perfil del usuario típico que adopta cada producto.
      - En inferencia, se aplica SOLO a usuarios que NO tienen el producto.
      - La probabilidad predicha = score de propensión de adopción.
    """
    print("\n" + "═"*60)
    print("  MODELOS DE PROPENSIÓN POR PRODUCTO")
    print("═"*60)

    kfold   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelos = {}
    resultados = []

    for col_target, nombre in PRODUCTOS_OBJETIVO.items():
        if col_target not in master.columns:
            print(f"\n  [{nombre}] SKIP — columna no encontrada")
            continue

        y        = master[col_target].astype(int)
        pos_rate = y.mean()

        if pos_rate < 0.03 or pos_rate > 0.97:
            print(f"\n  [{nombre}] SKIP — desequilibrio extremo ({pos_rate:.1%})")
            continue

        # Excluir el propio target y otros "tiene_X" del feature set
        exclude = {col_target}
        feats   = [f for f in FEATURES_PROPENSION
                   if f in master.columns and f not in exclude]
        X       = _preparar_X(master, feats)
        feats   = X.columns.tolist()   # sincronizar tras limpieza

        # CV para validación
        base_m  = GradientBoostingClassifier(
            n_estimators=120, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
        cv_aucs = cross_val_score(base_m, X, y, cv=kfold,
                                  scoring='roc_auc', n_jobs=1)

        # Modelo calibrado
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        cal_m = CalibratedClassifierCV(base_m, method='isotonic', cv=5)
        cal_m.fit(X_tr, y_tr)
        auc_te = roc_auc_score(y_te, cal_m.predict_proba(X_te)[:, 1])

        # Top features
        fi_list = [cc.estimator.feature_importances_
                   for cc in cal_m.calibrated_classifiers_]
        fi_mean = np.mean(fi_list, axis=0)
        top3    = pd.Series(fi_mean, index=feats).nlargest(3)

        print(f"\n  [{nombre}]")
        print(f"    CV AUC = {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f} | "
              f"Test AUC = {auc_te:.4f} | Positivos = {pos_rate:.1%}")
        print(f"    Top features: {', '.join(top3.index.tolist())}")

        modelos[col_target] = {'model': cal_m, 'feats': feats, 'nombre': nombre}
        resultados.append({
            'producto': nombre, 'col_target': col_target,
            'cv_auc_mean': round(cv_aucs.mean(), 4),
            'cv_auc_std':  round(cv_aucs.std(), 4),
            'test_auc':    round(auc_te, 4),
            'pct_positivos': round(pos_rate, 4),
        })

        # Serializar modelo
        fname = col_target.replace('tiene_', '') + '_model.pkl'
        pickle.dump(modelos[col_target],
                    open(os.path.join(_models_dir(), fname), 'wb'))

    # Guardar resumen de métricas
    pd.DataFrame(resultados).to_csv(_out('metricas_propension.csv'), index=False)
    print(f"\n  → Métricas guardadas en metricas_propension.csv")
    return modelos


# ═══════════════════════════════════════════════════════════
# SECCIÓN 5 — INFERENCIA: PERFIL COMPLETO POR USUARIO
# ═══════════════════════════════════════════════════════════

# Perfil de comunicación por segmento — SOLO preferencias,
# el AGENTE decide qué decir con estos parámetros
PERFIL_COMUNICACION = {
    'PYME Emprendedor': {
        'tono':    'formal',
        'canal':   'texto',
        'horario': '09:00-11:00 y 15:00-17:00',
        'estilo':  'propositivo, basado en datos, sin emojis',
    },
    'Digital Básico': {
        'tono':    'informal',
        'canal':   'texto',
        'horario': '20:00-01:00',
        'estilo':  'directo, corto, con emojis ocasionales',
    },
    'Premium Digital': {
        'tono':    'sofisticado',
        'canal':   'texto',
        'horario': '07:00-09:00',
        'estilo':  'proactivo, anticipar necesidades',
    },
    'Familia Establecida': {
        'tono':    'semi-formal',
        'canal':   'texto',
        'horario': '08:00-10:00 y 18:00-20:00',
        'estilo':  'empático, con beneficio concreto',
    },
    'Ahorrador Cauteloso': {
        'tono':    'simple',
        'canal':   'texto',
        'horario': '10:00-14:00',
        'estilo':  'educativo, sin tecnicismos, sin presión',
    },
}


def inferir_perfiles(master, modelos_propension):
    """
    Genera el perfil completo de cada usuario:
      - Segmento y preferencias de comunicación
      - Score de propensión por producto (solo para productos no contratados)
      - Top-3 productos recomendados
      - Flags de riesgo
      - Score de ruido (para el noise gate del agente)
    """
    print("\n" + "═"*60)
    print("  GENERANDO PERFILES")
    print("═"*60)

    perfiles = master[['user_id','segmento_id','segmento_nombre']].copy()

    # Preferencias de comunicación
    perfiles['tono']    = perfiles['segmento_nombre'].map(
        lambda s: PERFIL_COMUNICACION.get(s, {}).get('tono', 'informal'))
    perfiles['canal']   = perfiles['segmento_nombre'].map(
        lambda s: PERFIL_COMUNICACION.get(s, {}).get('canal', 'texto'))
    perfiles['horario'] = perfiles['segmento_nombre'].map(
        lambda s: PERFIL_COMUNICACION.get(s, {}).get('horario', ''))
    perfiles['estilo_comunicacion'] = perfiles['segmento_nombre'].map(
        lambda s: PERFIL_COMUNICACION.get(s, {}).get('estilo', ''))

    # Ajuste de canal por voz
    if 'conv_pct_voz' in master.columns:
        mask_voz = master['conv_pct_voz'] > 0.30
        perfiles.loc[mask_voz, 'canal'] = 'voz'

    # Scores de propensión por producto
    propension_cols = []
    for col_target, info in modelos_propension.items():
        nombre  = info['nombre']
        model   = info['model']
        feats   = info['feats']
        col_score = f'prop_{col_target.replace("tiene_", "")}'

        X = _preparar_X(master, feats)
        # Rellenar columnas que falten
        for f in feats:
            if f not in X.columns:
                X[f] = 0

        probs = model.predict_proba(X)[:, 1]
        perfiles[col_score] = probs.round(4)
        propension_cols.append((col_score, col_target, nombre))

    # Para usuarios que YA TIENEN el producto → score = 0
    # (no tiene sentido recomendarles lo que ya tienen)
    for col_score, col_target, nombre in propension_cols:
        if col_target in master.columns:
            tiene_mask = master[col_target].astype(bool)
            perfiles.loc[tiene_mask.values, col_score] = 0.0

    # Top-3 productos recomendados (mayor propensión entre los que no tiene)
    score_cols = [c for c, _, _ in propension_cols]
    nombre_map = {c: n for c, _, n in propension_cols}

    def _top3(row):
        scores = {nombre_map[c]: row[c] for c in score_cols if c in row.index}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return [t[0] for t in top if t[1] > 0.10]  # solo si prob > 10%

    perfiles['productos_recomendados'] = perfiles.apply(_top3, axis=1)
    perfiles['producto_top_1'] = perfiles['productos_recomendados'].apply(
        lambda x: x[0] if len(x) > 0 else '')
    perfiles['producto_top_2'] = perfiles['productos_recomendados'].apply(
        lambda x: x[1] if len(x) > 1 else '')
    perfiles['producto_top_3'] = perfiles['productos_recomendados'].apply(
        lambda x: x[2] if len(x) > 2 else '')
    perfiles = perfiles.drop(columns=['productos_recomendados'])

    # Flags de riesgo
    perfiles['flag_riesgo_churn'] = (
        (master['dias_desde_ultimo_login'] > 30) &
        (master['satisfaccion_1_10'] < 6)
    ).astype(int).values

    perfiles['flag_uso_atipico'] = master.get(
        'patron_uso_atipico', pd.Series(False, index=master.index)
    ).astype(int).values

    perfiles['flag_credito_estresado'] = (
        master.get('prod_util_max', pd.Series(0, index=master.index)) > 0.85
    ).astype(int).values

    # Score de ruido para el agente (noise gate)
    if 'conv_noise_pct' in master.columns:
        perfiles['noise_score_usuario'] = master['conv_noise_pct'].round(4).values
    else:
        perfiles['noise_score_usuario'] = 0.0

    # Contexto adicional para el agente
    for col in ['ingreso_mensual_mxn','score_buro','antiguedad_dias',
                'ocupacion','nivel_educativo','satisfaccion_1_10',
                'dias_desde_ultimo_login','es_hey_pro','nomina_domiciliada',
                'conv_n_total','conv_frustracion','conv_urgencia',
                'intent_credito','intent_inversion']:
        if col in master.columns:
            perfiles[col] = master[col].values

    n_total    = len(perfiles)
    n_con_rec  = (perfiles['producto_top_1'] != '').sum()
    n_churn    = perfiles['flag_riesgo_churn'].sum()
    n_estresado = perfiles['flag_credito_estresado'].sum()
    n_ruido    = (perfiles['noise_score_usuario'] > 0.40).sum()

    print(f"\n  Perfiles generados: {n_total:,}")
    print(f"  Con al menos 1 recomendación: {n_con_rec:,} ({n_con_rec/n_total:.0%})")
    print(f"  Flag riesgo churn:            {n_churn:,} ({n_churn/n_total:.0%})")
    print(f"  Flag crédito estresado:       {n_estresado:,} ({n_estresado/n_total:.0%})")
    print(f"  Score ruido > 40%:            {n_ruido:,} ({n_ruido/n_total:.0%})")

    return perfiles


# ═══════════════════════════════════════════════════════════
# SECCIÓN 6 — VISUALIZACIONES
# ═══════════════════════════════════════════════════════════

def plot_propension_heatmap(perfiles, modelos_propension):
    """Heatmap de propensión media por segmento."""
    score_cols  = [f'prop_{k.replace("tiene_","")}' for k in modelos_propension]
    nombre_map  = {f'prop_{k.replace("tiene_","")}': v['nombre']
                   for k, v in modelos_propension.items()}
    cols_ok     = [c for c in score_cols if c in perfiles.columns]
    if not cols_ok: return

    hm = perfiles.groupby('segmento_nombre')[cols_ok].mean()
    hm.columns = [nombre_map.get(c, c) for c in hm.columns]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    import seaborn as sns
    sns.heatmap(hm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Propensión media'})
    ax.set_title('Propensión media por producto y segmento',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(''); ax.set_xlabel('')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(_out('plot_propension_segmento.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → plot_propension_segmento.png")


def plot_top_productos(perfiles):
    """Frecuencia del producto top-1 recomendado."""
    counts = perfiles[perfiles['producto_top_1'] != '']['producto_top_1'].value_counts()
    if counts.empty: return

    fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
    COLORES = ['#378ADD','#1D9E75','#7F77DD','#BA7517','#D85A30',
               '#E24B4A','#5DA88F','#C4924C','#8B7EC8']
    bars = ax.barh(counts.index, counts.values,
                   color=COLORES[:len(counts)], height=0.55, alpha=0.85)
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{v:,} ({v/total:.0%})', va='center', fontsize=9)
    ax.set_title('Producto con mayor propensión por usuario',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Número de usuarios')
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(_out('plot_top_productos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → plot_top_productos.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  HEY BANCO DATATHON 2026 — PIPELINE v4")
    print("  Perfil de usuario + Propensión por producto")
    print("═"*60)

    # 1. Cargar y limpiar
    cl, pr, tx, cv = cargar_y_limpiar()

    # 2. Features + master
    print("\n" + "═"*60 + "\n  FEATURE ENGINEERING\n" + "═"*60)
    master = construir_master(cl, pr, tx, cv)

    # 3. Segmentación
    master = segmentar(master)

    # 4. Modelos de propensión
    modelos_propension = entrenar_propension(master)

    # 5. Perfiles completos
    perfiles = inferir_perfiles(master, modelos_propension)

    # 6. Visualizaciones
    print("\n" + "═"*60 + "\n  VISUALIZACIONES\n" + "═"*60)
    plot_propension_heatmap(perfiles, modelos_propension)
    plot_top_productos(perfiles)

    # 7. Guardar
    print("\n" + "═"*60 + "\n  GUARDANDO OUTPUTS\n" + "═"*60)
    perfiles.to_csv(_out('perfiles_usuarios.csv'), index=False)
    master.to_csv(_out('master_usuarios.csv'), index=False)
    print(f"  → perfiles_usuarios.csv  ({len(perfiles):,} usuarios)")
    print(f"  → master_usuarios.csv")
    print(f"  → outputs/modelos/       ({len(modelos_propension)} modelos .pkl)")

    # Ejemplo de perfil
    print("\n" + "═"*60)
    print("  EJEMPLO — perfil de USR-00001")
    print("═"*60)
    ejemplo = perfiles[perfiles['user_id'] == 'USR-00001']
    if not ejemplo.empty:
        row = ejemplo.iloc[0]
        print(f"  Segmento:         {row['segmento_nombre']}")
        print(f"  Tono:             {row['tono']}")
        print(f"  Canal:            {row['canal']}")
        print(f"  Horario:          {row['horario']}")
        print(f"  Producto top-1:   {row['producto_top_1']}")
        print(f"  Producto top-2:   {row['producto_top_2']}")
        print(f"  Producto top-3:   {row['producto_top_3']}")
        print(f"  Riesgo churn:     {bool(row['flag_riesgo_churn'])}")
        print(f"  Crédito estresado:{bool(row['flag_credito_estresado'])}")
        print(f"  Noise score:      {row['noise_score_usuario']:.0%}")

    print("\n" + "═"*60)
    print("  PIPELINE v4 COMPLETADO")
    print("  Siguiente paso: ejecutar hey_agent_havi.py")
    print("═"*60)
