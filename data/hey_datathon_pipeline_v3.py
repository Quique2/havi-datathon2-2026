"""
=============================================================
HEY BANCO — DATATHON 2026  ·  Pipeline v3
Motor de Inteligencia y Atención Personalizada para Havi

CAMBIOS RESPECTO A v2
─────────────────────
Sección 3c — calcular_noise_features():
  • Tipificación de ruido en 6 categorías (noise_type)
  • Artefacto de voz: doble espacio entre palabras (transcripción STT)
  • Puntuación compuesta separada por tipo

Sección 3 — features_conversaciones():
  • 6 nuevas features de ruido agregadas por usuario:
    conv_noise_pct, conv_noise_max, conv_afirm_neg_pct,
    conv_menu_nav_pct, conv_exploratorio_score, conv_noise_trend

Sección 5 — construir_targets():
  • target_interaccion_ruido: usuario con tendencia a sesiones de ruido

Sección 5 — FEATURES_MODELO y FEATURES_RESPONSE_MODEL:
  • Incluyen todas las nuevas features de ruido

Sección 5 — entrenar_modelos():
  • StratifiedKFold CV (k=5) → media ± std de AUC
  • CalibratedClassifierCV (isotonic) → probabilidades bien calibradas
  • Threshold óptimo por target (maximiza F1)
  • Plots: curvas ROC, feature importance, calibración

Sección 5B — entrenar_modelo_response_type():
  • CV + calibración + Macro-F1 por clase
  • Matriz de confusión guardada
  • Noise gate integrado: si P(ruido) > umbral → fuerza educativo

Sección 6 — generar_perfil_havi():
  • Noise gate en tiempo real (conv_noise_pct + noise_type dominante)
  • Override: bot_explorador → no cross-sell

FIXES v3 (compatibilidad Windows)
──────────────────────────────────
  • Eliminado label_binarize (no se usaba)
  • _resolve / _ensure_output_path / _out unificados para Windows/Linux
  • n_jobs=1 en cross_val_score (evita MemoryError en Windows con joblib)
=============================================================
"""

import os, re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    roc_auc_score, roc_curve, auc,
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter


# ─────────────────────────────────────────────
# PATHS — compatibles con Windows, Mac y Linux
# ─────────────────────────────────────────────

def _resolve(filename):
    """
    Busca el archivo en la misma carpeta que el script.
    Funciona en Windows, Mac y Linux (entorno datathon).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"\n  ❌ No se encontró '{filename}'"
        f"\n     Buscado en: {script_dir}"
        f"\n     Coloca los CSVs en la misma carpeta que el script."
    )

def _ensure_output_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

def _out(filename):
    """Alias de _ensure_output_path — usado en funciones de plot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


PATH_CLIENTES = _resolve('hey_clientes.csv')
PATH_PRODUCTOS = _resolve('hey_productos.csv')
PATH_TRANSACC  = _resolve('hey_transacciones.csv')
PATH_CONVS     = _resolve('dataset_50k_anonymized.csv')


# ═══════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA
# ═══════════════════════════════════════════════════════════

def cargar_datos():
    print("\n" + "═"*60)
    print("  CARGA Y DIAGNÓSTICO")
    print("═"*60)
    clientes  = pd.read_csv(PATH_CLIENTES)
    productos = pd.read_csv(PATH_PRODUCTOS)
    txn       = pd.read_csv(PATH_TRANSACC)
    convs     = pd.read_csv(PATH_CONVS)
    for n, df in [("clientes",clientes),("productos",productos),
                  ("transacciones",txn),("conversaciones",convs)]:
        print(f"  [{n}] {len(df):,} filas · {df.shape[1]} cols · "
              f"{df.isnull().sum().sum():,} nulos")
    return clientes, productos, txn, convs


# ═══════════════════════════════════════════════════════════
# SECCIÓN 2 — LIMPIEZA
# ═══════════════════════════════════════════════════════════

def limpiar_clientes(df):
    print("  [limpiar] clientes...")
    df = df.copy()
    if 'sexo' in df.columns:
        df.rename(columns={'sexo': 'genero'}, inplace=True)
    df['estado'] = df['estado'].fillna('Desconocido')
    df['ciudad'] = df['ciudad'].fillna('Desconocido')
    med = df.groupby('ocupacion')['satisfaccion_1_10'].transform('median')
    df['satisfaccion_1_10'] = df['satisfaccion_1_10'].fillna(med).fillna(7.5)
    for col in ['es_hey_pro','nomina_domiciliada','recibe_remesas',
                'usa_hey_shop','tiene_seguro','patron_uso_atipico']:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': True, 'False': False})
    print(f"    → {len(df):,} registros | nulos: {df.isnull().sum().sum()}")
    return df


def limpiar_productos(df):
    print("  [limpiar] productos...")
    df = df.copy()
    df['fecha_apertura']          = pd.to_datetime(df['fecha_apertura'], errors='coerce')
    df['fecha_ultimo_movimiento'] = pd.to_datetime(df['fecha_ultimo_movimiento'], errors='coerce')
    mask = df['utilizacion_pct'].notna() & ((df['utilizacion_pct'] < 0) | (df['utilizacion_pct'] > 1))
    df.loc[mask, 'utilizacion_pct'] = np.nan
    print(f"    → {len(df):,} registros | nulos estructurales aceptados")
    return df


def limpiar_transacciones(df):
    print("  [limpiar] transacciones...")
    df = df.copy()
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
    n = len(df)
    df = df.sort_values('fecha_hora').drop_duplicates(subset='transaccion_id', keep='first')
    print(f"    → {n - len(df)} duplicados eliminados (reenvíos)")
    df['ciudad_transaccion'] = df['ciudad_transaccion'].fillna('Desconocida')
    df['descripcion_libre']  = df['descripcion_libre'].fillna('')
    for col in ['es_internacional', 'patron_uso_atipico']:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': True, 'False': False})
    print(f"    → {len(df):,} registros limpios")
    return df


def limpiar_conversaciones(df):
    print("  [limpiar] conversaciones...")
    df = df.copy()
    n = len(df)
    df = df.dropna(subset=['input'])
    print(f"    → {n - len(df)} filas con input=NaN eliminadas")
    df['date']       = pd.to_datetime(df['date'], errors='coerce')
    df['es_voz']     = (df['channel_source'] == 2).astype(int)
    df['input_len']  = df['input'].str.len()
    df['output_len'] = df['output'].str.len()
    print(f"    → {len(df):,} registros limpios")
    return df


# ═══════════════════════════════════════════════════════════
# SECCIÓN 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

INTENT_RULES = {
    'credito':       ['crédito','credito','préstamo','prestamo','financiamiento',
                      'dinero prestado','cuánto me prestan'],
    'transferencia': ['transferencia','transf','spei','clabe','enviar','envío','envio'],
    'tarjeta':       ['tarjeta','bloquear','robo','perdida','cancelar','reposición'],
    'beneficios':    ['cashback','puntos','recompensa','hey pro','anualidad','coins'],
    'inversion':     ['inversión','inversion','rendimiento','gat','ahorro','plazo'],
    'aclaracion':    ['cobro','cargo','disputa','aclaración','aclaracion','reembolso',
                      'no reconozco'],
    'pago_servicio': ['pago','pagar','servicio','luz','agua','teléfono','impuesto','cfdi'],
    'seguridad':     ['contraseña','nip','pin','acceso','login','dispositivo','token','bloqueado'],
    'seguro':        ['seguro','vida','protección','proteccion'],
    'negocio':       ['negocio','empresa','terminal','pos','comercio','factura','rfc'],
}

PALABRAS_FINANCIERAS = [
    'tarjeta','cuenta','saldo','transferencia','credito','crédito',
    'pago','banco','inversión','inversion','spei','cajero','cashback',
    'havi','hey','oxxo','dinero','cobro','cargo','préstamo','prestamo',
    'nip','token','clabe','depósito','deposito','retiro','compra',
]

PALABRAS_URGENCIA    = ['urgente','urge','ya','ahora','inmediato','rápido','rapido','hoy']
PALABRAS_FRUSTRACION = ['error','no me deja','no funciona','problema','falla','bloqueado',
                        'no puedo','imposible','molesto','molesta','queja','mal servicio']

def _intent(texto):
    if pd.isna(texto) or not str(texto).strip(): return 'otro'
    t = str(texto).lower()
    for intent, palabras in INTENT_RULES.items():
        if any(p in t for p in palabras): return intent
    return 'otro'

def _urgencia(texto):
    if pd.isna(texto): return 0
    return int(any(p in str(texto).lower() for p in PALABRAS_URGENCIA))

def _frustracion(texto):
    if pd.isna(texto): return 0
    return int(any(p in str(texto).lower() for p in PALABRAS_FRUSTRACION))


# ── 3c — Detector de ruido ───────────────────────────────
#
# DECISIÓN DE DISEÑO: los mensajes de ruido NO se eliminan.
# Se convierten en señal para que el GBM aprenda cuándo Havi
# debe responder en modo educativo en vez de cross-sell.
#
# Tipos de ruido identificados en el corpus real:
#
# | Tipo               | Ejemplo                        | Frecuencia |
# |--------------------|--------------------------------|------------|
# | artefacto_voz      | " palabra  palabra  palabra"   | <1%        |
# | menu_navegacion    | "A" "B" "1" "2"                | ~0.9%      |
# | afirmacion_negacion| "si" "no" "ok" "gracias"       | ~3.3%      |
# | exploracion_corta  | textos ≤2 tokens sin fin.      | ~9.9%      |
# | broma_offtopic     | "cuéntame un chiste", "eres IA"| ~0.02%     |
# | solo_simbolos      | "???" "!!!"                    | ~0.06%     |
# | on_topic           | todo lo demás                  | ~85.8%     |

_AFIRM_NEG = frozenset([
    'si','sí','no','ok','vale','gracias','perfecto','listo',
    'entendido','claro','de acuerdo','por favor','adelante',
    'continuar','siguiente','no gracias','np',
])
_MENU_NAV = frozenset(['a','b','c','d','1','2','3','4','a)','b)','c)','d)','1.','2.','3.'])
_BROMAS   = ['chiste','cuéntame un chiste','cuentame un chiste','te amo','te quiero',
             'quien eres','quién eres','eres humano','eres ia','eres real',
             'gaming','cántame','cantame','me ayudas a ligar']

# Pesos para el noise_score compuesto (0–1, soft signal)
_W = {
    'artefacto_voz':       0.20,
    'menu_navegacion':     0.55,
    'afirmacion_negacion': 0.40,
    'exploracion_corta':   0.65,
    'broma_offtopic':      0.85,
    'solo_simbolos':       0.90,
    'on_topic':            0.00,
}


def calcular_noise_features(texto):
    """
    Calcula señales de ruido/relevancia a nivel de turno individual.

    Retorna un pd.Series con:
      noise_type           — categoría dominante del ruido
      noise_score_msg      — puntuación suave [0, 1]
      is_financial_msg     — contiene términos del dominio financiero/Hey
      is_voz_artefacto     — patrón de transcripción STT (dobles espacios)
      joke_score_msg       — señal de broma/off-topic
      low_intent_score_msg — señal de baja intención financiera
    """
    if pd.isna(texto):
        return pd.Series({
            'noise_type': 'nulo', 'noise_score_msg': 1.0,
            'is_financial_msg': 0, 'is_voz_artefacto': 0,
            'joke_score_msg': 0, 'low_intent_score_msg': 1,
        })

    t_raw  = str(texto)
    t_low  = t_raw.lower().strip()
    tokens = re.findall(r'\w+', t_low, flags=re.UNICODE)
    n_tok  = len(tokens)

    # 1. Artefacto de voz — patrón STT: doble espacio entre palabras
    is_voz_art  = int(bool(re.search(r'\w{3,}\s{2,}\w{3,}\s{2,}\w{3,}', t_raw)))

    # 2. Solo símbolos / emojis
    is_only_sym = int(len(t_low) > 0 and not bool(re.search(r'[a-záéíóúñ0-9]', t_low, re.I)))

    # 3. Broma / off-topic
    is_broma    = int(any(b in t_low for b in _BROMAS))

    # 4. Navegación de menú
    is_menu     = int(t_low in _MENU_NAV)

    # 5. Afirmación / negación pura
    is_afirm    = int(t_low in _AFIRM_NEG)

    # 6. Exploración corta — word boundary para evitar falsos positivos
    has_fin     = int(any(re.search(r'\b' + re.escape(fw) + r'\b', t_low)
                          for fw in PALABRAS_FINANCIERAS))
    is_explor   = int(n_tok <= 2 and not has_fin and not is_afirm and not is_menu)

    # Clasificación por prioridad
    if   is_only_sym: noise_type = 'solo_simbolos'
    elif is_broma:    noise_type = 'broma_offtopic'
    elif is_voz_art:  noise_type = 'artefacto_voz'
    elif is_menu:     noise_type = 'menu_navegacion'
    elif is_afirm:    noise_type = 'afirmacion_negacion'
    elif is_explor:   noise_type = 'exploracion_corta'
    else:             noise_type = 'on_topic'

    # Score compuesto — atenuar si hay términos financieros
    raw_score = _W[noise_type]
    if has_fin and raw_score > 0:
        raw_score *= 0.45
    noise_score = round(min(1.0, raw_score), 4)

    return pd.Series({
        'noise_type':           noise_type,
        'noise_score_msg':      noise_score,
        'is_financial_msg':     has_fin,
        'is_voz_artefacto':     is_voz_art,
        'joke_score_msg':       is_broma,
        'low_intent_score_msg': int(noise_type in ('exploracion_corta','afirmacion_negacion',
                                                    'menu_navegacion','broma_offtopic')),
    })


def _agregar_noise_features_usuario(convs):
    """
    Features de ruido agregadas a nivel de usuario.

    Features:
      conv_noise_pct         % turnos con noise_score > 0.35
      conv_noise_max         max noise_score del usuario
      conv_afirm_neg_pct     % turnos afirmacion/negacion
      conv_menu_nav_pct      % turnos navegacion de menú
      conv_broma_pct         % turnos broma/offtopic
      conv_voz_artefacto_pct % turnos con artefacto STT
      conv_exploratorio_score conv_noise_pct × conv_noise_max
      conv_noise_trend        diferencia noise_score 1a vs 2a mitad
    """
    NOISE_THRESHOLD = 0.35

    agg = convs.groupby('user_id').agg(
        conv_noise_pct         = ('noise_score_msg',
                                   lambda x: (x > NOISE_THRESHOLD).mean()),
        conv_noise_max         = ('noise_score_msg', 'max'),
        conv_afirm_neg_pct     = ('noise_type',
                                   lambda x: (x == 'afirmacion_negacion').mean()),
        conv_menu_nav_pct      = ('noise_type',
                                   lambda x: (x == 'menu_navegacion').mean()),
        conv_broma_pct         = ('noise_type',
                                   lambda x: (x == 'broma_offtopic').mean()),
        conv_voz_artefacto_pct = ('is_voz_artefacto', 'mean'),
    ).reset_index()

    agg['conv_exploratorio_score'] = (
        agg['conv_noise_pct'] * agg['conv_noise_max']
    ).round(4)

    # Trend: ¿el usuario se vuelve más o menos ruidoso con el tiempo?
    convs_sorted = convs.sort_values(['user_id', 'date'])

    def _trend(grp):
        n = len(grp)
        if n < 4: return 0.0
        mid     = n // 2
        primera = grp.iloc[:mid]['noise_score_msg'].mean()
        segunda = grp.iloc[mid:]['noise_score_msg'].mean()
        return round(segunda - primera, 4)

    trend = convs_sorted.groupby('user_id').apply(_trend).reset_index()
    trend.columns = ['user_id', 'conv_noise_trend']
    agg = agg.merge(trend, on='user_id', how='left')

    return agg


def features_conversaciones(convs):
    print("  [features] conversaciones...")
    convs = convs.copy()

    convs['intent']      = convs['input'].apply(_intent)
    convs['urgencia']    = convs['input'].apply(_urgencia)
    convs['frustracion'] = convs['input'].apply(_frustracion)

    noise_cols = convs['input'].apply(calcular_noise_features)
    convs = pd.concat([convs, noise_cols], axis=1)

    tc    = convs.groupby('conv_id')['input'].count().reset_index(name='turnos_conv')
    convs = convs.merge(tc, on='conv_id', how='left')

    agg = convs.groupby('user_id').agg(
        conv_n_total        = ('conv_id', 'nunique'),
        conv_turnos_total   = ('input', 'count'),
        conv_turnos_prom    = ('turnos_conv', 'mean'),
        conv_turnos_max     = ('turnos_conv', 'max'),
        conv_usa_voz        = ('es_voz', 'max'),
        conv_pct_voz        = ('es_voz', 'mean'),
        conv_input_len_med  = ('input_len', 'mean'),
        conv_input_len_max  = ('input_len', 'max'),
        conv_urgencia       = ('urgencia', 'sum'),
        conv_frustracion    = ('frustracion', 'sum'),
        conv_noise_score    = ('noise_score_msg', 'mean'),
        conv_joke_score     = ('joke_score_msg', 'mean'),
        conv_low_intent     = ('low_intent_score_msg', 'mean'),
        conv_relevancia_fin = ('is_financial_msg', 'mean'),
        intent_credito      = ('intent', lambda x: (x == 'credito').sum()),
        intent_transferencia= ('intent', lambda x: (x == 'transferencia').sum()),
        intent_tarjeta      = ('intent', lambda x: (x == 'tarjeta').sum()),
        intent_beneficios   = ('intent', lambda x: (x == 'beneficios').sum()),
        intent_inversion    = ('intent', lambda x: (x == 'inversion').sum()),
        intent_aclaracion   = ('intent', lambda x: (x == 'aclaracion').sum()),
        intent_seguridad    = ('intent', lambda x: (x == 'seguridad').sum()),
        intent_negocio      = ('intent', lambda x: (x == 'negocio').sum()),
    ).reset_index()

    # Reintentos < 48h
    convs_s = convs.sort_values(['user_id', 'date'])
    convs_s['prev_date'] = convs_s.groupby('user_id')['date'].shift(1)
    convs_s['hrs_gap']   = (convs_s['date'] - convs_s['prev_date']).dt.total_seconds() / 3600
    reintentos = convs_s[convs_s['hrs_gap'] < 48].groupby('user_id').size()
    agg = agg.merge(reintentos.rename('conv_reintentos_48h'), on='user_id', how='left')
    agg['conv_reintentos_48h'] = agg['conv_reintentos_48h'].fillna(0)

    # Noise features por usuario
    noise_user = _agregar_noise_features_usuario(convs)
    agg = agg.merge(noise_user, on='user_id', how='left')

    print(f"    → {agg.shape[1]-1} features para {len(agg):,} usuarios")
    return agg


def features_transacciones(txn):
    print("  [features] transacciones...")
    agg = txn.groupby('user_id').agg(
        txn_total          = ('transaccion_id', 'count'),
        txn_monto_total    = ('monto', 'sum'),
        txn_monto_prom     = ('monto', 'mean'),
        txn_monto_mediana  = ('monto', 'median'),
        txn_monto_max      = ('monto', 'max'),
        txn_monto_std      = ('monto', 'std'),
        txn_n_categorias   = ('categoria_mcc', 'nunique'),
        txn_n_canales      = ('canal', 'nunique'),
        txn_n_dias_activo  = ('fecha_hora', lambda x: x.dt.date.nunique()),
        txn_pct_digital    = ('canal', lambda x: x.isin(['app_ios','app_android','app_huawei','codi']).mean()),
        txn_pct_efectivo   = ('canal', lambda x: x.isin(['cajero_banregio','cajero_externo','oxxo','farmacia_ahorro']).mean()),
        txn_pct_pos        = ('canal', lambda x: (x == 'pos_fisico').mean()),
        txn_pct_inter      = ('es_internacional', 'mean'),
        txn_n_inter        = ('es_internacional', 'sum'),
        txn_pct_atipico    = ('patron_uso_atipico', 'mean'),
        txn_pct_fallida    = ('estatus', lambda x: (x == 'no_procesada').mean()),
        txn_n_intentos_ext = ('intento_numero', lambda x: (x > 1).sum()),
        txn_n_viajes       = ('categoria_mcc', lambda x: (x == 'viajes').sum()),
        txn_n_servicios_dig= ('categoria_mcc', lambda x: (x == 'servicios_digitales').sum()),
        txn_n_supermercado = ('categoria_mcc', lambda x: (x == 'supermercado').sum()),
        txn_n_restaurante  = ('categoria_mcc', lambda x: (x == 'restaurante').sum()),
        txn_usa_inversion  = ('tipo_operacion', lambda x: x.isin(['abono_inversion','retiro_inversion']).any()),
        txn_n_msi          = ('meses_diferidos', lambda x: x.notna().sum()),
        txn_cashback_total = ('cashback_generado', 'sum'),
        txn_hora_media     = ('hora_del_dia', 'mean'),
        txn_pct_finde      = ('dia_semana', lambda x: x.isin(['Saturday','Sunday']).mean()),
        txn_pct_nocturno   = ('hora_del_dia', lambda x: ((x >= 22) | (x <= 5)).mean()),
    ).reset_index()
    agg['txn_volatilidad']  = agg['txn_monto_std'] / (agg['txn_monto_prom'] + 1)
    agg['txn_frec_por_dia'] = agg['txn_total'] / (agg['txn_n_dias_activo'] + 1)
    print(f"    → {agg.shape[1]-1} features para {len(agg):,} usuarios")
    return agg


def features_productos(prods):
    print("  [features] productos...")
    agg = prods.groupby('user_id').agg(
        prod_n_total          = ('producto_id', 'count'),
        prod_n_activos        = ('estatus', lambda x: (x == 'activo').sum()),
        prod_n_cancelados     = ('estatus', lambda x: (x == 'cancelado').sum()),
        prod_n_revision       = ('estatus', lambda x: (x == 'revision_de_pagos').sum()),
        prod_tiene_credito    = ('tipo_producto', lambda x: x.isin(['tarjeta_credito_hey','credito_personal','credito_auto','credito_nomina','tarjeta_credito_garantizada']).any()),
        prod_tiene_inversion  = ('tipo_producto', lambda x: (x == 'inversion_hey').any()),
        prod_tiene_negocio    = ('tipo_producto', lambda x: x.isin(['cuenta_negocios','tarjeta_credito_negocios']).any()),
        prod_tiene_seguro     = ('tipo_producto', lambda x: x.isin(['seguro_vida','seguro_compras']).any()),
        prod_tiene_garantizada= ('tipo_producto', lambda x: (x == 'tarjeta_credito_garantizada').any()),
        prod_util_media       = ('utilizacion_pct', 'mean'),
        prod_util_max         = ('utilizacion_pct', 'max'),
        prod_saldo_total      = ('saldo_actual', 'sum'),
        prod_limite_total     = ('limite_credito', 'sum'),
    ).reset_index()
    agg['prod_util_media']          = agg['prod_util_media'].fillna(0)
    agg['prod_util_max']            = agg['prod_util_max'].fillna(0)
    agg['prod_ratio_problematicos'] = (
        agg['prod_n_cancelados'] + agg['prod_n_revision']
    ) / (agg['prod_n_total'] + 1)
    print(f"    → {agg.shape[1]-1} features para {len(agg):,} usuarios")
    return agg


def construir_master(clientes, productos, txn, convs):
    print("\n  [master] uniendo tablas...")
    f_conv = features_conversaciones(convs)
    f_txn  = features_transacciones(txn)
    f_prod = features_productos(productos)
    master = clientes.copy()
    master = master.merge(f_txn,  on='user_id', how='left')
    master = master.merge(f_prod, on='user_id', how='left')
    master = master.merge(f_conv, on='user_id', how='left')
    master = master.fillna(0)
    print(f"  → Master: {master.shape[0]:,} usuarios × {master.shape[1]} columnas")
    return master


# ═══════════════════════════════════════════════════════════
# SECCIÓN 4 — SEGMENTACIÓN
# ═══════════════════════════════════════════════════════════

FEATURES_CLUSTERING = [
    'edad','ingreso_mensual_mxn','score_buro','antiguedad_dias','num_productos_activos',
    'txn_monto_prom','txn_pct_digital','txn_pct_efectivo','txn_pct_inter',
    'txn_n_viajes','txn_n_servicios_dig','txn_volatilidad','txn_frec_por_dia',
    'txn_pct_nocturno','txn_pct_atipico','txn_n_msi','txn_n_intentos_ext',
    'prod_n_activos','prod_util_media','prod_util_max',
    'prod_tiene_credito','prod_tiene_inversion','prod_tiene_negocio',
    'prod_ratio_problematicos',
    'conv_n_total','conv_pct_voz','conv_input_len_med',
    'conv_urgencia','conv_frustracion',
    'conv_noise_score','conv_noise_pct','conv_exploratorio_score',
    'conv_afirm_neg_pct','conv_noise_trend',
    'intent_credito','intent_inversion','intent_negocio','intent_aclaracion',
]


def segmentar_usuarios(master, n_clusters=5, random_state=42):
    print("\n" + "═"*60)
    print("  SEGMENTACIÓN NO SUPERVISADA")
    print("═"*60)
    cols  = [c for c in FEATURES_CLUSTERING if c in master.columns]
    X     = master[cols].copy()
    scaler = StandardScaler()
    X_sc  = scaler.fit_transform(X)

    print("\n  Evaluando k óptimo...")
    inercias, silhouettes = [], []
    for k in range(2, 9):
        km   = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labs = km.fit_predict(X_sc)
        inercias.append(km.inertia_)
        sil = silhouette_score(X_sc, labs, sample_size=3000)
        silhouettes.append(sil)
        print(f"    k={k}  inercia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    km5 = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    master['segmento_id'] = km5.fit_predict(X_sc)
    sil5 = silhouette_score(X_sc, master['segmento_id'], sample_size=5000)
    db   = davies_bouldin_score(X_sc, master['segmento_id'])
    print(f"\n  k={n_clusters} → Silhouette={sil5:.4f} | Davies-Bouldin={db:.4f}")
    print(f"  Nota: silhouette < 0.30 es esperado en datos sintéticos homogéneos")

    pca = PCA(n_components=2, random_state=random_state)
    Xp  = pca.fit_transform(X_sc)
    master['pca_x'] = Xp[:, 0]
    master['pca_y'] = Xp[:, 1]
    print(f"  Varianza explicada PCA 2D: {pca.explained_variance_ratio_.sum():.2%}")
    return master, scaler, km5, cols, inercias, silhouettes


def nombrar_segmentos(master):
    cols_ok = [c for c in ['ingreso_mensual_mxn','score_buro','txn_pct_digital',
                            'txn_pct_efectivo','txn_n_viajes','prod_tiene_negocio',
                            'conv_pct_voz','conv_noise_pct'] if c in master.columns]
    perfil = master.groupby('segmento_id')[cols_ok].mean()
    print("\n  Perfil de segmentos:")
    print(perfil.round(2).to_string())
    NOMBRES = {
        0: 'PYME Emprendedor',
        1: 'Digital Básico',
        2: 'Premium Digital',
        3: 'Familia Establecida',
        4: 'Ahorrador Cauteloso',
    }
    master['segmento_nombre'] = master['segmento_id'].map(NOMBRES)
    return master


# ═══════════════════════════════════════════════════════════
# SECCIÓN 5 — MODELOS PREDICTIVOS
# ═══════════════════════════════════════════════════════════

def construir_targets(master):
    """
    Targets binarios para los clasificadores auxiliares.

    target_interaccion_ruido (NUEVO v3):
      Usuarios cuyas sesiones son mayoritariamente ruido (>40%).
      Havi los trata en modo educativo, sin cross-sell.
    """
    df = master.copy()

    df['target_tono_formal'] = (
        df['ocupacion'].isin(['Empresario', 'Jubilado']) |
        (df['ingreso_mensual_mxn'] > 37_000) |
        (df['nivel_educativo'] == 'Posgrado')
    ).astype(int)

    df['target_prefiere_voz'] = (df['conv_pct_voz'] > 0.10).astype(int)

    df['target_propension_credito'] = (
        (df['intent_credito'] > 0) &
        (~df['prod_tiene_credito'].astype(bool))
    ).astype(int)

    df['target_riesgo_churn'] = (
        (df['dias_desde_ultimo_login'] > 30) &
        (df['satisfaccion_1_10'] < 6)
    ).astype(int)

    df['target_engagement_alto'] = (
        (df['conv_n_total'] > 2) &
        (df['txn_frec_por_dia'] > 1)
    ).astype(int)

    # NUEVO v3 — umbral 0.40: >40% de sus mensajes son ruido
    df['target_interaccion_ruido'] = (df['conv_noise_pct'] > 0.40).astype(int)

    print("\n  Distribución de targets:")
    for t in ['target_tono_formal','target_prefiere_voz','target_propension_credito',
              'target_riesgo_churn','target_engagement_alto','target_interaccion_ruido']:
        if t in df.columns:
            pct = df[t].mean()
            print(f"    {t}: {pct:.1%} positivos ({int(pct*len(df)):,})")
    return df


FEATURES_MODELO = [
    'edad','ingreso_mensual_mxn','score_buro','antiguedad_dias',
    'num_productos_activos','satisfaccion_1_10',
    'txn_monto_prom','txn_pct_digital','txn_pct_efectivo',
    'txn_n_viajes','txn_volatilidad','txn_frec_por_dia',
    'txn_pct_atipico','txn_pct_fallida','txn_n_intentos_ext',
    'prod_n_activos','prod_util_media','prod_util_max',
    'prod_tiene_credito','prod_tiene_inversion','prod_tiene_negocio',
    'prod_ratio_problematicos',
    'conv_n_total','conv_pct_voz','conv_input_len_med',
    'conv_urgencia','conv_frustracion','conv_reintentos_48h',
    # Noise features (NUEVO v3)
    'conv_noise_score','conv_noise_pct','conv_noise_max',
    'conv_exploratorio_score','conv_noise_trend',
    'conv_afirm_neg_pct','conv_menu_nav_pct',
    'conv_broma_pct','conv_voz_artefacto_pct',
    'conv_joke_score','conv_low_intent','conv_relevancia_fin',
    'intent_credito','intent_inversion','intent_negocio',
    'intent_aclaracion','intent_tarjeta','intent_seguridad',
    'segmento_id',
]

TARGETS_BINARIOS = [
    'target_tono_formal',
    'target_prefiere_voz',
    'target_propension_credito',
    'target_riesgo_churn',
    'target_engagement_alto',
    'target_interaccion_ruido',   # NUEVO v3
]


def _preparar_X(X):
    """
    Limpia un DataFrame de features antes de pasarlo a sklearn:
      1. Elimina columnas duplicadas (causa del AttributeError en Windows)
      2. Convierte booleanos y object-bool a int
    """
    X = X.loc[:, ~X.columns.duplicated()].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            try:
                X[c] = X[c].astype(int)
            except (ValueError, TypeError):
                X = X.drop(columns=[c])
    return X


def _gbm_base(n_estimators=120, max_depth=4, lr=0.05):
    return GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, subsample=0.8, random_state=42,
    )


def _threshold_optimo_f1(y_true, y_prob):
    """Encuentra el umbral que maximiza F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_prob >= t).astype(int)
        f1    = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2), round(best_f1, 4)


def entrenar_modelos(master, plot=True):
    """
    GBM calibrado por cada target binario.

    • StratifiedKFold (k=5): AUC media ± std
    • CalibratedClassifierCV (isotonic): probabilidades calibradas
    • Threshold óptimo por F1
    • n_jobs=1 — evita MemoryError de joblib en Windows
    """
    print("\n" + "═"*60)
    print("  MODELOS BINARIOS (v3 — CV + calibración)")
    print("═"*60)

    cols     = [c for c in FEATURES_MODELO if c in master.columns]
    kfold    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelos  = {}
    roc_data = {}

    for target in TARGETS_BINARIOS:
        if target not in master.columns:
            continue

        X = master[cols].copy()
        X = _preparar_X(X)
        cols = X.columns.tolist()   # sincronizar tras deduplicar/limpiar
        y = master[target]

        pos_rate = y.mean()
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"\n  [{target}] OMITIDO — desequilibrio extremo ({pos_rate:.2%})")
            continue

        # CV en estimador base — n_jobs=1 para evitar MemoryError en Windows
        base_model = _gbm_base()
        cv_aucs = cross_val_score(
            base_model, X, y, cv=kfold, scoring='roc_auc', n_jobs=1
        )

        # Modelo calibrado sobre split 80/20
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        cal_model = CalibratedClassifierCV(_gbm_base(), method='isotonic', cv=5)
        cal_model.fit(X_tr, y_tr)
        y_prob_te = cal_model.predict_proba(X_te)[:, 1]

        auc_te    = roc_auc_score(y_te, y_prob_te)
        brier     = brier_score_loss(y_te, y_prob_te)
        thr, f1_t = _threshold_optimo_f1(y_te, y_prob_te)
        y_pred_t  = (y_prob_te >= thr).astype(int)
        f1_macro  = f1_score(y_te, y_pred_t, average='macro', zero_division=0)

        # Feature importance media de los calibradores internos
        fi_list = [cc.estimator.feature_importances_
                   for cc in cal_model.calibrated_classifiers_]
        fi_mean = np.mean(fi_list, axis=0)
        top5    = pd.Series(fi_mean, index=cols).nlargest(5)

        print(f"\n  [{target}]")
        print(f"    CV AUC    : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
        print(f"    Test AUC  : {auc_te:.4f}")
        print(f"    Brier     : {brier:.4f}  (menor = mejor calibración)")
        print(f"    Threshold : {thr} → F1 test = {f1_t:.4f} | Macro-F1 = {f1_macro:.4f}")
        print(f"    Top-5 features:")
        for feat, imp in top5.items():
            print(f"      {feat:<40} {imp:.4f}")

        modelos[target] = {
            'model': cal_model, 'threshold': thr,
            'cv_aucs': cv_aucs, 'auc_test': auc_te, 'brier': brier,
            'top_features': top5, 'cols': cols,
        }

        fpr, tpr, _ = roc_curve(y_te, y_prob_te)
        roc_data[target] = (fpr, tpr, auc_te, thr)

    if plot and roc_data:
        _plot_roc_curves(roc_data)
        _plot_feature_importances(modelos)
        _plot_calibracion(master, modelos)

    return modelos


def _plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    COLORS = ['#378ADD','#1D9E75','#7F77DD','#BA7517','#D85A30','#E24B4A']
    etiquetas = {
        'target_tono_formal':        'Tono formal',
        'target_prefiere_voz':       'Prefiere voz',
        'target_propension_credito': 'Propensión crédito',
        'target_riesgo_churn':       'Riesgo churn',
        'target_engagement_alto':    'Engagement alto',
        'target_interaccion_ruido':  'Interacción ruido ★',
    }
    for i, (target, (fpr, tpr, auc_v, thr)) in enumerate(roc_data.items()):
        lbl = etiquetas.get(target, target)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=1.5,
                label=f'{lbl} (AUC={auc_v:.3f})')
    ax.plot([0,1],[0,1],'--', color='#888', lw=0.8, label='Aleatorio')
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos')
    ax.set_title('Curvas ROC — modelos binarios Havi v3', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(_out('plot_roc_binarios.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Curvas ROC guardadas")


def _plot_feature_importances(modelos):
    n = len(modelos)
    if n == 0: return
    cols_g = min(3, n)
    rows_g = (n + cols_g - 1) // cols_g
    fig, axes = plt.subplots(rows_g, cols_g,
                             figsize=(5*cols_g, 3.5*rows_g), facecolor='white')
    axes = np.array(axes).flatten() if n > 1 else [axes]
    etiquetas = {
        'target_tono_formal':        'Tono formal',
        'target_prefiere_voz':       'Prefiere voz',
        'target_propension_credito': 'Propensión crédito',
        'target_riesgo_churn':       'Riesgo churn',
        'target_engagement_alto':    'Engagement alto',
        'target_interaccion_ruido':  'Interacción ruido ★',
    }
    COLORS = ['#378ADD','#1D9E75','#7F77DD','#BA7517','#D85A30','#E24B4A']
    for i, (target, info) in enumerate(modelos.items()):
        ax    = axes[i]
        top   = info['top_features']
        color = COLORS[i % len(COLORS)]
        ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.85, height=0.6)
        ax.set_title(etiquetas.get(target, target), fontsize=9, fontweight='bold')
        ax.set_xlabel('Importancia', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Feature Importance por modelo', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_out('plot_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Feature importance guardada")


def _plot_calibracion(master, modelos):
    """
    Curva de calibración — especialmente importante para
    target_interaccion_ruido porque el noise gate usa probabilidades.
    """
    targets_plot = [t for t in ['target_interaccion_ruido',
                                'target_propension_credito',
                                'target_riesgo_churn'] if t in modelos]
    if not targets_plot: return

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    COLORS  = ['#D85A30','#378ADD','#BA7517']
    etiquetas = {
        'target_interaccion_ruido':  'Interacción ruido ★',
        'target_propension_credito': 'Propensión crédito',
        'target_riesgo_churn':       'Riesgo churn',
    }
    for i, target in enumerate(targets_plot):
        info  = modelos[target]
        model = info['model']
        cols  = info['cols']
        X = master[[c for c in cols if c in master.columns]].copy()
        X = _preparar_X(X)
        y     = master[target]
        probs = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=8, strategy='quantile')
        ax.plot(mean_pred, frac_pos, 'o-', color=COLORS[i], lw=1.5,
                label=etiquetas.get(target, target), markersize=5)

    ax.plot([0,1],[0,1],'--', color='#888', lw=0.8, label='Calibración perfecta')
    ax.set_xlabel('Probabilidad predicha media')
    ax.set_ylabel('Fracción de positivos reales')
    ax.set_title(
        'Curva de calibración — modelos clave\n'
        '(★ noise gate debe estar bien calibrado)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(_out('plot_calibracion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Calibración guardada")


# ── Target multicategoría ─────────────────────────────────

FEATURES_RESPONSE_MODEL = FEATURES_MODELO + [
    'txn_cashback_total','txn_n_inter',
    'intent_aclaracion','intent_seguridad',
    'conv_reintentos_48h',
    'target_riesgo_churn','target_propension_credito',
    'target_prefiere_voz','target_tono_formal',
    'target_interaccion_ruido',    # NUEVO v3
]


def construir_response_type_optimo(master):
    """
    Prioridad: soporte > retencion > alerta > ruido > cross_sell > educativo
    NUEVO v3: ruido explícito antes de cross_sell.
    """
    df = master.copy()

    soporte    = ((df.get('conv_frustracion', 0) >= 1) |
                  (df.get('intent_aclaracion', 0) > 0) |
                  (df.get('intent_seguridad', 0) > 0) |
                  (df.get('txn_pct_fallida', 0) > 0.12) |
                  (df.get('conv_reintentos_48h', 0) >= 2))
    retencion  = ((df.get('dias_desde_ultimo_login', 0) > 30) &
                  (df.get('satisfaccion_1_10', 10) < 7))
    alerta     = ((df.get('txn_pct_atipico', 0) > 0.08) |
                  (df.get('txn_n_intentos_ext', 0) >= 3) |
                  ((df.get('txn_pct_inter', 0) > 0.05) & (df.get('txn_n_inter', 0) >= 2)))
    ruido      = (df.get('target_interaccion_ruido', 0) == 1)
    cross_sell = (
        ((df.get('intent_credito', 0) > 0) & (~df.get('prod_tiene_credito', False).astype(bool))) |
        ((df.get('intent_inversion', 0) > 0) & (~df.get('prod_tiene_inversion', False).astype(bool))) |
        ((df.get('intent_negocio', 0) > 0) & (~df.get('prod_tiene_negocio', False).astype(bool))) |
        ((df.get('txn_cashback_total', 0) > 0) & (~df.get('es_hey_pro', False).astype(bool)))
    )
    educativo  = ((df.get('conv_low_intent', 0) > 0.35) |
                  (df.get('conv_noise_score', 0) > 0.25) |
                  ((df.get('score_buro', 850) < 620) & (~df.get('prod_tiene_credito', False).astype(bool))))

    df['response_type_optimo'] = np.select(
        [soporte, retencion, alerta, ruido, cross_sell, educativo],
        ['soporte','retencion','alerta_preventiva','educativo','cross_sell','educativo'],
        default='educativo'
    )
    print("\n  Distribución response_type_optimo:")
    dist = df['response_type_optimo'].value_counts(normalize=True).mul(100).round(1)
    for k, v in dist.items():
        print(f"    {k:<25} {v}%")
    return df


def entrenar_modelo_response_type(master):
    """
    GBM multiclase calibrado para predecir response_type_optimo.
    n_jobs=1 — evita MemoryError de joblib en Windows.
    """
    print("\n" + "═"*60)
    print("  MODELO CENTRAL: RESPONSE_TYPE (multiclase)")
    print("═"*60)

    df   = master.copy()
    cols = [c for c in FEATURES_RESPONSE_MODEL if c in df.columns]
    X    = df[cols].copy()
    X    = _preparar_X(X)
    cols = X.columns.tolist()   # sincronizar tras deduplicar/limpiar
    y    = df['response_type_optimo']

    # CV macro-F1 — n_jobs=1 para Windows
    kfold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_m = GradientBoostingClassifier(
        n_estimators=160, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    cv_f1 = cross_val_score(base_m, X, y, cv=kfold, scoring='f1_macro', n_jobs=1)
    print(f"  CV Macro-F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    cal_m = CalibratedClassifierCV(base_m, method='isotonic', cv=5)
    cal_m.fit(X_tr, y_tr)
    y_pred = cal_m.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Test Accuracy={acc:.4f} | Test Macro-F1={f1_mac:.4f}")
    print("\n  Classification report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    fi_list = [cc.estimator.feature_importances_
               for cc in cal_m.calibrated_classifiers_]
    fi_mean = np.mean(fi_list, axis=0)
    top12   = pd.Series(fi_mean, index=cols).nlargest(12)
    print("  Top-12 features:")
    for feat, imp in top12.items():
        mark = " ★" if 'noise' in feat else ""
        print(f"    {feat:<45} {imp:.4f}{mark}")

    _plot_confusion_matrix(y_te, y_pred, cal_m.classes_)
    return cal_m, cols


def _plot_confusion_matrix(y_te, y_pred, classes):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    cm   = confusion_matrix(y_te, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title('Matriz de confusión — response_type_optimo',
                 fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(_out('plot_confusion_matrix_response.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Confusion matrix guardada")


def predecir_response_type(master, modelo_response, cols_response):
    df = master.copy()
    X  = df[[c for c in cols_response if c in df.columns]].copy()
    X  = _preparar_X(X)
    probs   = modelo_response.predict_proba(X)
    classes = modelo_response.classes_
    df['response_type_pred']  = classes[np.argmax(probs, axis=1)]
    df['response_confianza']  = probs.max(axis=1)
    for i, c in enumerate(classes):
        df[f'prob_response_{c}'] = probs[:, i]
    return df


# ═══════════════════════════════════════════════════════════
# SECCIÓN 5C — NOISE GATE (inferencia en tiempo real)
# ═══════════════════════════════════════════════════════════
#
# CAPA 1 — nivel usuario (batch semanal):
#   conv_noise_pct > 0.40 → modo educativo, sin cross-sell
#
# CAPA 2 — nivel mensaje (tiempo real):
#   noise_score_msg > 0.50 → respuesta neutra/lúdica, sin producto

NOISE_GATE_UMBRAL_USUARIO  = 0.40
NOISE_GATE_UMBRAL_MENSAJE  = 0.50


def noise_gate_turno_actual(texto_actual):
    """
    Evalúa si el mensaje actual es ruido.
    Llamar antes de que Havi genere la respuesta.

    Returns:
        gate_activado      bool  — True → no hacer cross-sell
        noise_type         str   — categoría del ruido
        noise_score        float — puntuación [0, 1]
        respuesta_sugerida str|None — respuesta lúdica si aplica
    """
    features = calcular_noise_features(texto_actual)
    ns       = features['noise_score_msg']
    nt       = features['noise_type']

    gate_activado = bool(ns >= NOISE_GATE_UMBRAL_MENSAJE)

    respuestas_ludicas = {
        'broma_offtopic':      "¡Jaja! Soy Havi, tu asistente Hey Banco 😄 ¿En qué te puedo ayudar con tu cuenta hoy?",
        'solo_simbolos':       "¿Me puedes escribir en texto lo que necesitas? ¡Estoy aquí para ayudarte!",
        'exploracion_corta':   "¿Hay algo en lo que te pueda ayudar? Puedo orientarte con tu cuenta, tarjetas, crédito o inversiones.",
        'menu_navegacion':     None,
        'afirmacion_negacion': None,
        'artefacto_voz':       None,
        'on_topic':            None,
    }
    resp = respuestas_ludicas.get(nt, None) if gate_activado else None

    return gate_activado, nt, round(ns, 3), resp


# ═══════════════════════════════════════════════════════════
# SECCIÓN 6 — MOTOR DE DECISIÓN HAVI
# ═══════════════════════════════════════════════════════════

HAVI_PROFILES = {
    "PYME Emprendedor": {
        "tono":      "Formal, propositivo",
        "canal_pref":"Chat texto + correo resumen",
        "horario":   "09:00–11:00 y 15:00–17:00",
        "msg_style": "Propuesta con datos, sin emoji",
        "productos": ["tarjeta_credito_negocios","cuenta_negocios","credito_personal"],
        "trigger":   "Transferencias +30% vs mes anterior",
    },
    "Digital Básico": {
        "tono":      "Informal, directo",
        "canal_pref":"Push notification + chat texto",
        "horario":   "20:00–01:00",
        "msg_style": "Corto, accionable en 1 tap",
        "productos": ["tarjeta_credito_hey","credito_personal","inversion_hey"],
        "trigger":   "Utilización > 70% o 3 días sin abrir app",
    },
    "Premium Digital": {
        "tono":      "Sofisticado, proactivo",
        "canal_pref":"Push app iOS + chat texto",
        "horario":   "07:00–09:00",
        "msg_style": "Anticipar necesidades sin que pregunten",
        "productos": ["inversion_hey","credito_auto","seguro_compras"],
        "trigger":   "Transacción internacional o utilización > 50%",
    },
    "Familia Establecida": {
        "tono":      "Semi-formal, empático",
        "canal_pref":"Notificación de quincena + chat",
        "horario":   "08:00–10:00 y 18:00–20:00",
        "msg_style": "Beneficio concreto + CTA clara",
        "productos": ["credito_nomina","inversion_hey","seguro_vida"],
        "trigger":   "Día de nómina o NPS < 7",
    },
    "Ahorrador Cauteloso": {
        "tono":      "Simple, empático, sin tecnicismos",
        "canal_pref":"Chat texto",
        "horario":   "10:00–14:00",
        "msg_style": "Educativo, sin presión",
        "productos": ["tarjeta_credito_garantizada","inversion_hey"],
        "trigger":   "Depósito en OXXO o consulta saldo 3+ veces/semana",
    },
}

MENSAJES_EJEMPLO = {
    "PYME Emprendedor":    "Buenos días. Tus transferencias subieron 35% este mes. Podemos incrementar tu límite de negocio. ¿Lo revisamos?",
    "Digital Básico":      "¡Hey! 👋 Tu tarjeta va al 73%. ¿Diferimos tus últimas compras en 3 MSI?",
    "Premium Digital":     "Detectamos una compra en el extranjero. Tu seguro de compras cubre hasta $50,000 MXN. ¿Lo activamos?",
    "Familia Establecida": "Hola, tu nómina llegó. Tienes un crédito preaprobado a 18 meses. ¿Lo revisamos juntos?",
    "Ahorrador Cauteloso": "Hola. Con tu tarjeta garantizada puedes mejorar tu historial paso a paso. ¿Te cuento cómo?",
}


def generar_perfil_havi(user_row):
    """
    Genera el perfil de comunicación para un usuario.

    Prioridad de overrides:
      1. NOISE GATE → fuerza educativo, sin cross-sell
      2. Canal voz detectado
      3. Frustración alta → soporte primero
      4. Utilización alta → diferimiento, no más crédito
      5. Riesgo de churn → retención empática
      6. Tipo de respuesta del modelo central
    """
    seg = user_row.get('segmento_nombre', 'Digital Básico')
    p   = HAVI_PROFILES.get(seg, HAVI_PROFILES["Digital Básico"]).copy()
    overrides = []

    # 1. NOISE GATE (capa usuario)
    noise_pct = user_row.get('conv_noise_pct', 0)
    if noise_pct >= NOISE_GATE_UMBRAL_USUARIO:
        p['msg_style']     = ('Modo exploración: respuestas educativas y simples. '
                              'NO presentar productos ni cross-sell.')
        p['productos']     = []
        p['response_type'] = 'educativo'
        p['overrides']     = [f'noise_gate_usuario ({noise_pct:.0%} ruido)']
        p['mensaje_ejemplo'] = ("¡Hola! Soy Havi. Puedo ayudarte con tu cuenta, "
                                "tarjetas, transferencias o inversiones. ¿Qué necesitas?")
        return p

    # 2. Canal voz
    if user_row.get('conv_pct_voz', 0) > 0.30:
        p['canal_pref'] = 'Voz (preferencia detectada)'
        overrides.append('canal→voz')

    # 3. Frustración alta
    if user_row.get('conv_frustracion', 0) >= 2:
        p['msg_style'] = 'Resolver primero. Sin venta hasta confirmar solución.'
        overrides.append('frustración_alta')

    # 4. Utilización alta
    if user_row.get('prod_util_max', 0) > 0.85:
        p['productos'] = ['diferimiento_MSI', 'inversion_hey']
        overrides.append('util_alta→no_credito')

    # 5. Riesgo de churn
    if (user_row.get('dias_desde_ultimo_login', 0) > 30 and
            user_row.get('satisfaccion_1_10', 10) < 6):
        p['msg_style'] = 'Reactivación empática + beneficio inmediato'
        overrides.append('riesgo_churn')

    # 6. Tipo de respuesta del modelo central
    rt = user_row.get('response_type_pred',
                       user_row.get('response_type_optimo', 'educativo'))
    if rt == 'soporte':
        p['msg_style'] = 'Diagnóstico claro, pasos concretos. Sin venta.'
        p['productos']  = []
        overrides.append('response→soporte')
    elif rt == 'retencion':
        p['msg_style'] = 'Reactivación empática con beneficio inmediato'
        overrides.append('response→retencion')
    elif rt == 'alerta_preventiva':
        p['msg_style'] = 'Alerta preventiva: seguridad, claridad, acción rápida'
        p['productos']  = ['bloqueo_tarjeta', 'aclaracion_movimiento']
        overrides.append('response→alerta')
    elif rt == 'cross_sell':
        p['msg_style'] = 'Oferta contextual sin presión'
        overrides.append('response→cross_sell')
    else:
        p['msg_style'] = 'Educativo, simple, guiado paso a paso'
        overrides.append('response→educativo')

    p['response_type']   = rt
    p['overrides']       = overrides
    p['mensaje_ejemplo'] = MENSAJES_EJEMPLO.get(seg, "")
    return p


def generar_tabla_perfiles(master):
    print("\n  [Havi] generando perfiles...")
    registros = []
    for _, row in master.iterrows():
        p = generar_perfil_havi(row)
        registros.append({
            'user_id':            row['user_id'],
            'segmento_id':        row.get('segmento_id', -1),
            'segmento_nombre':    row.get('segmento_nombre', ''),
            'tono':               p['tono'],
            'canal_pref':         p['canal_pref'],
            'horario':            p['horario'],
            'response_type':      p.get('response_type', ''),
            'response_confianza': round(row.get('response_confianza', 0), 3),
            'noise_gate_activo':  int('noise_gate_usuario' in ' '.join(p['overrides'])),
            'conv_noise_pct':     round(row.get('conv_noise_pct', 0), 3),
            'producto_1':         p['productos'][0] if p['productos'] else '',
            'producto_2':         p['productos'][1] if len(p['productos']) > 1 else '',
            'msg_style':          p['msg_style'],
            'trigger':            p['trigger'],
            'overrides':          ', '.join(p['overrides']) if p['overrides'] else 'ninguno',
            'mensaje_ejemplo':    p['mensaje_ejemplo'],
        })
    df = pd.DataFrame(registros)
    ng = df['noise_gate_activo'].sum()
    print(f"    → {len(df):,} perfiles | noise gate activo: {ng:,} ({ng/len(df):.1%})")
    return df


def simular_impacto_negocio(master):
    print("\n" + "═"*60 + "\n  IMPACTO DE NEGOCIO\n" + "═"*60)
    n            = len(master)
    churners     = int((master.get('target_riesgo_churn', 0) == 1).sum())
    cross        = int((master.get('response_type_pred','') == 'cross_sell').sum())
    soporte_n    = int((master.get('response_type_pred','') == 'soporte').sum())
    exploradores = int((master.get('target_interaccion_ruido', 0) == 1).sum())

    kpis = {
        'Total usuarios':                      n,
        'Usuarios en riesgo de churn':         churners,
        'Retenciones estimadas (8% tasa)':     int(churners * 0.08),
        'Oportunidades cross-sell':            cross,
        'Conversiones estimadas (4% tasa)':    int(cross * 0.04),
        'Casos de soporte':                    soporte_n,
        'Deflexión repetidos estimada (12%)':  int(soporte_n * 0.12),
        'Exploradores del bot (noise gate)':   exploradores,
        'Mensajes promocionales filtrados':    exploradores,
    }
    print("\n  KPIs simulados para pitch:")
    for k, v in kpis.items():
        print(f"    {k:<45} {v:>8,}")
    return kpis


# ═══════════════════════════════════════════════════════════
# SECCIÓN 7 — VISUALIZACIONES ADICIONALES
# ═══════════════════════════════════════════════════════════

def plot_noise_por_segmento(master):
    if 'segmento_nombre' not in master.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='white')
    COLORES = {
        'PYME Emprendedor':    '#7F77DD',
        'Digital Básico':      '#378ADD',
        'Premium Digital':     '#D85A30',
        'Familia Establecida': '#1D9E75',
        'Ahorrador Cauteloso': '#BA7517',
    }
    ax1   = axes[0]
    segs  = master['segmento_nombre'].unique()
    data  = [master[master['segmento_nombre'] == s]['conv_noise_pct'].values for s in segs]
    colors = [COLORES.get(s, '#888') for s in segs]
    bp = ax1.boxplot(data, patch_artist=True, labels=segs,
                     medianprops={'color':'white','lw':2})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax1.axhline(NOISE_GATE_UMBRAL_USUARIO, color='#E24B4A', ls='--', lw=1.2,
                label=f'Umbral noise gate ({NOISE_GATE_UMBRAL_USUARIO:.0%})')
    ax1.set_title('Ruido (conv_noise_pct) por segmento', fontsize=10, fontweight='bold')
    ax1.set_ylabel('% mensajes ruidosos')
    ax1.set_facecolor('#f8f8f8')
    ax1.tick_params(axis='x', rotation=20, labelsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y', linewidth=0.5)

    ax2  = axes[1]
    gate = master.groupby('segmento_nombre').apply(
        lambda g: (g['conv_noise_pct'] >= NOISE_GATE_UMBRAL_USUARIO).mean() * 100
    ).sort_values()
    bars = ax2.barh(gate.index, gate.values,
                    color=[COLORES.get(s,'#888') for s in gate.index],
                    height=0.55, alpha=0.85)
    for bar, v in zip(bars, gate.values):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=9)
    ax2.set_title('% usuarios con noise gate activo por segmento',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('% usuarios')
    ax2.set_xlim(0, gate.max() * 1.3 + 1)
    ax2.set_facecolor('#f8f8f8')
    ax2.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(_out('plot_noise_por_segmento.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Noise por segmento guardado")


def plot_noise_types_breakdown(convs_raw):
    convs = convs_raw.dropna(subset=['input']).copy()
    noise = convs['input'].apply(calcular_noise_features)
    convs = pd.concat([convs, noise], axis=1)
    tipos = convs['noise_type'].value_counts()
    COLORES = {
        'on_topic':            '#1D9E75',
        'exploracion_corta':   '#BA7517',
        'afirmacion_negacion': '#378ADD',
        'menu_navegacion':     '#7F77DD',
        'broma_offtopic':      '#D85A30',
        'solo_simbolos':       '#E24B4A',
        'artefacto_voz':       '#888888',
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='white')
    ax1   = axes[0]
    total = len(convs)
    bars  = ax1.barh(tipos.index, tipos.values,
                     color=[COLORES.get(t,'#888') for t in tipos.index],
                     height=0.55, alpha=0.85)
    for bar, v in zip(bars, tipos.values):
        ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                 f'{v:,} ({v/total:.1%})', va='center', fontsize=9)
    ax1.set_title('Tipos de ruido en conversaciones Havi', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Número de turnos')
    ax1.set_xlim(0, tipos.max() * 1.4)
    ax1.set_facecolor('#f8f8f8')
    ax1.grid(True, alpha=0.3, axis='x', linewidth=0.5)

    ax2          = axes[1]
    ruido_data   = convs[convs['noise_type'] != 'on_topic']['noise_score_msg']
    ontopic_data = convs[convs['noise_type'] == 'on_topic']['noise_score_msg']
    ax2.hist(ontopic_data, bins=20, color='#1D9E75', alpha=0.6,
             label='on_topic', density=True)
    ax2.hist(ruido_data,   bins=20, color='#D85A30', alpha=0.6,
             label='ruido',    density=True)
    ax2.axvline(NOISE_GATE_UMBRAL_MENSAJE, color='#E24B4A', ls='--', lw=1.5,
                label=f'Umbral mensaje ({NOISE_GATE_UMBRAL_MENSAJE})')
    ax2.set_title('Distribución noise_score por tipo de mensaje',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('noise_score_msg')
    ax2.set_ylabel('Densidad')
    ax2.legend(fontsize=9)
    ax2.set_facecolor('#f8f8f8')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(_out('plot_noise_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Tipos de ruido guardados")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  HEY BANCO DATATHON 2026 — PIPELINE v3")
    print("  (GBM + Noise Gate)")
    print("═"*60)

    # 1. Cargar
    clientes, productos, txn, convs = cargar_datos()

    # 2. Limpiar
    print("\n" + "═"*60 + "\n  LIMPIEZA\n" + "═"*60)
    clientes  = limpiar_clientes(clientes)
    productos = limpiar_productos(productos)
    txn       = limpiar_transacciones(txn)
    convs     = limpiar_conversaciones(convs)

    # 3. Features + master
    print("\n" + "═"*60 + "\n  FEATURE ENGINEERING\n" + "═"*60)
    master = construir_master(clientes, productos, txn, convs)

    # 4. Segmentación
    master, scaler, km, cols_clust, inercias, sils = segmentar_usuarios(master)
    master = nombrar_segmentos(master)

    # 5. Targets + modelos
    master = construir_targets(master)
    master = construir_response_type_optimo(master)
    modelos_binarios = entrenar_modelos(master, plot=True)
    modelo_response, cols_response = entrenar_modelo_response_type(master)
    master = predecir_response_type(master, modelo_response, cols_response)
    impacto = simular_impacto_negocio(master)

    # 6. Motor Havi
    perfiles = generar_tabla_perfiles(master)

    # 7. Visualizaciones adicionales
    print("\n" + "═"*60 + "\n  VISUALIZACIONES\n" + "═"*60)
    plot_noise_por_segmento(master)
    plot_noise_types_breakdown(convs)

    # 8. Guardar outputs
    print("\n" + "═"*60 + "\n  GUARDANDO OUTPUTS\n" + "═"*60)
    master.to_csv(_out('master_usuarios_v3.csv'), index=False)
    perfiles.to_csv(_out('perfiles_havi_v3.csv'), index=False)
    print(f"  → master_usuarios_v3.csv")
    print(f"  → perfiles_havi_v3.csv")

    # Demo: noise gate en tiempo real
    print("\n" + "═"*60)
    print("  DEMO: NOISE GATE EN TIEMPO REAL")
    print("═"*60)
    mensajes_demo = [
        "Cuánto es mi saldo disponible en la tarjeta de crédito?",
        "jaja hola",
        "B",
        "si",
        "Quiero hacer una transferencia a otro banco por SPEI",
        "Cuéntame un chiste",
        "No me deja hacer la transferencia, me sale error de token",
    ]
    print(f"\n  {'Mensaje':<50} {'Gate':^6} {'Tipo':<22} {'Score':^6}")
    print("  " + "─"*90)
    for msg in mensajes_demo:
        gate, tipo, score, resp = noise_gate_turno_actual(msg)
        flag = "NO" if gate else "OK"
        print(f"  {msg[:49]:<50} {flag:^6} {tipo:<22} {score:^6.3f}")
        if resp:
            print(f"    Havi: {resp[:85]}")

    print("\n" + "═"*60)
    print("  PIPELINE v3 COMPLETADO")
    print("═"*60)