# Havi — Hey Banco · Datathon 2026
## Deploy en Railway

### Archivos que van en el repositorio

```
tu-repo/
├── hey_server.py               ← servidor Flask (este archivo)
├── hey_pipeline_v4.py          ← pipeline de perfiles
├── hey_databases.py            ← constructor de índices
├── hey_agent_havi_v2.py        ← lógica del agente
├── requirements.txt            ← dependencias Python
├── Procfile                    ← comando de arranque
├── railway.json                ← configuración Railway
├── .gitignore
├── hey_clientes.csv            ← datos (sintéticos, ok subirlos)
├── hey_productos.csv
├── hey_transacciones.csv
├── dataset_50k_anonymized.csv
└── outputs/                    ← SUBIR estos archivos pre-generados
    ├── perfiles_usuarios.csv
    ├── master_usuarios.csv
    ├── metricas_propension.csv
    ├── noise_detector.pkl
    └── rag_index.pkl
```

> **Importante:** genera los archivos de `outputs/` localmente ANTES
> de hacer el push. El servidor puede construirlos al arrancar, pero
> tarda ~5 minutos y Railway puede dar timeout.

---

### Paso 1 — Generar los outputs localmente

```cmd
cd C:\Users\kiki7\Downloads\Datathon-2026\data

C:\Users\kiki7\miniconda3\python.exe hey_databases.py
C:\Users\kiki7\miniconda3\python.exe hey_pipeline_v4.py
```

Verifica que existan:
- `outputs/noise_detector.pkl`
- `outputs/rag_index.pkl`
- `outputs/perfiles_usuarios.csv`

---

### Paso 2 — Crear repositorio en GitHub

```cmd
cd C:\Users\kiki7\Downloads\Datathon-2026\data
git init
git add .
git commit -m "Havi Datathon 2026 - deploy inicial"
```

1. Ve a **github.com → New repository**
2. Nombre: `havi-datathon-2026`
3. Privado (recomendado — tiene tu API key en variables, no en código)
4. Copia la URL del repo

```cmd
git remote add origin https://github.com/TU_USUARIO/havi-datathon-2026.git
git branch -M main
git push -u origin main
```

---

### Paso 3 — Deploy en Railway

1. Ve a **railway.app** e inicia sesión con GitHub
2. **New Project → Deploy from GitHub repo**
3. Selecciona `havi-datathon-2026`
4. Railway detecta automáticamente el `Procfile`

**Agregar la API key:**
- En Railway → tu proyecto → **Variables**
- Click en **New Variable**
- Nombre: `ANTHROPIC_API_KEY`
- Valor: `sk-ant-api03-...`
- Click en **Add**

5. Railway hace el deploy automáticamente
6. Ve a **Settings → Domains → Generate Domain**
7. Obtienes tu URL: `https://havi-datathon-2026.railway.app`

---

### Paso 4 — Verificar que funciona

Abre en el navegador:
```
https://tu-app.railway.app/
```

Debe responder:
```json
{
  "status": "ok",
  "usuarios": 15025,
  "segmentos": 6,
  "rag_docs": 18432
}
```

Probar el chat:
```bash
curl -X POST https://tu-app.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USR-00001", "mensaje": "¿Cuánto debo de mi tarjeta?"}'
```

---

### Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/perfil/USR-00001` | Perfil completo del usuario |
| POST | `/chat` | Turno de conversación |
| GET | `/clusters` | Resumen de los 6 segmentos |
| GET | `/metricas` | AUC de propensión por producto |
| GET | `/usuarios?n=10` | Lista de usuarios disponibles |
| GET | `/usuarios?segmento=Premium` | Filtrar por segmento |

---

### Troubleshooting

**El servidor dice "iniciando":**
Está construyendo las bases de datos. Espera 3-5 minutos y recarga.

**Error 503 con "Error de inicialización":**
Revisa los logs en Railway → tu proyecto → **Deployments → Ver logs**.
El error más común es que falta `ANTHROPIC_API_KEY` en Variables.

**Los archivos .pkl son demasiado grandes para GitHub (>100MB):**
```cmd
# Instalar Git LFS
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add outputs/
git commit -m "Agregar modelos con LFS"
git push
```

**Railway da timeout en el healthcheck:**
El `railway.json` tiene `healthcheckTimeout: 300` (5 minutos).
Si sigue fallando, sube los .pkl pre-generados en lugar de construirlos al arrancar.
