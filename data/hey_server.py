from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, os
import pandas as pd
from hey_agent_havi_v2 import (
    cargar_bases_de_datos, HaviSession
)

app = Flask(__name__)
CORS(app)

# Cargar una sola vez al iniciar
print("Cargando bases de datos...")
model, detector, rag_index = cargar_bases_de_datos()
sesiones = {}   # user_id → HaviSession

@app.route('/chat', methods=['POST'])
def chat():
    data      = request.json
    user_id   = data.get('user_id', 'USR-00001')
    mensaje   = data.get('mensaje', '')

    # Crear sesión si no existe
    if user_id not in sesiones:
        sesiones[user_id] = HaviSession(user_id, model, detector, rag_index)

    respuesta, info = sesiones[user_id].chat(mensaje)
    return jsonify({ 'respuesta': respuesta, 'info': info })

@app.route('/perfil/<user_id>', methods=['GET'])
def perfil(user_id):
    perfiles = pd.read_csv('outputs/perfiles_usuarios.csv')
    row = perfiles[perfiles['user_id'] == user_id]
    if row.empty:
        return jsonify({'error': 'No encontrado'}), 404
    return jsonify(row.iloc[0].to_dict())

if __name__ == '__main__':
    app.run(port=5000, debug=False)