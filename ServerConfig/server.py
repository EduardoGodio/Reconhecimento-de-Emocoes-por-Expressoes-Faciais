# UTILIZAR COMANDO NO TERMINAL: ngrok http 5000

import requests
import PIL.Image as Image
from flask import Flask, jsonify, request
import io
from rec_expression import process_image

app = Flask(__name__)

@app.route("/")
def home():
    return "Servidor público Python ativo!"

@app.route("/post_expression", methods=["POST"])
def post_expression():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "expression": "Nenhum arquivo enviado."}), 400
        
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read()))

        expression, trust = process_image(img)
        trust = trust.item()

        result = jsonify({"status": "ok", "expression": expression, "trust": trust})

        print(result)
        return result, 200
    
    except Exception as e:
        result = jsonify({"status": "error", "expression": str(e)})
        print(result)

        return result, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # servidor acessível na rede local