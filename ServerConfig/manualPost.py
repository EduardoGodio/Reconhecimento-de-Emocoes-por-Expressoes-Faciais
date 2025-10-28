import requests
import PIL.Image as Image

#url = "http://127.0.0.1:5000/post_expression"  # seu endpoint (Para servidor local)
url = r"URL_NGROK/post_expression"  # seu endpoint (Para servidor externo via ngrok)
image_path = "neutro3.png"         # caminho da imagem local

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print("\nStatus code:", response.status_code)
print("Resposta:", response.text)