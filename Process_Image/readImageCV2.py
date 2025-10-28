import cv2
import numpy as np
import os
import PIL.Image as Image 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def read_image_cv2(image_path):
    print(cv2.__version__)

    img = cv2.imread(image_path)

    if img is not None: return img
    else: return None

formato = '.' + input("Formato da imagem (ex: jpg, png): ")
arquivo_imagem = input("Nome da imagem sem extensão (ex: feliz): ")
caminho_imagem = r'TestImages' + os.sep + arquivo_imagem + formato
caminho_png = r'TestImages' + os.sep + arquivo_imagem + '.png'
caminho_modelo = r'Modelo15_ep1000.h5'

if formato.lower() != '.png':
    # Abrir a imagem
    image = Image.open(caminho_imagem)

    # Converter e salvar como PNG
    image.save(caminho_png, format="PNG")

    # Usar a imagem PNG para o restante do código
    caminho_imagem = caminho_png    

img = read_image_cv2(caminho_imagem)
print(img)

# Carregar imagem
if img is None:
    raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")

# Labels (ajuste conforme suas classes)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] # POSIÇÕES AJUSTADAS

# Carregar modelo
modelo = load_model(caminho_modelo)

# Carregar classificador de rosto (HaarCascade padrão do OpenCV)
detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Converter para escala de cinza
imagem_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar rostos na imagem
rostos = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(rostos) == 0:
    print("Nenhum rosto detectado.")
else:
    for (x, y, w, h) in rostos:
        # Recortar o rosto
        rosto = imagem_cinza[y:y+h, x:x+w]

        # Redimensionar para 48x48
        rosto_redimensionado = cv2.resize(rosto, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalizar e preparar para o modelo
        rosto_array = rosto_redimensionado.astype("float") / 255.0
        rosto_array = img_to_array(rosto_array)
        rosto_array = np.expand_dims(rosto_array, axis=0)

        # Fazer predição
        pred = modelo.predict(rosto_array, verbose=0)[0]
        indice = np.argmax(pred)
        expressao = labels[indice]
        confianca = pred[indice] * 100

        print(pred)
        print(labels)
        print(indice)
        print(f"Expressão detectada: {expressao} ({confianca:.2f}%)")

        show = input("Deseja mostrar o rosto detectado e redimensionado? (s/n): ").lower()
        if show == 's':
            show = True
        else:
            show = False

        # Opcional: mostrar o rosto detectado e redimensionado
        if show == True:
            cv2.imwrite(str("RESIZE_GRAY_" + arquivo_imagem + ".png"), rosto_redimensionado)
            cv2.imshow("Rosto (48x48)", rosto_redimensionado)
            cv2.waitKey(0)
            cv2.destroyAllWindows()