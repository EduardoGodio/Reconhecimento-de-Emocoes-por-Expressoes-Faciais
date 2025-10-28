import cv2
import numpy as np
import os
import PIL.Image as Image 
import io
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

def process_image(image: Image):
    def read_image_cv2(image_buffer):
        """
        Lê uma imagem a partir de um buffer de bytes (ex: io.BytesIO).
        """
        try:
            # Pega todos os bytes do buffer
            image_buffer.seek(0)
            image_bytes = image_buffer.read()
            
            # Converte os bytes brutos para um array numpy
            np_array = np.frombuffer(image_bytes, np.uint8)
            
            # Decodifica o array numpy em uma imagem OpenCV
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Não foi possível decodificar a imagem. Os dados podem estar corrompidos ou em formato inválido.")

            return img
            
        except Exception as e:
            print(f"Erro ao ler imagem do buffer: {e}")
            return None

    def convert_to_png(image: Image):
        # Cria um buffer de memória
        buffer_png = io.BytesIO()

        # Converte e "salva" a imagem para o buffer de memória
        image.save(buffer_png, format="PNG")

        # Retorna o buffer
        buffer_png.seek(0)
        return buffer_png

    buffer_png = convert_to_png(image)

    if buffer_png is None:
        raise ValueError("Não foi possível converter a imagem para PNG.")
    
    img = read_image_cv2(buffer_png)

    # Carregar imagem
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {buffer_png}")

    # Labels com expressões faciais
    #labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] # POSIÇÕES AJUSTADAS

    # Carregar modelo
    modelo = load_model("Modelo15_ep1000.h5")

    # Carregar classificador de rosto (HaarCascade padrão do OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector_face = cv2.CascadeClassifier(cascade_path)

    if detector_face.empty():
        raise IOError(f"Não foi possível carregar o arquivo Haar Cascade em: {cascade_path}.")

    # Converter para escala de cinza
    imagem_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if imagem_cinza is None:
        raise ValueError("A imagem em escala de cinza está vazia (None). A leitura ou conversão falhou.")

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

            return expressao, confianca