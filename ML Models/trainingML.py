import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import math
import tqdm
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- Função para Obter o Caminho do Dataset ---
def get_dataset():
    # Se download ja tiver sido feito, o dataset já está no cache e retorna o caminho
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Dataset baixado em:", path)
    print(os.listdir(path))
    return path

# --- Função para Converter Imagens do Dataset em Arrays NumPy ---
def ds_to_arrays(path, split):
    imgs, labs = [], []                          # Inicializa listas para imagens e rótulos
    split_path = os.path.join(path, split)       # Cria o caminho para 'train' ou 'test'
    class_names = sorted(os.listdir(split_path)) # Pega os nomes das classes (pastas)

    # Cria um dicionário para mapear nome da classe (ex: 'happy') para um índice (ex: 3)
    label_map = {name: idx for idx, name in enumerate(class_names)}

    # Itera por cada pasta de classe (emoção) com uma barra de progresso (tqdm)
    for label_name in tqdm.tqdm(class_names):
        label_dir = os.path.join(split_path, label_name)

        # Itera por cada arquivo de imagem dentro da pasta da emoção
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            try:
                # Abre a imagem, converte para escala de cinza ('L') e redimensiona para 48x48
                img = Image.open(file_path).convert('L').resize((48, 48))
                # Converte a imagem para array numpy (float32) e adiciona à lista 'imgs'
                imgs.append(np.array(img, dtype="float32"))
                # Adiciona o índice numérico do rótulo (ex: 3) à lista 'labs'
                labs.append(label_map[label_name])
            except Exception as e:
                print("Erro ao carregar imagem:", file_path, e)

    # Empilha todas as imagens da lista em um único array NumPy
    X = np.stack(imgs, axis=0)

    # Adiciona uma dimensão de canal (de (N, 48, 48) para (N, 48, 48, 1))
    X = X[..., np.newaxis] / 255.0

    # Converte os rótulos para codificação one-hot
    y = to_categorical(labs, num_classes=len(class_names))
    
    return X, y, class_names

# --- Função para Carregar Dados de Treino e Teste ---
def get_training_values(path):
    # Cria conjuntos de treino e teste
    X_train, y_train, class_names = ds_to_arrays(path, 'train')
    X_test, y_test, _ = ds_to_arrays(path, 'test')
    return X_train, y_train, X_test, y_test, class_names

# --- Função para Plotar o Histórico de Treinamento ---
def history_plot(history):
    plt.figure(figsize=(12, 4))

    # Primeiro subplot: Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Segundo subplot: Perda (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# --- Função para Treinar um Modelo do Zero ---
def training(X_train, y_train, X_test, y_test, model_file, epochs):
# ==== DATA AUGMENTATION ====
    # Configura o gerador de imagens para aplicar transformações aleatórias
    datagen = ImageDataGenerator(
        rotation_range=10,      # Rotaciona até 10 graus
        width_shift_range=0.1,  # Desloca horizontalmente até 10%
        height_shift_range=0.1, # Desloca verticalmente até 10%
        horizontal_flip=True    # Inverte horizontalmente
    )
    datagen.fit(X_train)

    # ==== MODELO CNN ====
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
        input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()

    # ==== TREINAMENTO ====
    batch_size = 64

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
    )

    # ==== SALVAR MODELO ====
    model.save(model_file.strip() + ".h5")
    print("Modelo treinado e salvo com sucesso!")

    return history

# --- Função para Fazer Fine-Tuning (Ajuste Fino) em um Modelo Existente ---
def fine_tune_model(base_model_path,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output_model_path,
                    epochs=10,
                    unfrozen_layers=3,
                    learning_rate=1e-5):

    print("Carregando modelo base...")
    model = load_model(base_model_path)

    # ==== CONGELAR CAMADAS ====
    print(f"Congelando todas as camadas, exceto as últimas {unfrozen_layers}...")
    for layer in model.layers[:-unfrozen_layers]:
        layer.trainable = False
    for layer in model.layers[-unfrozen_layers:]:
        layer.trainable = True

    # ==== COMPILAÇÃO COM LR BAIXO ====
    # Re-compila o modelo com uma taxa de aprendizado (learning_rate) muito baixa
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ==== DATA AUGMENTATION ====
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # ==== FINE-TUNING ====
    print("Iniciando fine-tuning...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_test, y_test),
        epochs=epochs,
        verbose=1
    )

    # ==== SALVAR MODELO AJUSTADO ====
    model.save(output_model_path.strip() + '_fineTune' + ".h5")
    print(f"Fine-tuning concluído! Modelo salvo como '{output_model_path}.h5'")

    return history

# --- Função Principal que Executa o Script ---
def run():

    def define_input():
        model_file = str(input("Digite o nome do arquivo: "))
        num_samples = int(input("Digite o número de exemplos pra serem mostrados: "))
        epochs = int(input("Digite o número de épocas para treinamento: "))

        if epochs == 0 or epochs is None:
            epochs = 30
        elif epochs < 0:
            epochs = 30
        elif epochs != 30:
            # Verifica se o modelo não existe
            # Se o modelo já existe, ele será usado para fine-tuning
            if str(model_file + '.h5') not in os.listdir():
                model_file = model_file + f"_ep{epochs}"
                print('Arquivo que será gerado: ' + model_file + '.h5')
            else:
                print('Modelo para Fine Tunning: ' + str(model_file + '.h5'))

        return model_file, num_samples, epochs

    model_file, num_samples, epochs = define_input()

    #==== DEFINIÇÃO DO CAMINHO BASE DO DATASET ====
    path = get_dataset()

    X_train, y_train, X_test, y_test, class_names = get_training_values(path)
    
    # Verifica se o arquivo .h5 do modelo JÁ EXISTE no diretório atual
    if str(model_file.strip() + '.h5') in os.listdir():
        print(f"O modelo '{model_file}.h5' já existe. Iniciando Fine-Tuning...")
        history = fine_tune_model(
            base_model_path=str(model_file.strip() + '.h5'),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_model_path=model_file.strip(),
            epochs=epochs,
            unfrozen_layers=3,
            learning_rate=1e-5
        )
    else:
        # Se NÃO existe, inicia o TREINAMENTO DO ZERO
        history = training(X_train, y_train, X_test, y_test, model_file, epochs)
    
    history_plot(history)
    #====

    print(class_names)

    #==== CARREGA O MODELO TREINADO (CASO JA TENHA SIDO TREINADO)====
    model = load_model(model_file.strip() + ".h5")
    #====

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    #==== MOSTRA PREVISÕES ALEATÓRIAS ====
    for _ in range(num_samples):
        idx = np.random.randint(0, X_test.shape[0])
        sample = X_test[idx:idx+1]
        pred = model.predict(sample, verbose=0)
        label = emotion_labels[np.argmax(pred)]
        true = emotion_labels[np.argmax(y_test[idx])]
        print(f'True: {true:>8} - Predicted: {label:>8}')

    #==== PLOTAGEM DAS PREVISÕES ====
    indices = np.random.choice(range(X_test.shape[0]), size=num_samples, replace=False)

    cols = 10
    rows = math.ceil(num_samples / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(indices[:num_samples]):
        img = X_test[idx].squeeze()
        pred = model.predict(X_test[idx:idx+1], verbose=0)
        pred_label = emotion_labels[np.argmax(pred)]
        true_label = emotion_labels[np.argmax(y_test[idx])]

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray')
        
        # Deixa o título compacto e centralizado
        ax.set_title(f'True: ' + str({true_label[:5]}) + '\nPred: ' + str({pred_label[:5]}),
                    fontsize=9, pad=2, color=('green' if true_label == pred_label else 'red'))
        
        ax.axis('off')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # espaçamento horizontal e vertical
    plt.show()

if __name__ == "__main__":
    run()