import os
import math
import numpy as np
import matplotlib.pyplot as plt
from trainingML import get_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# === entradas ===
model_file = input("Nome do arquivo do modelo (sem .h5): ").strip()
num_samples = int(input("Número de exemplos a mostrar: "))

base_path = get_dataset()
test_dir = os.path.join(base_path, "test")

# === carrega modelo ===
model = load_model(model_file + ".h5")

# === gerador de teste (normalização) ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=1,   # pega uma imagem por vez
    shuffle=True
)

# === mapeamento correto de labels conforme o gerador ===
# class_indices é dict {class_name: index}
class_indices = test_generator.class_indices
# converte para lista onde index -> class_name
emotion_labels = [None] * len(class_indices)
for name, idx in class_indices.items():
    emotion_labels[idx] = name

print("Mapeamento de classes:", class_indices)
print("Lista ordenada de labels:", emotion_labels)

# === coleta amostras aleatórias usando o gerador ===
images, true_labels, preds, confs = [], [], [], []

for _ in range(num_samples):
    img_batch, label_batch = next(test_generator)           # (1,48,48,1), (1,7)
    pred_proba = model.predict(img_batch, verbose=0)[0]    # vetor de probabilidades
    pred_idx = np.argmax(pred_proba)
    true_idx = np.argmax(label_batch[0])

    images.append(img_batch[0].squeeze())
    true_labels.append(emotion_labels[true_idx])
    preds.append(emotion_labels[pred_idx])
    confs.append(pred_proba[pred_idx])

# === plot organizado em grid ===
cols = 5
rows = math.ceil(num_samples / cols)
plt.figure(figsize=(cols * 2.8, rows * 3))

for i in range(num_samples):
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(images[i], cmap='gray')
    correct = (true_labels[i] == preds[i])
    color = 'green' if correct else 'red'
    # mostra label + confiança em %
    ax.set_title(f'True: {true_labels[i]} \nPreds: {preds[i]} ({confs[i]*100:.1f}%)',
                 fontsize=9, color=color, pad=2)
    ax.axis('off')

plt.suptitle("Predições do Modelo — Verde: Correto | Vermelho: Incorreto", fontsize=12, y=1.02)
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()
