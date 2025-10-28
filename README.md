# Reconhecimento de expressões faciais

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Este projeto foi desenvolvido no âmbito da disciplina de Arquitetura de Computadores II. Seu objetivo principal é a criação de uma API em Python capaz de analisar expressões faciais a partir de imagens capturadas pelo microcontrolador ESP32-CAM

---

## 🚀 Sobre o Projeto

### Objetivo Principal
O objetivo central é construir um sistema completo ponta-a-ponta para análise de expressões faciais em tempo real. O sistema utiliza um dispositivo de borda de baixo custo (o ESP32-CAM) para a captura de imagens e uma API robusta em Python para o processamento e análise.

O fluxo de trabalho pretendido é:

1. O ESP32-CAM captura uma imagem do usuário.
2. A imagem é enviada via Wi-Fi para um endpoint específico da API.
3. A API Python recebe a imagem, pré-processa (detectando o rosto) e a submete a um modelo de aprendizado de máquina.
4. A API retorna uma resposta (geralmente em formato JSON) contendo a emoção classificada (ex: "feliz", "neutro", "surpreso").

### Tecnologias Principais
Para alcançar o objetivo, as seguintes tecnologias foram combinadas:

#### Hardware (Dispositivo de Borda):

- ESP32-CAM: Microcontrolador com câmera integrada e conectividade Wi-Fi, programado para capturar e enviar as imagens via requisições HTTP POST.

#### Backend (API):

- Python: Linguagem principal para o desenvolvimento da API.
 - Flask: Utilizado para criar os endpoints da API, gerenciar as requisições HTTP e enviar as respostas.
 - OpenCV: Fundamental para o pré-processamento. Usada para decodificar a imagem recebida, detectar o rosto usando Haar Cascades e preparar a imagem para o modelo de treinamento.
 - TensorFlow/Keras: Utilizada para treinar e carregar o modelo de classificação de emoções tendo em base o dataset FER2013.

## ⚙️ Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento local.

**Crie um ambiente virtual** (Recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

**Instale as dependências:**
    (Note: O nome padrão do arquivo é `requirements.txt`)
    ```sh
    pip install -r requirements.txt
    ```

## 📁 Hierarquia do Projeto
/ (Diretório Raiz)  
├── 📁 ML Models/  
├── 📁 Process_Image/  
├── 📁 ServerConfig/  
├── 📄 README.md  
├── 📄 requirements.txt  
└── 📄 startServer.bat  
