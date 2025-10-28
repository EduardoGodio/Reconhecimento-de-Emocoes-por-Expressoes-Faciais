# TrainingML.py

### Objetivo:
* Realizar o treinamento do modelo de CNN que será utilizado durante o processo de identificação da emoção.
* Permitir tanto o treinamento de um novo modelo quanto o fine-tuning (ajuste fino) de um modelo existente, com base no dataset FER2013.

### Funcionamento/Processo:
1. O script é iniciado solicitando três entradas do usuário:
    * Nome do modelo (String): O nome base do arquivo (sem a extensão .h5).
    * Número de exemplos (int): A quantidade de imagens de teste que serão plotadas no final.
    * Número de épocas (int): Quantas épocas o modelo deve treinar.

2. O script usa o Nome do modelo para verificar se um arquivo .h5 correspondente já existe no diretório:
    * Se o arquivo EXISTIR: O script carrega o modelo e inicia o processo de fine-tuning (ajuste fino).
    * Se o arquivo NÃO EXISTIR: O script inicia o processo de treinamento de um modelo completamente novo, do zero.

3. O modelo é treinado (ou ajustado) usando os dados do FER2013 pelo número de épocas especificado.

### Retorno/Resultado:
* Um arquivo .h5 (do modelo novo ou do modelo ajustado) salvo no disco.
* Gráficos do histórico de treinamento (acurácia e perda).
* Uma visualização de predições no console e em gráfico, usando o número de exemplos fornecido.

# loadModel.py

### Objetivo:
Este script funciona como uma ferramenta de avaliação visual para um modelo de reconhecimento de emoções já treinado.

### Funcionamento/Processo:
1. Em vez de treinar, este script foca em testar. Ele primeiro pede ao usuário qual arquivo de modelo (.h5) deve carregar.

2. Em seguida, ele prepara o __"ImageDataGenerator"__ para ler as imagens do diretório de teste. Ele usa esse gerador para pegar imagens aleatórias, uma de cada vez.

3. Para cada imagem, o script pede ao modelo carregado que faça uma previsão. Ele então armazena essa previsão (ex: 'Happy'), a confiança do modelo (ex: '95.2%') e qual era a emoção verdadeira (ex: 'Happy').

### Retorno/Resultado:
No final, o script exibe uma grade com todas as imagens que testou. Cada imagem tem um título que mostra a previsão vs. o rótulo verdadeiro. Os títulos são coloridos de verde (se o modelo acertou) ou vermelho (se errou), facilitando a visualização rápida do desempenho do modelo.

# 📁 Hierarquia do Diretório
/ 📁 ML Models  
├── 📁 Example_Models/  
├── 📁 TestResults/  
├── 📄 loadModel.py  
├── 📄 README.md  
├── 📄 requirements.txt  
└── 📄 trainingML.py
