# OACE-method-random-walk

## Descrição
**OACE-ModelEvaluation** é um projeto destinado a aplicar o método OACE (Optimized Assertiveness-Cost Evaluation) em diferentes cenários de aprendizado de máquina. O método OACE é uma abordagem inovadora para avaliar o desempenho de modelos de aprendizado de máquina, equilibrando assertividade (precisão, acurácia, recall) e custo computacional (tempo de inferência, número de parâmetros do modelo, tamanho do modelo).

## Objetivo
O objetivo deste projeto é investigar a eficácia do método OACE em três cenários distintos utilizando diferentes datasets:
1. **Alta Importância da Assertividade**: Utilizando o dataset de classificação de pneumonia.
2. **Equilíbrio entre Assertividade e Custo**: Utilizando o dataset CIFAR-10 para classificação de objetos.
3. **Alta Importância do Custo**: Utilizando o dataset TrashNet para classificação de objetos recicláveis.

## Cenários de Estudo
- **Cenário 1: Alta Importância da Assertividade**
  - **Dataset**: [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  - **Parâmetro λ** > 0.5

- **Cenário 2: Equilíbrio entre Assertividade e Custo**
  - **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - **Parâmetro** 0 < **λ** < 1

- **Cenário 3: Alta Importância do Custo**
  - **Dataset**: [TrashNet](https://github.com/garythung/trashnet)
  - **Parâmetro λ** < 0.5

## Metodologia

- **Random Walk**: Aplicamos um algoritmo de Random Walk para explorar diferentes configurações de taxa de aprendizado e seleção de modelos, identificando o melhor modelo em cada cenário.

- **Modelos Utilizados**: EfficientNet, MobileNet, ResNet, Inception, ViT e VGG.

- **Pré-aquecimento dos Modelos**: Antes de aplicar o método OACE, os modelos passam por um pré-aquecimento utilizando uma versão reduzida dos datasets para capturar as métricas de custo e normalizá-los.

- **Definição de Pesos das Métricas (AHP)**: Para ponderar a importância de cada métrica, os pesos relativos foram calculados utilizando o método *Analytic Hierarchy Process* (AHP), que quantifica a relevância dos critérios com base em julgamentos par-a-par.
    - A matriz de importância relativa, aplicada tanto aos critérios de **assertividade** (precisão, acurácia, recall) quanto aos de **custo** (MTP, TPI, MS), é dada por:
      
      $$
      A_{m \times n} = \begin{bmatrix}
      1 & 5 & 7 \\
      \frac{1}{5} & 1 & 3 \\
      \frac{1}{7} & \frac{1}{3} & 1
      \end{bmatrix}
      $$
      
      Onde os índices $m, n \in [1, 3]$ referem-se aos critérios na ordem em que foram listados.

    - O vetor de pesos resultante, comum a todos os cenários, foi $w_i^a, w_j^c \in \{0.731, 0.188, 0.081\}$.

    - **Justificativa da Ponderação**:
        - **Assertividade**: A *precisão* obteve o maior peso (0.731) por refletir melhor a proporção de predições corretas positivas e evitar falsos positivos, superando a *acurácia* (0.188) e o *recall* (0.081).
        - **Custo**: O *MTP (Mean Time Prediction)* foi priorizado (0.731) por impactar diretamente a velocidade de execução e a memória, sendo considerado mais crítico que o *TPI (Time Per Inference)* (0.188) e o *MS (Model Size)* (0.081).

## Estrutura do Projeto
- `base/`: Arquivos base e configurações.
- `datasets.py`: Código para importação e manipulação dos datasets utilizados no projeto.
- `hyperparameter_optimization.py`: Scripts para otimização de hiperparâmetros dos modelos.
- `main.py`: Script principal para execução do método OACE e experimentos.
- `metrics.py`: Implementação das métricas de avaliação de modelos.
- `models.py`: Definição e configuração dos modelos de aprendizado de máquina.
- `validation.py`: Scripts para validação dos modelos treinados.
- `visualization.py`: Ferramentas para visualização dos resultados e métricas.
- `requirements.txt`: Lista de dependências necessárias para o projeto.

## Como Usar
1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/OACE-ModelEvaluation.git
    ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute o scripts `main.py` para executar o algoritmo, treinar os modelos e avaliar o desempenho utilizando o método OACE.
