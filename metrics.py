"""
Implementação das métricas e funções para cálculo de
"""
import numpy as np
from models import *
import numpy as np
from pyDecision.algorithm import ahp_method
import json
import pandas as pd
import matplotlib.pyplot as plt

metrics_a = ["precision", "accuracy", "recall"]
metrics_c = ["mtp", "tpi", "ms"]
epsilon = 1e-5 # Parâmetro da padronização para evitar divisão por zero
##################
### AHP Method ###
##################
weight_derivation = 'geometric' # 'mean'; 'geometric' or 'max_eigen'

dataset = np.array([# Dataset for assertiveness metrics (P -> A -> R)
  #P      A      R
  [1  ,   5,     7   ],   #P
  [1/5,   1,     3   ],   #A
  [1/7,   1/3,   1   ],   #R
])
dataset = np.array([ # Dataset for cost metrics (MTP -> TPI -> MS)
  #MTP    TPI    MS
  [  1,     5,   7   ],   #MTP
  [1/5,     1,   3   ],   #TPI
  [1/7,   1/3,   1   ],   #MS
])
# Call AHP Function
weights, rc = ahp_method(dataset, wd = weight_derivation)
w1, w2, w3 = weights[0], weights[1], weights[2]

wa = [w1, w2, w3] # Precision, Acuraccy, Recall
wc = [w1, w2, w3] # MTP, TPI, MS

### OACE Method ###

def get_max_min_metrics(metrics_dict):
    metricas_assertividade = ['accuracy', 'precision', 'recall']
    metricas_custo = ['tpi', 'mtp', 'ms']

    # Inicializando valores máximos e mínimos
    max_assertividade = {metrica: float('-inf') for metrica in metricas_assertividade}
    min_assertividade = {metrica: float('inf') for metrica in metricas_assertividade}

    max_custo = {metrica: float('-inf') for metrica in metricas_custo}
    min_custo = {metrica: float('inf') for metrica in metricas_custo}

    # Iterando pelo dicionário para encontrar os valores máximos e mínimos
    for model_name, metrics in metrics_dict.items():
        for metrica in metricas_assertividade:
            valor = metrics['assertividade'][metrica]
            if valor > max_assertividade[metrica]:
                max_assertividade[metrica] = valor
            if valor < min_assertividade[metrica]:
                min_assertividade[metrica] = valor

        for metrica in metricas_custo:
            valor = metrics['custo'][metrica]
            if valor > max_custo[metrica]:
                max_custo[metrica] = valor
            if valor < min_custo[metrica]:
                min_custo[metrica] = valor

    # Convertendo os dicionários em listas simples
    max_assertividade_list = [max_assertividade[metrica] for metrica in metricas_assertividade]
    min_assertividade_list = [min_assertividade[metrica] for metrica in metricas_assertividade]

    max_custo_list = [max_custo[metrica] for metrica in metricas_custo]
    min_custo_list = [min_custo[metrica] for metrica in metricas_custo]

    return max_assertividade_list, min_assertividade_list, max_custo_list, min_custo_list

def calculo_maximum_minimum(metrics_per_iteration, metricas_a, metricas_c):
    # Se metrics_per_iteration não for uma lista, transforma-o em uma lista
    if isinstance(metrics_per_iteration, dict):
        metrics_per_iteration = [metrics_per_iteration]
    
    # Inicializar listas para armazenar valores das métricas
    valores_a = {metrica: [] for metrica in metricas_a}
    valores_c = {metrica: [] for metrica in metricas_c}
    
    # Iterar sobre as métricas recebidas em cada iteração e armazenar valores nas listas
    for metrics in metrics_per_iteration:
        for metrica in metricas_a:
            valores_a[metrica].append(float(metrics['assertividade'][metrica]))
        for metrica in metricas_c:
            valores_c[metrica].append(float(metrics['custo'][metrica]))
    
    # Usar numpy para calcular os valores máximos e mínimos
    max_metrics_a = [np.max(valores_a[metrica]) for metrica in metricas_a]
    min_metrics_a = [np.min(valores_a[metrica]) for metrica in metricas_a]
    
    max_metrics_c = [np.max(valores_c[metrica]) for metrica in metricas_c]
    min_metrics_c = [np.min(valores_c[metrica]) for metrica in metricas_c]
    
    # Retornar os valores máximos e mínimos das métricas de assertividade e custo como listas
    return max_metrics_a, min_metrics_a, max_metrics_c, min_metrics_c

def N(a_i, maximo, minimo):
    if maximo == minimo:
        return 0.0  # Evita divisão por zero, retorna 0 como valor normalizado padrão
    return (a_i - minimo) / (maximo - minimo)

def A(metrics, wa, metricas_a, maximos_a, minimos_a):
    print("max_assertividad_funcao_A: ", maximos_a)
    print("min_assertividade_funcao_A: ", minimos_a)
    a_i = [metrics["assertividade"][metrica] for metrica in metricas_a]
    return sum([N(a, maximo, minimo) * w for a, maximo, minimo, w in zip(a_i, maximos_a, minimos_a, wa)])

def C(metrics, wc, metricas_c, maximos_c, minimos_c):
    print("max_assertividad_funcao_C: ", maximos_c)
    print("min_assertividade_funcao_C: ", minimos_c)
    c_i = [metrics["custo"][metrica] for metrica in metricas_c]
    return sum([N(c, maximo, minimo) * w for c, maximo, minimo, w in zip(c_i, maximos_c, minimos_c, wc)])

def F(metrics, lbd, wa, wc, metrics_per_iteration, metricas_a, metricas_c, maximos_a, minimos_a, maximos_c, minimos_c):
    assertividade = A(metrics, wa, metricas_a, maximos_a, minimos_a)
    custo = C(metrics, wc, metricas_c, maximos_c, minimos_c)
    return lbd * assertividade - (1 - lbd) * custo
"""
def calculo_maximum_minimum(modelos, metricas_a, metricas_c):
    maximos_a = [np.max([modelo["assertividade"][metrica] for modelo in modelos]) for metrica in metricas_a]
    minimos_a = [np.min([modelo["assertividade"][metrica] for modelo in modelos]) for metrica in metricas_a]
    
    maximos_c = [np.max([modelo["custo"][metrica] for modelo in modelos]) for metrica in metricas_c]
    minimos_c = [np.min([modelo["custo"][metrica] for modelo in modelos]) for metrica in metricas_c]
    
    return maximos_a, minimos_a, maximos_c, minimos_c

def N(a_i, maximo, minimo):
    if maximo == minimo:
        return 0.0  # Evita divisão por zero, retorna 0 como valor normalizado padrão
    return (a_i - minimo) / (maximo - minimo)

# Funções de cálculo de A e C
def A(modelo, wa, modelos, metricas_a, maximos_a, minimos_a):
    a_i = [modelo["assertividade"][metrica] for metrica in metricas_a]
    return sum([N(a, maximo, minimo) * w for a, maximo, minimo, w in zip(a_i, maximos_a, minimos_a, wa)])

def C(modelo, wc, modelos, metricas_c, maximos_c, minimos_c):
    c_i = [modelo["custo"][metrica] for metrica in metricas_c]
    return sum([N(c, maximo, minimo) * w for c, maximo, minimo, w in zip(c_i, maximos_c, minimos_c, wc)])

# Função de cálculo do score Sphi
def F(modelo, lbd, wa, wc, modelos, metricas_a, metricas_c, maximos_a, minimos_a, maximos_c, minimos_c):
    assertividade = A(modelo, wa, modelos, metricas_a, maximos_a, minimos_a)
    custo = C(modelo, wc, modelos, metricas_c, maximos_c, minimos_c)
    return lbd * assertividade - (1 - lbd) * custo
    """