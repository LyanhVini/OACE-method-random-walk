"""
Implementação das métricas e funções para cálculo do método OACE
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
    """Calcula os máximos e mínimos das métricas de custo resultante do aquecimento dos modelos"""
    metricas_custo = ['mtp', 'tpi', 'ms']
    
    max_custo = {metrica: float('-inf') for metrica in metricas_custo}
    min_custo = {metrica: float('inf') for metrica in metricas_custo}

    for model_name, metrics in metrics_dict.items():
        for metrica in metricas_custo:
            valor = metrics[metrica]
            if valor > max_custo[metrica]:
                max_custo[metrica] = valor
            if valor < min_custo[metrica]:
                min_custo[metrica] = valor

    # Convertendo os dicionários de máximos e mínimos em listas simples
    max_custo_list = [max_custo[metrica] for metrica in metricas_custo]
    min_custo_list = [min_custo[metrica] for metrica in metricas_custo]

    return max_custo_list, min_custo_list

def calculo_maximum_minimum(metrics_list, metricas_a):
    """Calcula os máximos e mínimos das métricas de assertividade obtida a cada iteração"""
    valores_a = {metrica: [] for metrica in metricas_a}
    
    # Iterar sobre as métricas recebidas em cada iteração e armazenar valores nas listas
    for metrics in metrics_list:
        for metrica in metricas_a:
            valores_a[metrica].append(float(metrics['assertividade'][metrica]))
    
    # Usar numpy para calcular os valores máximos e mínimos
    max_metrics_a = [np.max(valores_a[metrica]) for metrica in metricas_a]
    min_metrics_a = [np.min(valores_a[metrica]) for metrica in metricas_a]
    
    return max_metrics_a, min_metrics_a

def N(a_i, maximo, minimo):
    """Função para normalização min() and max()"""
    if maximo == minimo:
        return 0.0  # Evita divisão por zero, retorna 0 como valor normalizado padrão
    return (a_i - minimo) / (maximo - minimo)

def A(metrics, wa, metricas_a, maximos_a, minimos_a):
    """Função de Assertividade do Método"""
    print("max_assertiveness_A: ", maximos_a)
    print("min_assertiveness_A: ", minimos_a)
    a_i = [metrics["assertividade"][metrica] for metrica in metricas_a]
    return sum([N(a, maximo, minimo) * w for a, maximo, minimo, w in zip(a_i, maximos_a, minimos_a, wa)])

def C(metrics, wc, metricas_c, maximos_c, minimos_c):
    """Função de Custo do Método"""
    print("max_cost_C: ", maximos_c)
    print("min_cost_C: ", minimos_c)
    c_i = [metrics["custo"][metrica] for metrica in metricas_c]
    return sum([N(c, maximo, minimo) * w for c, maximo, minimo, w in zip(c_i, maximos_c, minimos_c, wc)])

def F_score(lbd, assertiveness, cost):
    """Função Objetivo do Método"""
    return lbd * assertiveness - (1 - lbd) * cost