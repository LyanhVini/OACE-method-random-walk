"""
Validação dos modelos    
"""
from main import *


def summarize_scores(oace_metrics_per_iteration):
    summary = {}
    
    for model, iterations in oace_metrics_per_iteration.items():
        scores = [metrics["Score"] for metrics in iterations.values()]
        solutions = {metrics["Score"]: metrics["Solution"] for metrics in iterations.values()}
        
        if scores:
            best_sc = max(scores)
            worst_sc = min(scores)
            average_sc = sum(scores) / len(scores)
            best_solution = solutions[best_sc]
            worst_solution = solutions[worst_sc]
        else:
            best_sc = worst_sc = average_sc = None
            best_solution = worst_solution = None
            
        summary[model] = {
            'Best Score': best_sc,
            'Average Score': average_sc,
            'Worst Score': worst_sc,
            'Best Solution': best_solution,
            'Worst Solution': worst_solution
        }
        
    return summary

def plot_convergence(oace_metrics_per_iteration):
    pass

def plot_train_model(metrics_per_iteration):
    pass

