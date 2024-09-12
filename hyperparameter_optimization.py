"""
Implementação do Random Walk para otimização dos hiperparâmetros.
- Random Walk deve caminhar aleatoriamente por lr e na seleção do modelo a ser treinado a cada iteração
- A cada iteração deve ser armazenado as informações do modelo, hiperparametrização, métricas, score \sphi, score de assertividade e custo.
- As listas oace_metrics_per_iteraction e metrics_per_iteraction para indexar os dicionários por iteração e não por arquitetura
- Ao final, deve ser retornado os 5 melhores treinamentos durante todas a iterações
"""
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from metrics import *
from torchvision.models.inception import InceptionOutputs
import torch.nn.functional as F
from datasets import *
from models import get_resnet50, get_inception_v3, get_vgg19, get_efficientnet_b0, get_mobilenet_v2
from sklearn.metrics import accuracy_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_solution(models):
    """ Gera uma solução inicial para o Random Walk, em termos de lr e do modelo a ser usado"""
    lr = random.uniform(1e-4, 1e-2)
    model_index = random.randint(0, len(models) - 1)
    print("model: ", model_index)
    return [lr, model_index]

def random_walk_step(solution, models, step_size=0.1):
    """Executa um passo do Random Walk para explorar novas soluções"""
    new_solution = solution.copy()
    new_solution[0] = min(max(new_solution[0] + random.uniform(-step_size * new_solution[0], step_size * new_solution[0]), 1e-4), 1e-2) 
    new_solution[1] = random.randint(0, len(models) - 1) 
    return new_solution

def train_models(model, trainLoader, validLoader, criterion, optimizer, epochs=10, early_stopping_rounds=5):
    """Treina o modelo e monitora a perda de validação para aplicar early stopping."""
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):
        
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)
            
            # Redimensionamento dinâmico se necessário
            if isinstance(model, models.Inception3):
                data = F.interpolate(data, size=(299, 299), mode='bilinear', align_corners=False)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Verifica se a saída é do tipo InceptionOutputs
            if isinstance(output, InceptionOutputs):
                output = output.logits  # Usa apenas a saída principal
              
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        with torch.no_grad():
            for data, target in validLoader:
                data, target = data.to(device), target.to(device)
                
                 # Redimensionamento dinâmico se necessário
                if isinstance(model, models.Inception3):
                    data = F.interpolate(data, size=(299, 299), mode='bilinear', align_corners=False)
                
                output = model(data)
                # Verifica se a saída é do tipo InceptionOutputs
                if isinstance(output, InceptionOutputs):
                    output = output.logits  # Usa apenas a saída principal
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss/len(trainLoader.dataset)
        valid_loss = valid_loss/len(validLoader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f"Early stopping after {epoch} epochs due to no improvement.")
            break

    return model

def evaluate_solution(model, trainLoader, testLoader, validLoader, criterion, optimizer):
    """
    - Avalia uma solução de modelo treinado medindo sua precisão, acurácia, recall, tempo de inferência, tamanho e número de parâmetros.
    - Esta função treina o modelo, realiza inferências no conjunto de testes e calcula várias métricas de desempenho, incluindo assertividade e custo computacional.
    """
    model = train_models(model, trainLoader, validLoader, criterion, optimizer)
    
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, target in testLoader:
            inputs, target = inputs.to(device), target.to(device)
            
            # Redimensionamento dinâmico se necessário
            if isinstance(model, models.Inception3):
                inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
            
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            inference_times.append(inference_time)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)# No dataset char x-ray average binary deve ser utilizada, para o restanto, micro ou average
    
    #print(f"Precision: {precision} \t Recall: {recall} \t Accuracy: {accuracy}") 
    #conf_matrix = confusion_matrix(all_labels, all_preds)
    #print(f"Confusion Matrix:\n{conf_matrix}")
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

    return precision, accuracy, recall, avg_inference_time, model_size, num_params

def optimize_hyperparameters(models, trainloader, testloader, validLoader, classes, lbd, wa, wc, max_iterations=20):
    """Otimiza os hiperparâmetros dos modelos utilizando Random Walk e retorna as melhores soluções."""
    best_model = None
    best_score = float('-inf')
    best_solution = None

    metrics_per_iteration = {iteration: {} for iteration in range(1, max_iterations + 1)}
    oace_metrics_per_iteration = {iteration: {} for iteration in range(1, max_iterations + 1)}
    
    accumulated_assertiveness_metrics = []
    
    with open('warm_up_metrics.pkl', 'rb') as f:
            warm_up_metrics = pickle.load(f)
        
    print("Resultado das Métricas no Aquecimento: ", warm_up_metrics)
        
    maximos_c, minimos_c = get_max_min_metrics(warm_up_metrics)
    solution = generate_solution(models)

    for iteration in range(1, max_iterations + 1):
        
        print(f"Iteration {iteration}: model {solution[1]} e lr {solution[0]}")

        model_name, Model = models[solution[1]]
        model = Model(num_classes=len(classes)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=solution[0])
        criterion = nn.CrossEntropyLoss()

        precision, accuracy, recall, avg_inference_time, model_size, num_params = evaluate_solution(
            model, trainloader, testloader, validLoader, criterion, optimizer)

        # Armazena as métricas da iteração atual
        current_metrics = {
            "model_name": model_name,
            "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
            "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
            "solution": {"lr": solution[0], "model_index": solution[1]}
        }
        
        # Adiciona as métricas atuais à lista de métricas acumuladas
        accumulated_assertiveness_metrics.append(current_metrics)
        # Atualiza o dicionário com as métricas da iteração atual
        metrics_per_iteration[iteration] = current_metrics
        print("Métricas por Iteração: ", metrics_per_iteration)

        maximos_a, minimos_a = calculo_maximum_minimum(accumulated_assertiveness_metrics, 
                                                       ["precision", "accuracy", "recall"])
        
        print("max_assertividade: ", maximos_a)
        print("min_assertividade: ", minimos_a)
        print("max_custo: ", maximos_c)
        print("min_custo: ", minimos_c)
        
        a_value = A(metrics_per_iteration[iteration], wa, ["precision", "accuracy", "recall"], maximos_a, minimos_a)
        c_value = C(metrics_per_iteration[iteration], wc, ["mtp", "tpi", "ms"], maximos_c, minimos_c)
        score = F_score(lbd, a_value, c_value)

        oace_metrics_per_iteration[iteration] = {
            "model_name": model_name, 
            "A": a_value,
            "C": c_value,
            "Score": score,
            "solution": {"lr": solution[0], "model_index": solution[1]}
        }
        
        print("Resultado do OACE por Iteração: ", oace_metrics_per_iteration)

        if score > best_score:
            best_score = score
            best_model = model_name
            best_solution = solution.copy()

        solution = random_walk_step(solution, models)

    return best_model, best_score, best_solution, metrics_per_iteration, oace_metrics_per_iteration

def warm_calculate_metrics(model, dataloader, device):
    """Calcula as métricas de custo no aquecimento para um modelo dado"""
    model.eval()
    total_inference_time = 0.0

    inference_times = []

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024**2)  # Size in MB

    return num_params, avg_inference_time, model_size

def warm_up_models(models, dataloader, device):
    metrics = {}
    for model_name, get_model_func in models:
        model = get_model_func().to(device)
        num_params, avg_inference_time, model_size = warm_calculate_metrics(model, dataloader, device)
        metrics[model_name] = {
            'mtp': num_params,
            'tpi': avg_inference_time,
            'ms': model_size
        }
    return metrics