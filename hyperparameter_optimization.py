"""
Implementação do Random Walk para otimização dos hiperparâmetros.
- Random Walk deve caminhar aleatoriamente por lr e na seleção do modelo a ser treinado a cada iteração
- A cada iteração deve ser armazenado as informações do modelo, hiperparametrização, métricas, score \sphi, score de assertividade e custo.
- As listas oace_metrics_per_iteraction e metrics_per_iteraction para indexar os dicionários por iteração e não por arquitetura
- Ao final, deve ser retornado os 5 melhores treinamentos durante todas a iterações
"""

import random
import time
import torch
import os
import torch.nn as nn
from torchvision import datasets
import torch.optim as optim
from metrics import *
from datasets import *
from tqdm import tqdm
from models import get_resnet50, get_inception_v3, get_beit, get_vgg19, get_efficientnet_b0, get_mobilenet_v2
from sklearn.metrics import accuracy_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_solution(models):
    lr = random.uniform(1e-4, 1e-2)
    model_index = random.randint(0, len(models) - 1)
    print("model: ", model_index)
    return [lr, model_index]

def random_walk_step(solution, models, step_size=0.1):
    new_solution = solution.copy()
    new_solution[0] = min(max(new_solution[0] + random.uniform(-step_size * new_solution[0], step_size * new_solution[0]), 1e-4), 1e-2)  # lr
    new_solution[1] = random.randint(0, len(models) - 1)  # Randomly pick a model
    return new_solution

def train_models(model, trainLoader, validLoader, criterion, optimizer, epochs=5, early_stopping_rounds=5):
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        with torch.no_grad():
            for data, target in validLoader:
                data, target = data.to(device), target.to(device)
                output = model(data)
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

def evaluate_solution(model, trainLoader, testLoader, validLoader, classes, criterion, optimizer, epochs=1):

    model = train_models(model, trainLoader, validLoader, criterion, optimizer, epochs)
    
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, target in testLoader:
            inputs, target = inputs.to(device), target.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            inference_times.append(inference_time)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

    return precision, accuracy, recall, avg_inference_time, model_size, num_params

def optimize_hyperparameters(models, trainloader, testloader, validLoader, classes, lbd, wa, wc, max_iterations=100):
    best_model = None
    best_score = float('-inf')
    best_solution = None

    #results = {model_name: [] for model_name, _ in models}
    metrics_per_iteration = {iteration: {} for iteration in range(1, max_iterations + 1)}
    #oace_metrics_per_iteration = {model_name: {} for model_name, _ in models}
    oace_metrics_per_iteration = {iteration: {} for iteration in range(1, max_iterations + 1)}

    num_models = len(models)
    solution = generate_solution(models)

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration {iteration}")

        model_name, Model = models[solution[1]]
        model = Model(num_classes=len(classes)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=solution[0])
        criterion = nn.CrossEntropyLoss()

        precision, accuracy, recall, avg_inference_time, model_size, num_params = evaluate_solution(
            model, trainloader, testloader, validLoader, classes, criterion, optimizer)

        #if iteration not in metrics_per_iteration[model_name]:
        #    metrics_per_iteration[model_name][iteration] = {}

        metrics_per_iteration[iteration] = {
            "model_name": model_name,
            "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
            "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
            "solution": {"lr": solution[0], "model_index": solution[1]}
        }
        """
        maximos_a, minimos_a, maximos_c, minimos_c = calculo_maximum_minimum(
            [metrics for model_metrics in metrics_per_iteration.values() for metrics in model_metrics.values()],
            ["precision", "accuracy", "recall"], ["mtp", "tpi", "ms"])
        a_value = A(metrics_per_iteration[model_maximos_a = {metrica: np.max([metrics["assertividade"][metrica] for metrics in metrics_per_iteration.values()]) for metrica in metricas_a}
            minimos_a = {metrica: np.min([metrics["assertividade"][metrica] for metrics in metrics_per_iteration.values()]) for metrica in metricas_a}
            
            # Calcule os máximos e mínimos para cada métrica de custo
            maximos_c = {metrica: np.max([metrics["custo"][metrica] for metrics in metrics_per_iteration.values()]) for metrica in metricas_c}
            minimos_c = {metrica: np.min([metrics["custo"][metrica] for metrics in metrics_per_iteration.values()]) for metrica in metricas_c}name][iteration], wa, metrics_per_iteration, ["precision", "accuracy", "recall"], maximos_a, minimos_a)
        c_value = C(metrics_per_iteration[model_name][iteration], wc, metrics_per_iteration, ["mtp", "tpi", "ms"], maximos_c, minimos_c)
        score = F(metrics_per_iteration[model_name][iteration], lbd, wa, wc, metrics_per_iteration, ["precision", "accuracy", "recall"],
                  ["mtp", "tpi", "ms"], maximos_a, minimos_a, maximos_c, minimos_c)
        """
        print(metrics_per_iteration)
        
        #maximos_a, minimos_a, maximos_c, minimos_c = calculo_maximum_minimum(metrics_per_iteration[iteration],
        #                                                                   ["precision", "accuracy", "recall"], ["mtp", "tpi", "ms"])
        with open('warm_up_metrics.pkl', 'rb') as f:
            warm_up_metrics = pickle.load(f)
        
        print("Metrics 2: ", warm_up_metrics)
        
        maximos_a, minimos_a, maximos_c, minimos_c = get_max_min_metrics(warm_up_metrics)
        print("max_assertividade: ", maximos_a)
        print("min_assertividade: ", minimos_a)
        print("max_custo: ", maximos_c)
        print("min_custo: ", minimos_c)
        a_value = A(metrics_per_iteration[iteration], wa, ["precision", "accuracy", "recall"], maximos_a, minimos_a)
        c_value = C(metrics_per_iteration[iteration], wc, ["mtp", "tpi", "ms"], maximos_c, minimos_c)
        score = F(metrics_per_iteration[iteration], lbd, wa, wc, metrics_per_iteration, ["precision", "accuracy", "recall"],
                  ["mtp", "tpi", "ms"], maximos_a, minimos_a, maximos_c, minimos_c)
        
        #if iteration not in oace_metrics_per_iteration[model_name]:
        #    oace_metrics_per_iteration[model_name][iteration] = {}

        oace_metrics_per_iteration[iteration] = {
            "model_name": model_name, 
            "A": a_value,
            "C": c_value,
            "Score": score,
            "solution": {"lr": solution[0], "model_index": solution[1]}
        }

        if score > best_score:
            best_score = score
            best_model = model_name
            best_solution = solution.copy()

        solution = random_walk_step(solution, models)

    return best_model, best_score, best_solution, metrics_per_iteration, oace_metrics_per_iteration

def warm_calculate_metrics(model, dataloader, device):
    """Função para fazer o aquecimento dos modelos e capturar o resultado das métricas"""
    model.eval()
    total_inference_time = 0.0
    num_samples = len(dataloader)
    
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024**2)  # Size in MB
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall, avg_inference_time, num_params, model_size

def warm_up_models(models, dataloader, device):
    metrics = {}
    for model_name, get_model_func in models:
        model = get_model_func().to(device)
        accuracy, precision, recall, avg_inference_time, num_params, model_size = warm_calculate_metrics(model, dataloader, device)
        metrics[model_name] = {
            'assertividade': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            },
            'custo': {
                'tpi': avg_inference_time,
                'mtp': num_params,
                'ms': model_size
            }
        }
    return metrics
"""
def generate_solution():
    lr = random.uniform(1e-4, 1e-2)
    return [lr]

def random_walk_step(solution, step_size=0.1):
    new_solution = solution.copy()
    new_solution[0] = min(max(new_solution[0] + random.uniform(-step_size * new_solution[0], step_size * new_solution[0]), 1e-4), 1e-2)  # lr
    return new_solution

def train_models(model, trainLoader, validLoader, criterion, optimizer, epochs=10, early_stopping_rounds=5):
    
    best_val_loss = np.Inf # track change in validation loss
    no_improvement_count = 0

    for epoch in range(1, epochs+1):

        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in trainLoader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in validLoader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
        
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
            print("Early stopping devido à falta de melhora na perda de validação.")
            break
        
        # save model if validation loss has decreased
        #if valid_loss <= valid_loss_min:
        #    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #    valid_loss_min,
        #    valid_loss))
        #    torch.save(model.state_dict(), 'model_cifar.pt')
        #    valid_loss_min = valid_loss
        
    return model

def evaluate_solution(model, trainLoader, testLoader, validLoader, classes, criterion, optimizer, epochs=20):

    verify_dataset(trainLoader, validLoader, testLoader)
    
    model = train_models(model, trainLoader, validLoader, criterion, optimizer, epochs)
    
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        #for i, data in enumerate(testLoader, 0):#testLoader
        for inputs, target in testLoader:
            #inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            inference_times.append(inference_time)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)  # em MB

    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'Accuracy: {accuracy:.6f}')
    print(f'Average Inference Time: {avg_inference_time:.6f} seconds')
    print(f'Model Size: {model_size:.2f} MB')
    print(f'Number of Parameters: {num_params:.2f} million')
    
    return precision, accuracy, recall, avg_inference_time, model_size, num_params

def optimize_hyperparameters(models, trainloader, testloader, validLoader, classes, lbd, wa, wc, max_no_improvement=10, max_iterations=10):
    
    best_model = None
    best_score = float('-inf')
    best_solution = None

    results = {model_name: [] for model_name, _ in models}
    metrics_per_iteration = {model_name: {} for model_name, _ in models}
    oace_metrics_per_iteration = {model_name: {} for model_name, _ in models}

    initial_solution = generate_solution()

    for model_name, Model in models:
        print(f"Optimizing hyperparameters for {model_name}...")
        best_model_score = float('-inf')
        best_model_solution = None

        solutions = []
        oace_scores = []

        solution = initial_solution.copy()
        solutions.append(solution)
        print("solution: ", solution)
        model = Model(num_classes=len(classes)).to(device)#10
        #optimizer = optim.SGD(model.parameters(), lr=solution[0])
        optimizer = optim.Adam(model.parameters(), lr=solution[0])
        criterion = nn.CrossEntropyLoss()

        # Placeholder para métricas do dataset
        modelos = []

        # Avaliação da solução inicial
        precision, accuracy, recall, avg_inference_time, model_size, num_params = evaluate_solution(
            model, trainloader, testloader, validLoader, classes, criterion, optimizer)
        modelos.append({
            "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
            "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
        })

        # Armazenar métricas da iteração 1
        metrics_per_iteration[model_name][1] = {
            "assertividade": {"prec": precision, "acc": accuracy, "recall": recall},
            "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
            "solution": {"lr": solution[0]}
        }
        
        print("metrics_per_iteration: ", metrics_per_iteration)

        maximos_a, minimos_a, maximos_c, minimos_c = calculo_maximum_minimum(
            modelos, ["precision", "accuracy", "recall"], ["mtp", "tpi", "ms"])
        a_value = A(modelos[-1], wa, modelos, ["precision", "accuracy", "recall"], maximos_a, minimos_a)
        c_value = C(modelos[-1], wc, modelos, ["mtp", "tpi", "ms"], maximos_c, minimos_c)
        score = F(modelos[-1], lbd, wa, wc, modelos, ["precision", "accuracy", "recall"],
                  ["mtp", "tpi", "ms"], maximos_a, minimos_a, maximos_c, minimos_c)
        oace_scores.append(score)

        # Armazenar métricas OACE da iteração 1
        oace_metrics_per_iteration[model_name][1] = {
            "A": a_value,
            "C": c_value,
            "Score": score,
            "Solution": {"lr": solution[0]}
        }
        
        print("oace_metrics_per_iteration: ", oace_metrics_per_iteration)

        no_improvement_count = 0

        for t in range(2, max_iterations + 1):
            new_solution = random_walk_step(solution)
            solutions.append(new_solution)
            model = Model(num_classes=classes).to(device)
            optimizer = optim.SGD(model.parameters(), lr=new_solution[0])
            precision, accuracy, recall, avg_inference_time, model_size, num_params = evaluate_solution(
                model, trainloader, testloader, validLoader, classes, criterion, optimizer)
            modelos.append({
                "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
                "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size}
            })

            # Armazenar métricas da iteração t
            metrics_per_iteration[model_name][t] = {
                "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
                "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
                "solution": {"lr": new_solution[0]}
            }
            
            print("metrics_per_iteration*: ", metrics_per_iteration)

            maximos_a, minimos_a, maximos_c, minimos_c = calculo_maximum_minimum(
                modelos, ["precision", "accuracy", "recall"], ["mtp", "tpi", "ms"])
            a_value = A(modelos[-1], wa, modelos, ["precision", "accuracy", "recall"], maximos_a, minimos_a)
            c_value = C(modelos[-1], wc, modelos, ["mtp", "tpi", "ms"], maximos_c, minimos_c)
            new_score = F(modelos[-1], lbd, wa, wc, modelos, ["precision", "accuracy", "recall"],
                          ["mtp", "tpi", "ms"], maximos_a, minimos_a, maximos_c, minimos_c)
            oace_scores.append(new_score)

            # Armazenar métricas OACE da iteração t
            oace_metrics_per_iteration[model_name][t] = {
                "A": a_value,
                "C": c_value,
                "Score": new_score,
                "Solution": {"lr": new_solution[0]}
            }
            print("oace_metrics_per_iteration*: ", oace_metrics_per_iteration)

            if new_score > best_model_score:
                best_model_score = new_score
                best_model_solution = new_solution
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                print(f"Stopping optimization for {model_name} after {t} iterations without improvement.")
                break

            solution = new_solution

        results[model_name] = oace_scores
        
        # Criar a condição para cada caso, o melhor modelo 
        # No final deve retornar o melhor score, a média e o pior score para todos os modelos, seguida da hiperparametrização ou solução para o melhor e o pior score

        if best_model_score > best_score:
            best_score = best_model_score
            best_model = model_name
            best_solution = best_model_solution

    return best_model, best_score, best_solution, results, metrics_per_iteration, oace_metrics_per_iteration
"""