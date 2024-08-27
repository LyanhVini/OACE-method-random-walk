"""
Script principal para executar todo o pipeline
Códigos de Auxílio:
    - https://github.com/rasbt/deeplearning-models/blob/master/pytorch-lightning_ipynb/cnn/cnn-alexnet-grouped-cifar10.ipynb
    - https://github.com/simoninithomas/cifar-10-classifier-pytorch/blob/master/PyTorch%20Cifar-10%20Classifier.ipynb
    - https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""
from datasets import *
from models import *
from metrics import *
from hyperparameter_optimization import *
from validation import *
import json
import torchvision.models as models
import pickle


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CHEST X-RAY:
    train_dir = 'datasets/chest_xray/train'
    val_dir = 'datasets/chest_xray/val'
    test_dir = 'datasets/chest_xray/test'
    trainLoader_1, validLoader_1, testLoader_1, classes_1 = chest_x_ray(train_dir, val_dir, test_dir)
    # CIFAR-10:
    trainLoader_2, validLoader_2, testLoader_2, classes_2 = cifar_10(batch_size=20)
    # TrashNet:
    dataset_dir = 'datasets/dataset-resized'
    trainLoader_3, validLoader_3, testLoader_3, classes_3 = trashNet(dataset_dir)
    
    models = [
    ("EfficientNetB0", get_efficientnet_b0),
    ("MobileNetV2", get_mobilenet_v2),
    ("ResNet50", get_resnet50),
    ("InceptionV3", get_inception_v3), 
    ("ViT", get_vit),
    ("VGG19", get_vgg19)
    ]

    lbd = 0.7 # lambda do método
    wa = [0.731, 0.188, 0.081] # Precision, Acuraccy, Recall
    wc = [0.731, 0.188, 0.081] # MTP, TPI, MSn
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    subsample_loader, classes = trashNet_subsample(dataset_dir)
    
    print("Classes:", classes)
    for images, labels in subsample_loader:
        print("Batchs of images shape:", images.shape)
        print("Batch of labels shape:", labels.shape)
        break
    
    warm_up_metrics = warm_up_models(models, subsample_loader, device)
    
    with open('warm_up_metrics.pkl', 'wb') as f:
        pickle.dump(warm_up_metrics, f)
    
    # Otimização de hiperparâmetros
    best_model_1, best_score_1, best_solution_1, metrics_per_iteration_1, oace_metrics_per_iteration_1 = optimize_hyperparameters(
                                                                                                                                models, trainLoader_1, testLoader_1, validLoader_1, classes_1, lbd, wa, wc)
    
    print(f"Best Model: {best_model_1}")
    print(f"Best Score: {best_score_1}")
    print(f"Best Solution: {best_solution_1}")
    
    with open('metrics_per_iteration.json', 'w') as f:
        json.dump(metrics_per_iteration_1, f)
            
    with open('oace_metrics_per_iteration.json', 'w') as f:
        json.dump(oace_metrics_per_iteration_1, f)
        
    """
    best_model_2, best_score_2, best_solution_2, metrics_per_iteration_2, oace_metrics_per_iteration_2 = optimize_hyperparameters(
                                                                                                                                models, trainLoader_2, testLoader_2, validLoader_2, classes_2, lbd, wa, wc)
    
    print(f"Best Model: {best_model_2}")
    print(f"Best Score: {best_score_2}")
    print(f"Best Solution: {best_solution_2}")
    
    best_model_3, best_score_3, best_solution_3, metrics_per_iteration_3, oace_metrics_per_iteration_3 = optimize_hyperparameters(
                                                                                                                                models, trainLoader_3, testLoader_3, validLoader_3, classes_3, lbd, wa, wc)
    
    print(f"Best Model: {best_model_3}")
    print(f"Best Score: {best_score_3}")
    print(f"Best Solution: {best_solution_3}")
    
    sumary_scores_1 = summarize_scores(oace_metrics_per_iteration_1)
    sumary_scores_2 = summarize_scores(oace_metrics_per_iteration_2)
    sumary_scores_3 = summarize_scores(oace_metrics_per_iteration_3)
    
    with open('metrics_per_iteration.json', 'w') as f:
        json.dump(metrics_per_iteration_1, f)
        
    with open('metrics_per_iteration.json', 'w') as f:
        json.dump(metrics_per_iteration_2, f)
    
    with open('metrics_per_iteration.json', 'w') as f:
        json.dump(metrics_per_iteration_3, f)

    with open('oace_metrics_per_iteration.json', 'w') as f:
        json.dump(oace_metrics_per_iteration_1, f)
    
    with open('oace_metrics_per_iteration.json', 'w') as f:
        json.dump(oace_metrics_per_iteration_2, f)
    
    with open('oace_metrics_per_iteration.json', 'w') as f:
        json.dump(oace_metrics_per_iteration_3, f)
        
    with open('sumary_scores.json', 'w') as f:
        json.dump(sumary_scores_1, f)
        
    with open('sumary_scores.json', 'w') as f:
        json.dump(sumary_scores_2, f)
    
    with open('sumary_scores.json', 'w') as f:
        json.dump(sumary_scores_3, f)
    """

if __name__ == "__main__":    
    main()