"""
Definição e treinamento dos modelos.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BeitModel, BeitConfig, BeitForImageClassification

def get_efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes=2):
    
    model = models.efficientnet_b0(weights=weights)
    print(model)
    #model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # Verificar a estrutura do classificador
    if hasattr(model.classifier, 'in_features'):
        in_features = model.classifier.in_features
    elif hasattr(model.classifier, 'fc'):
        in_features = model.classifier.fc.in_features
    else:
        in_features = model.classifier[-1].in_features
    
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model

def get_mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, num_classes=2):
    model = models.mobilenet_v2(weights=weights)
    print(model)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_resnet50(weights=models.ResNet50_Weights.DEFAULT, num_classes=2):
    model = models.resnet50(weights=weights)
    print(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, num_classes=10):
    model = models.inception_v3(weights=weights, aux_logits=True)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(model)
    return model

def get_vit(weights=models.ViT_B_32_Weights.IMAGENET1K_V1, num_classes=10):
    model = models.vit_b_32(weights=weights)
    print(model)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def get_beit(weights='microsoft/beit-base-patch16-224', num_classes=10):
    #config = BeitConfig.from_pretrained(weights)
    #model = BeitModel(config)
    #model.classifier = nn.Linear(model.pooler.dense.out_features, num_classes)
    # Carregar configuração
    config = BeitConfig.from_pretrained(weights)
    # Ajustar número de classes no classificador
    config.num_labels = num_classes
    # Carregar modelo com a configuração ajustada
    model = BeitForImageClassification(config)
    print(model)
    return model

def get_vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1, num_classes=10):
    model = models.vgg19(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    print(model)
    return model


