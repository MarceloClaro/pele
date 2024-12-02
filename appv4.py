import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet50, densenet121
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import streamlit as st
import gc
import logging
import base64
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import cv2
import io
import warnings
from datetime import datetime  # Importação para data e hora

# Supressão dos avisos relacionados ao torch.classes
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')

def set_seed(seed):
    """
    Define uma seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed para reprodutibilidade

# Definir as transformações para aumento de dados (aplicando transformações aleatórias)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    ], p=0.5),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Transformações para validação e teste
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset personalizado para classificação
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def seed_worker(worker_id):
    """
    Função para definir a seed em cada worker do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_data(dataset, classes):
    """
    Exibe algumas imagens do conjunto de dados com suas classes.
    """
    st.write("Visualização de algumas imagens do conjunto de dados:")
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)  # Converter a imagem PIL em array NumPy
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

def plot_class_distribution(dataset, classes):
    """
    Exibe a distribuição das classes no conjunto de dados e mostra os valores quantitativos.
    """
    # Extrair os rótulos das classes para todas as imagens no dataset
    labels = [label for _, label in dataset]

    # Criar um DataFrame para facilitar o plot com Seaborn
    df = pd.DataFrame({'Classe': labels})

    # Plotar o gráfico com as contagens
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Classe', data=df, ax=ax, palette="Set2", hue='Classe', dodge=False)

    # Definir ticks e labels
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)

    # Remover a legenda
    ax.get_legend().remove()

    # Adicionar as contagens acima das barras
    class_counts = df['Classe'].value_counts().sort_index()
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

    ax.set_title("Distribuição das Classes (Quantidade de Imagens)")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")

    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna o modelo pré-treinado selecionado para classificação.
    """
    if model_name == 'ResNet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == 'ResNet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == 'DenseNet121':
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
    else:
        st.error("Modelo não suportado.")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    if model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        st.error("Modelo não suportado.")
        return None

    model = model.to(device)
    return model

def apply_transforms_and_get_embeddings(dataset, model, transform, batch_size=16):
    """
    Aplica as transformações às imagens, extrai os embeddings e retorna um DataFrame.
    """
    def pil_collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pil_collate_fn)
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    augmented_images_list = []

    model_embedding = nn.Sequential(*list(model.children())[:-1])
    model_embedding.eval()
    model_embedding.to(device)

    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    index_pointer = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images_augmented = [transform(img) for img in images]
            images_augmented = torch.stack(images_augmented).to(device)
            embeddings = model_embedding(images_augmented)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
            embeddings_list.extend(embeddings)
            labels_list.extend(labels.numpy())
            augmented_images_list.extend([img.permute(1, 2, 0).numpy() for img in images_augmented.cpu()])
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
                batch_indices = indices[index_pointer:index_pointer + len(images)]
                file_paths = [dataset.dataset.samples[i][0] for i in batch_indices]
                file_paths_list.extend(file_paths)
                index_pointer += len(images)
            else:
                file_paths_list.extend(['N/A'] * len(labels))

    df = pd.DataFrame({
        'file_path': file_paths_list,
        'label': labels_list,
        'embedding': embeddings_list,
        'augmented_image': augmented_images_list
    })

    return df

def display_all_augmented_images(df, class_names, max_images=20):
    """
    Exibe todas as imagens augmentadas do DataFrame de forma organizada.
    """
    st.write(f"**Visualização das Primeiras {max_images} Imagens após Data Augmentation:**")
    df = df.head(max_images)
    num_images = len(df)
    if num_images == 0:
        st.write("Nenhuma imagem para exibir.")
        return
    
    cols_per_row = 5  # Número de colunas por linha
    rows = (num_images + cols_per_row - 1) // cols_per_row  # Calcula o número de linhas necessárias
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < num_images:
                image = df.iloc[idx]['augmented_image']
                label = df.iloc[idx]['label']
                with cols[col]:
                    st.image(image, caption=class_names[label], use_container_width=True)

def main():
    st.title("Classificação e Segmentação de Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar um modelo de classificação de imagens, aplicar algoritmos de clustering para análise comparativa e realizar segmentação de objetos.")

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, step=1, key="num_classes")
    model_name = st.sidebar.selectbox("Modelo Pré-treinado:", options=['ResNet18', 'ResNet50', 'DenseNet121'], key="model_name")
    fine_tune = st.sidebar.checkbox("Fine-Tuning Completo", value=False, key="fine_tune")
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=500, value=200, step=1, key="epochs")
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.0001, key="learning_rate")
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2, key="batch_size")
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="train_split")
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05, key="valid_split")

    # Upload do arquivo ZIP
    zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"], key="zip_file_uploader")
    if zip_file is not None and num_classes > 0 and train_split + valid_split <= 0.95:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data_dir = temp_dir

        # Treinar o modelo de classificação
        st.write("Iniciando o treinamento supervisionado...")
        model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
        if model_for_embeddings is None:
            st.error("Erro ao carregar o modelo.")
            return
        
        # Exibir Data Augmentation e Embeddings do Conjunto de Treinamento
        full_dataset = datasets.ImageFolder(root=data_dir)
        train_end = int(train_split * len(full_dataset))
        train_indices = list(range(train_end))
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        st.write("**Processando o conjunto de treinamento para incluir Data Augmentation e Embeddings...**")
        train_df = apply_transforms_and_get_embeddings(train_dataset, model_for_embeddings, train_transforms, batch_size=batch_size)
        st.write("**Dataframe do Conjunto de Treinamento com Data Augmentation e Embeddings:**")
        st.dataframe(train_df.drop(columns=['augmented_image']))
        display_all_augmented_images(train_df, full_dataset.classes)

    else:
        st.warning("Por favor, forneça os dados e as configurações corretas.")

if __name__ == "__main__":
    main()
