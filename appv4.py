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
import json
import uuid  # Para gerar identificadores únicos

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
    # As linhas abaixo são recomendadas para garantir reprodutibilidade
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
    try:
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
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

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

    # Extrair todas as camadas exceto a última para obter embeddings
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

def train_model(train_loader, valid_loader, model, criterion, optimizer, epochs, patience):
    """
    Função principal para treinamento do modelo de classificação.
    """
    set_seed(42)  # Garantir reprodutibilidade

    # Inicializar variáveis para Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    # Listas para armazenar perdas e acurácias
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Placeholders para gráficos dinâmicos
    placeholder = st.empty()
    progress_bar = st.progress(0)
    epoch_text = st.empty()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
                return None

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validação
        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(valid_loader.dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_loader.dataset)
        valid_losses.append(valid_epoch_loss)
        valid_accuracies.append(valid_epoch_acc.item())

        # Atualizar gráficos dinamicamente
        with placeholder.container():
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Gráfico de Perda
            ax[0].plot(range(1, len(train_losses) + 1), train_losses, label='Treino')
            ax[0].plot(range(1, len(valid_losses) + 1), valid_losses, label='Validação')
            ax[0].set_title(f'Perda por Época ({timestamp})')
            ax[0].set_xlabel('Épocas')
            ax[0].set_ylabel('Perda')
            ax[0].legend()

            # Gráfico de Acurácia
            ax[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Treino')
            ax[1].plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label='Validação')
            ax[1].set_title(f'Acurácia por Época ({timestamp})')
            ax[1].set_xlabel('Épocas')
            ax[1].set_ylabel('Acurácia')
            ax[1].legend()

            st.pyplot(fig)
            plt.close(fig)  # Fechar a figura para liberar memória

        # Atualizar texto de progresso
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f'Época {epoch+1}/{epochs}')

        # Early Stopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write('Early stopping!')
                if best_model_wts is not None:
                    model.load_state_dict(best_model_wts)
                break

    # Carregar os melhores pesos do modelo se houver
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Gráficos de Perda e Acurácia finais
    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)

    # Liberar memória
    del train_loader, valid_loader
    gc.collect()

    return model

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plota os gráficos de perda e acurácia e exibe no Streamlit.
    """
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Gráfico de Perda
    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title(f'Perda por Época ({timestamp})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Gráfico de Acurácia
    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title(f'Acurácia por Época ({timestamp})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

def evaluate_model(model, test_loader, classes):
    """
    Avalia o modelo no conjunto de teste e exibe métricas.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Relatório de Classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.text("Relatório de Classificação:")
    st.write(report_df)

    # Salvar relatório de classificação
    report_filename = f'classification_report.json'
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=4)
    st.write(f"Relatório de classificação salvo como `{report_filename}`")

    # Disponibilizar para download
    unique_id = uuid.uuid4()
    with open(report_filename, "rb") as file:
        st.download_button(
            label="Download do Relatório de Classificação",
            data=file,
            file_name=report_filename,
            mime="application/json",
            key=f"download_classification_report_{unique_id}"
        )

    # Matriz de Confusão Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão Normalizada')
    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

    # Salvar matriz de confusão
    cm_filename = f'confusion_matrix.png'
    fig.savefig(cm_filename)
    st.write(f"Matriz de Confusão Normalizada salva como `{cm_filename}`")

    # Disponibilizar para download
    with open(cm_filename, "rb") as file:
        st.download_button(
            label="Download da Matriz de Confusão",
            data=file,
            file_name=cm_filename,
            mime="image/png",
            key=f"download_confusion_matrix_{unique_id}"
        )

    # Curva ROC
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)  # Fechar a figura para liberar memória

        # Salvar Curva ROC
        roc_filename = f'roc_curve.png'
        fig.savefig(roc_filename)
        st.write(f"Curva ROC salva como `{roc_filename}`")

        # Disponibilizar para download
        with open(roc_filename, "rb") as file:
            st.download_button(
                label="Download da Curva ROC",
                data=file,
                file_name=roc_filename,
                mime="image/png",
                key=f"download_roc_curve_{unique_id}"
            )
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")

        # Salvar AUC-ROC
        auc_filename = f'auc_roc.txt'
        with open(auc_filename, 'w') as f:
            f.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")
        st.write(f"AUC-ROC Média Ponderada salvo como `{auc_filename}`")

        # Disponibilizar para download
        with open(auc_filename, "rb") as file:
            st.download_button(
                label="Download do AUC-ROC",
                data=file,
                file_name=auc_filename,
                mime="text/plain",
                key=f"download_auc_roc_{unique_id}"
            )

    # Calcule as métricas de desempenho
    accuracy = report.get('accuracy', 0)
    precision = report.get('weighted avg', {}).get('precision', 0)
    recall = report.get('weighted avg', {}).get('recall', 0)
    f1_score = report.get('weighted avg', {}).get('f1-score', 0)
    # 'roc_auc' já foi calculado acima

    # Retornar as métricas em um dicionário
    metrics = {
        'Model': 'Model_Name',  # Substituir conforme necessário
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'ROC_AUC': roc_auc if 'roc_auc' in locals() else np.nan
    }

    # Salvar métricas em arquivo JSON
    metrics_filename = f'metrics.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    st.write(f"Métricas salvas como `{metrics_filename}`")

    # Disponibilizar para download
    with open(metrics_filename, "rb") as file:
        st.download_button(
            label="Download das Métricas",
            data=file,
            file_name=metrics_filename,
            mime="application/json",
            key=f"download_metrics_{uuid.uuid4()}"
        )

def main():
    st.title("Classificação de Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar modelos de classificação de imagens utilizando PyTorch e Streamlit.")

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, step=1, key="num_classes")
    model_name = st.sidebar.selectbox("Modelo Pré-treinado:", options=['ResNet18', 'ResNet50', 'DenseNet121'], key="model_name")
    fine_tune = st.sidebar.checkbox("Fine-Tuning Completo", value=False, key="fine_tune")
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=500, value=20, step=1, key="epochs")
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001, key="learning_rate")
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2, key="batch_size")
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="train_split")
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05, key="valid_split")
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01, key="l2_lambda")
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1, key="patience")
    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada para Classes Desbalanceadas", value=False, key="use_weighted_loss")

    # Upload do arquivo ZIP
    zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"], key="zip_file_uploader")
    if zip_file is not None and num_classes > 0 and train_split + valid_split <= 0.95:
        try:
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            data_dir = temp_dir

            # Carregar o dataset
            full_dataset = datasets.ImageFolder(root=data_dir)

            # Verificar se há classes suficientes
            if len(full_dataset.classes) < num_classes:
                st.error(f"O número de classes encontradas ({len(full_dataset.classes)}) é menor do que o número especificado ({num_classes}).")
                shutil.rmtree(temp_dir)
                return

            st.write(f"**Classes Encontradas:** {full_dataset.classes}")

            # Visualizar algumas imagens
            visualize_data(full_dataset, full_dataset.classes)

            # Plotar distribuição das classes
            plot_class_distribution(full_dataset, full_dataset.classes)

            # Divisão dos dados
            dataset_size = len(full_dataset)
            indices = list(range(dataset_size))
            np.random.shuffle(indices)

            train_end = int(train_split * dataset_size)
            valid_end = int((train_split + valid_split) * dataset_size)

            train_indices = indices[:train_end]
            valid_indices = indices[train_end:valid_end]
            test_indices = indices[valid_end:]

            # Verificar se há dados suficientes em cada conjunto
            if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
                st.error("Divisão dos dados resultou em um conjunto vazio. Ajuste os percentuais de divisão.")
                shutil.rmtree(temp_dir)
                return

            # Criar datasets para treino, validação e teste
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

            # Atualizar os datasets com as transformações
            train_dataset_augmented = CustomDataset(train_dataset, transform=train_transforms)
            valid_dataset_transformed = CustomDataset(valid_dataset, transform=test_transforms)
            test_dataset_transformed = CustomDataset(test_dataset, transform=test_transforms)

            # Visualizar as primeiras 20 imagens após Data Augmentation
            st.write("**Visualização das Primeiras 20 Imagens após Data Augmentation:**")
            # Carregar um modelo sem fine-tuning para extrair embeddings
            model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
            if model_for_embeddings is None:
                st.error("Erro ao carregar o modelo para extração de embeddings.")
                shutil.rmtree(temp_dir)
                return

            train_df = apply_transforms_and_get_embeddings(train_dataset_augmented, model_for_embeddings, train_transforms, batch_size=batch_size)
            display_all_augmented_images(train_df, full_dataset.classes, max_images=20)

            # Visualizar os embeddings com PCA
            st.write("**Visualização dos Embeddings com PCA:**")
            embeddings = np.vstack(train_df['embedding'].values)
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            labels = train_df['label'].values
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=labels, palette='Set2')
            plt.title('Embeddings com PCA')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            st.pyplot(plt)
            plt.close()

            # Criar DataLoaders
            g = torch.Generator()
            g.manual_seed(42)

            train_loader = DataLoader(train_dataset_augmented, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            valid_loader = DataLoader(valid_dataset_transformed, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset_transformed, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

            # Carregar o modelo
            model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
            if model is None:
                st.error("Erro ao carregar o modelo.")
                shutil.rmtree(temp_dir)
                return

            # Definir a função de perda
            if use_weighted_loss:
                # Calcula os pesos das classes com base no conjunto de treinamento
                targets = []
                for _, labels in train_loader:
                    targets.extend(labels.numpy())
                class_counts = np.bincount(targets, minlength=num_classes)
                class_counts = class_counts + 1e-6  # Para evitar divisão por zero
                class_weights = 1.0 / class_counts
                class_weights = torch.FloatTensor(class_weights).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                st.sidebar.write("**Peso das Classes:**")
                st.sidebar.write(class_weights.cpu().numpy())
            else:
                criterion = nn.CrossEntropyLoss()

            # Definir o otimizador com L2 regularization (weight_decay)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)

            # Iniciar o treinamento
            st.write("**Iniciando o treinamento do modelo...**")
            model = train_model(train_loader, valid_loader, model, criterion, optimizer, epochs, patience)
            if model is None:
                st.error("Erro durante o treinamento.")
                shutil.rmtree(temp_dir)
                return

            st.success("Treinamento concluído com sucesso!")

            # Salvar o modelo treinado
            model_filename = f'{model_name}_trained.pth'
            torch.save(model.state_dict(), model_filename)
            st.write(f"Modelo salvo como `{model_filename}`")

            # Salvar as classes em um arquivo
            classes_data = "\n".join(full_dataset.classes)
            classes_filename = 'classes.txt'
            with open(classes_filename, 'w') as f:
                f.write(classes_data)
            st.write(f"Classes salvas como `{classes_filename}`")

            # Disponibilizar para download
            unique_id = uuid.uuid4()
            with open(model_filename, "rb") as file:
                st.download_button(
                    label="Download do Modelo Treinado",
                    data=file,
                    file_name=model_filename,
                    mime="application/octet-stream",
                    key=f"download_model_{unique_id}"
                )

            with open(classes_filename, "rb") as file:
                st.download_button(
                    label="Download das Classes",
                    data=file,
                    file_name=classes_filename,
                    mime="text/plain",
                    key=f"download_classes_{unique_id}"
                )

            # Avaliação no conjunto de teste
            st.write("**Avaliação no Conjunto de Teste**")
            evaluate_model(model, test_loader, full_dataset.classes)

            # Limpar o diretório temporário
            shutil.rmtree(temp_dir)

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            shutil.rmtree(temp_dir)

    else:
        st.warning("Por favor, forneça os dados e as configurações corretas.")

if __name__ == "__main__":
    main()
