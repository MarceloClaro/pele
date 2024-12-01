import os
import zipfile
import shutil
import tempfile
import random
import json
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.utils import resample
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
import gc
import logging
import base64
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import io
import warnings
from datetime import datetime
from scipy import stats

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
    plt.tight_layout()
    plt.savefig("visualize_data.png")
    st.image("visualize_data.png", caption='Exemplos do Conjunto de Dados')
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

    plt.tight_layout()
    plt.savefig("class_distribution.png")
    st.image("class_distribution.png", caption='Distribuição das Classes')
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
    # Definir função de coleta personalizada
    def pil_collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pil_collate_fn)
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    augmented_images_list = []

    # Remover a última camada do modelo para extrair os embeddings
    model_embedding = nn.Sequential(*list(model.children())[:-1])
    model_embedding.eval()
    model_embedding.to(device)

    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    index_pointer = 0  # Ponteiro para acompanhar os índices

    with torch.no_grad():
        for images, labels in data_loader:
            images_augmented = [transform(img) for img in images]
            images_augmented = torch.stack(images_augmented).to(device)
            embeddings = model_embedding(images_augmented)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
            embeddings_list.extend(embeddings)
            labels_list.extend(labels.numpy())
            augmented_images_list.extend([img.permute(1, 2, 0).numpy() for img in images_augmented.cpu()])
            # Atualizar o file_paths_list para corresponder às imagens atuais
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
                batch_indices = indices[index_pointer:index_pointer + len(images)]
                file_paths = [dataset.dataset.samples[i][0] for i in batch_indices]
                file_paths_list.extend(file_paths)
                index_pointer += len(images)
            else:
                file_paths_list.extend(['N/A'] * len(labels))

    # Criar o DataFrame
    df = pd.DataFrame({
        'file_path': file_paths_list,
        'label': labels_list,
        'embedding': embeddings_list,
        'augmented_image': augmented_images_list
    })

    return df


def display_all_augmented_images(df, class_names, max_images=None):
    """
    Exibe todas as imagens augmentadas do DataFrame de forma organizada.
    """
    if max_images is not None:
        df = df.head(max_images)
        st.write(f"**Visualização das Primeiras {max_images} Imagens após Data Augmentation:**")
    else:
        st.write("**Visualização de Todas as Imagens após Data Augmentation:**")

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
                    st.image(image, caption=class_names[label], use_column_width=True)


def visualize_embeddings(df, class_names):
    """
    Reduz a dimensionalidade dos embeddings e os visualiza em 2D.
    """
    embeddings = np.vstack(df['embedding'].values)
    labels = df['label'].values

    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Criar DataFrame para plotagem
    plot_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'label': labels
    })

    # Plotar
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='label', palette='Set2', legend='full')

    # Configurações do gráfico
    plt.title('Visualização dos Embeddings com PCA')
    plt.legend(title='Classes', labels=class_names)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')

    # Salvar o plot
    plt.tight_layout()
    plt.savefig("embeddings_pca.png")
    st.image("embeddings_pca.png", caption='Visualização dos Embeddings com PCA')
    plt.close()  # Fechar a figura para liberar memória


def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience, model_id=None, run_id=None):
    """
    Função principal para treinamento do modelo de classificação.
    """
    set_seed(42)

    # Exibir as configurações técnicas do modelo
    st.subheader(f"Treinamento do {model_name} - Execução {run_id}")
    st.write("**Configurações Técnicas:**")
    config = {
        'Modelo': model_name,
        'Fine-Tuning Completo': fine_tune,
        'Épocas': epochs,
        'Taxa de Aprendizagem': learning_rate,
        'Tamanho do Lote': batch_size,
        'Train Split': train_split,
        'Valid Split': valid_split,
        'L2 Regularization': l2_lambda,
        'Paciência Early Stopping': patience,
        'Use Weighted Loss': use_weighted_loss
    }
    config_df = pd.DataFrame(list(config.items()), columns=['Parâmetro', 'Valor'])
    st.table(config_df)

    # Salvar configurações em arquivo JSON
    config_filename = f'config_{model_name}_run{run_id}.json'
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    st.write(f"Configurações salvas como `{config_filename}`")

    # Ajustar o batch size para modelos maiores
    if model_name in ['ResNet50', 'DenseNet121']:
        batch_size = min(batch_size, 8)  # Ajuste conforme necessário
        st.write(f"Ajustando o tamanho do lote para {batch_size} devido ao uso do {model_name}")

    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Verificar se há classes suficientes
    if len(full_dataset.classes) < num_classes:
        st.error(f"O número de classes encontradas ({len(full_dataset.classes)}) é menor do que o número especificado ({num_classes}).")
        return None

    # Exibir dados
    visualize_data(full_dataset, full_dataset.classes)
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
        return None

    # Criar datasets para treino, validação e teste
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Criar dataframes para os conjuntos de treinamento, validação e teste com data augmentation e embeddings
    model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
    if model_for_embeddings is None:
        return None

    st.write("**Processando o conjunto de treinamento para incluir Data Augmentation e Embeddings...**")
    train_df = apply_transforms_and_get_embeddings(train_dataset, model_for_embeddings, train_transforms, batch_size=batch_size)
    st.write("**Processando o conjunto de validação...**")
    valid_df = apply_transforms_and_get_embeddings(valid_dataset, model_for_embeddings, test_transforms, batch_size=batch_size)
    st.write("**Processando o conjunto de teste...**")
    test_df = apply_transforms_and_get_embeddings(test_dataset, model_for_embeddings, test_transforms, batch_size=batch_size)

    # Mapear rótulos para nomes de classes
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_df['class_name'] = train_df['label'].map(idx_to_class)
    valid_df['class_name'] = valid_df['label'].map(idx_to_class)
    test_df['class_name'] = test_df['label'].map(idx_to_class)

    # Exibir dataframes no Streamlit sem a coluna 'augmented_image' e sem limitar a 5 linhas
    st.write("**Dataframe do Conjunto de Treinamento com Data Augmentation e Embeddings:**")
    st.dataframe(train_df.drop(columns=['augmented_image']))

    st.write("**Dataframe do Conjunto de Validação:**")
    st.dataframe(valid_df.drop(columns=['augmented_image']))

    st.write("**Dataframe do Conjunto de Teste:**")
    st.dataframe(test_df.drop(columns=['augmented_image']))

    # Exibir todas as imagens augmentadas (ou limitar conforme necessário)
    display_all_augmented_images(train_df, full_dataset.classes, max_images=100)  # Ajuste 'max_images' conforme necessário

    # Visualizar os embeddings
    visualize_embeddings(train_df, full_dataset.classes)

    # Exibir contagem de imagens por classe nos conjuntos de treinamento e teste
    st.write("**Distribuição das Classes no Conjunto de Treinamento:**")
    train_class_counts = train_df['class_name'].value_counts()
    st.bar_chart(train_class_counts)

    st.write("**Distribuição das Classes no Conjunto de Teste:**")
    test_class_counts = test_df['class_name'].value_counts()
    st.bar_chart(test_class_counts)

    # Atualizar os datasets com as transformações para serem usados nos DataLoaders
    train_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, train_indices), transform=train_transforms)
    valid_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, valid_indices), transform=test_transforms)
    test_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, test_indices), transform=test_transforms)

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(42)

    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6  # Para evitar divisão por zero
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Carregar o modelo
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
    if model is None:
        return None

    # Definir o otimizador com L2 regularization (weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)

    # Inicializar as listas de perdas e acurácias no st.session_state com chaves únicas
    train_losses_key = f"train_losses_{model_id}_{run_id}"
    valid_losses_key = f"valid_losses_{model_id}_{run_id}"
    train_accuracies_key = f"train_accuracies_{model_id}_{run_id}"
    valid_accuracies_key = f"valid_accuracies_{model_id}_{run_id}"

    # Verificar se as chaves já existem no st.session_state; se não, inicializá-las
    if train_losses_key not in st.session_state:
        st.session_state[train_losses_key] = []
    if valid_losses_key not in st.session_state:
        st.session_state[valid_losses_key] = []
    if train_accuracies_key not in st.session_state:
        st.session_state[train_accuracies_key] = []
    if valid_accuracies_key not in st.session_state:
        st.session_state[valid_accuracies_key] = []

    # Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None  # Inicializar

    # Placeholders para gráficos dinâmicos
    placeholder = st.empty()
    progress_bar = st.progress(0)
    epoch_text = st.empty()

    # Treinamento
    for epoch in range(epochs):
        set_seed(42 + epoch)
        running_loss = 0.0
        running_corrects = 0
        model.train()

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

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        st.session_state[train_losses_key].append(epoch_loss)
        st.session_state[train_accuracies_key].append(epoch_acc.item())

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

        valid_epoch_loss = valid_running_loss / len(valid_dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_dataset)
        st.session_state[valid_losses_key].append(valid_epoch_loss)
        st.session_state[valid_accuracies_key].append(valid_epoch_acc.item())

        # Atualizar gráficos dinamicamente
        with placeholder.container():
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))

            # Gráfico de Perda
            ax[0].plot(range(1, len(st.session_state[train_losses_key]) + 1), st.session_state[train_losses_key], label='Treino')
            ax[0].plot(range(1, len(st.session_state[valid_losses_key]) + 1), st.session_state[valid_losses_key], label='Validação')
            ax[0].set_title(f'Perda por Época - {model_name} (Execução {run_id})')
            ax[0].set_xlabel('Épocas')
            ax[0].set_ylabel('Perda')
            ax[0].legend()

            # Gráfico de Acurácia
            ax[1].plot(range(1, len(st.session_state[train_accuracies_key]) + 1), st.session_state[train_accuracies_key], label='Treino')
            ax[1].plot(range(1, len(st.session_state[valid_accuracies_key]) + 1), st.session_state[valid_accuracies_key], label='Validação')
            ax[1].set_title(f'Acurácia por Época - {model_name} (Execução {run_id})')
            ax[1].set_xlabel('Épocas')
            ax[1].set_ylabel('Acurácia')
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(f'loss_accuracy_{model_name}_run{run_id}.png')
            st.image(f'loss_accuracy_{model_name}_run{run_id}.png', caption='Perda e Acurácia por Época')
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
    plot_metrics(
        st.session_state[train_losses_key],
        st.session_state[valid_losses_key],
        st.session_state[train_accuracies_key],
        st.session_state[valid_accuracies_key],
        model_name=model_name,
        run_id=run_id
    )

    # Avaliação Final no Conjunto de Teste
    st.write("**Avaliação no Conjunto de Teste**")
    metrics = compute_metrics(model, test_loader, full_dataset.classes, model_name, run_id)

    # Análise de Erros
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes, model_name, run_id)

    # Clusterização e Análise Comparativa
    st.write("**Análise de Clusterização**")
    perform_clustering(model, test_loader, full_dataset.classes, model_name, run_id)

    # Liberar memória dos data loaders
    del train_loader, valid_loader, test_loader
    torch.cuda.empty_cache()  # Limpar cache da GPU
    gc.collect()

    # Armazenar o modelo e as classes no st.session_state
    st.session_state['model'] = model
    st.session_state['classes'] = full_dataset.classes
    st.session_state['trained_model_name'] = model_name  # Armazena o nome do modelo treinado

    return model, full_dataset.classes, metrics


def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, model_name, run_id):
    """
    Plota os gráficos de perda e acurácia e salva-os em arquivos.
    """
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Perda
    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title(f'Perda por Época - {model_name} (Execução {run_id})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Gráfico de Acurácia
    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title(f'Acurácia por Época - {model_name} (Execução {run_id})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    plt.tight_layout()
    plot_filename = f'loss_accuracy_final_{model_name}_run{run_id}.png'
    fig.savefig(plot_filename)
    st.image(plot_filename, caption='Perda e Acurácia Finais')
    plt.close(fig)  # Fechar a figura para liberar memória

    # Disponibilizar para download
    with open(plot_filename, "rb") as file:
        btn = st.download_button(
            label="Download dos Gráficos de Perda e Acurácia",
            data=file,
            file_name=plot_filename,
            mime="image/png"
        )
    if btn:
        st.success("Gráficos baixados com sucesso!")


def compute_metrics(model, dataloader, classes, model_name, run_id):
    """
    Calcula métricas detalhadas e exibe matriz de confusão e relatório de classificação.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
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
    report_filename = f'classification_report_{model_name}_run{run_id}.csv'
    report_df.to_csv(report_filename)
    st.write(f"Relatório de classificação salvo como `{report_filename}`")

    # Matriz de Confusão Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão Normalizada')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{model_name}_run{run_id}.png'
    fig.savefig(cm_filename)
    st.image(cm_filename, caption='Matriz de Confusão Normalizada')
    plt.close(fig)  # Fechar a figura para liberar memória

    # Disponibilizar para download
    with open(cm_filename, "rb") as file:
        btn = st.download_button(
            label="Download da Matriz de Confusão",
            data=file,
            file_name=cm_filename,
            mime="image/png"
        )
    if btn:
        st.success("Matriz de Confusão baixada com sucesso!")

    # Curva ROC
    roc_auc = None
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
        plt.tight_layout()
        roc_filename = f'roc_curve_{model_name}_run{run_id}.png'
        fig.savefig(roc_filename)
        st.image(roc_filename, caption='Curva ROC')
        plt.close(fig)  # Fechar a figura para liberar memória

        # Disponibilizar para download
        with open(roc_filename, "rb") as file:
            btn = st.download_button(
                label="Download da Curva ROC",
                data=file,
                file_name=roc_filename,
                mime="image/png"
            )
        if btn:
            st.success("Curva ROC baixada com sucesso!")
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")

        # Salvar AUC-ROC
        auc_filename = f'auc_roc_{model_name}_run{run_id}.txt'
        with open(auc_filename, 'w') as f:
            f.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")
        st.write(f"AUC-ROC Média Ponderada salvo como `{auc_filename}`")

        # Disponibilizar para download
        with open(auc_filename, "rb") as file:
            btn = st.download_button(
                label="Download do AUC-ROC",
                data=file,
                file_name=auc_filename,
                mime="text/plain"
            )
        if btn:
            st.success("AUC-ROC baixado com sucesso!")

    # Calcule as métricas de desempenho
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    # 'roc_auc' já foi calculado acima

    # Retornar as métricas em um dicionário
    metrics = {
        'Model': model_name,
        'Run_ID': run_id,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'ROC_AUC': roc_auc if roc_auc is not None else np.nan
    }

    # Salvar métricas em arquivo CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_filename = f'metrics_{model_name}_run{run_id}.csv'
    metrics_df.to_csv(metrics_filename, index=False)
    st.write(f"Métricas salvas como `{metrics_filename}`")

    # Disponibilizar para download
    with open(metrics_filename, "rb") as file:
        btn = st.download_button(
            label="Download das Métricas",
            data=file,
            file_name=metrics_filename,
            mime="text/csv"
        )
    if btn:
        st.success("Métricas baixadas com sucesso!")

    return metrics


def error_analysis(model, dataloader, classes, model_name, run_id):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break

    if misclassified_images:
        st.write("Algumas imagens mal classificadas:")
        fig, axes = plt.subplots(1, min(5, len(misclassified_images)), figsize=(15, 3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i]
            image = image.permute(1, 2, 0).numpy()
            axes[i].imshow(image)
            axes[i].set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'misclassified_{model_name}_run{run_id}.png')
        st.image(f'misclassified_{model_name}_run{run_id}.png', caption='Exemplos de Erros de Classificação')
        plt.close(fig)  # Fechar a figura para liberar memória

        # Disponibilizar para download
        with open(f'misclassified_{model_name}_run{run_id}.png', "rb") as file:
            btn = st.download_button(
                label="Download das Imagens Mal Classificadas",
                data=file,
                file_name=f'misclassified_{model_name}_run{run_id}.png',
                mime="image/png"
            )
        if btn:
            st.success("Imagens mal classificadas baixadas com sucesso!")
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")


def perform_clustering(model, dataloader, classes, model_name, run_id):
    """
    Realiza a extração de features e aplica algoritmos de clusterização.
    """
    # Extrair features usando o modelo pré-treinado
    features = []
    labels = []

    # Remover a última camada (classificador)
    if isinstance(model, nn.Sequential):
        model_feat = model
    else:
        model_feat = nn.Sequential(*list(model.children())[:-1])
    model_feat.eval()
    model_feat.to(device)

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            output = model_feat(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Clusterização com KMeans
    kmeans = KMeans(n_clusters=len(classes), random_state=42)
    clusters_kmeans = kmeans.fit_predict(features)

    # Clusterização Hierárquica
    agglo = AgglomerativeClustering(n_clusters=len(classes))
    clusters_agglo = agglo.fit_predict(features)

    # Plotagem dos resultados
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico KMeans
    scatter = ax[0].scatter(features_2d[:, 0], features_2d[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.6)
    legend1 = ax[0].legend(*scatter.legend_elements(), title="Clusters")
    ax[0].add_artist(legend1)
    ax[0].set_title('Clusterização com KMeans')

    # Gráfico Agglomerative Clustering
    scatter = ax[1].scatter(features_2d[:, 0], features_2d[:, 1], c=clusters_agglo, cmap='viridis', alpha=0.6)
    legend1 = ax[1].legend(*scatter.legend_elements(), title="Clusters")
    ax[1].add_artist(legend1)
    ax[1].set_title('Clusterização Hierárquica')

    plt.tight_layout()
    clustering_filename = f'clustering_{model_name}_run{run_id}.png'
    fig.savefig(clustering_filename)
    st.image(clustering_filename, caption='Resultados da Clusterização')
    plt.close(fig)  # Fechar a figura para liberar memória

    # Disponibilizar para download
    with open(clustering_filename, "rb") as file:
        btn = st.download_button(
            label="Download dos Resultados de Clusterização",
            data=file,
            file_name=clustering_filename,
            mime="image/png"
        )
    if btn:
        st.success("Resultados de clusterização baixados com sucesso!")

    # Métricas de Avaliação
    ari_kmeans = adjusted_rand_score(labels, clusters_kmeans)
    nmi_kmeans = normalized_mutual_info_score(labels, clusters_kmeans)
    ari_agglo = adjusted_rand_score(labels, clusters_agglo)
    nmi_agglo = normalized_mutual_info_score(labels, clusters_agglo)

    st.write(f"**KMeans** - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
    st.write(f"**Agglomerative Clustering** - ARI: {ari_agglo:.4f}, NMI: {nmi_agglo:.4f}")

    # Salvar métricas de clusterização
    clustering_metrics = {
        'Model': model_name,
        'Run_ID': run_id,
        'KMeans_ARI': ari_kmeans,
        'KMeans_NMI': nmi_kmeans,
        'Agglomerative_ARI': ari_agglo,
        'Agglomerative_NMI': nmi_agglo
    }
    clustering_metrics_df = pd.DataFrame([clustering_metrics])
    clustering_metrics_filename = f'clustering_metrics_{model_name}_run{run_id}.csv'
    clustering_metrics_df.to_csv(clustering_metrics_filename, index=False)
    st.write(f"Métricas de clusterização salvas como `{clustering_metrics_filename}`")

    # Disponibilizar para download
    with open(clustering_metrics_filename, "rb") as file:
        btn = st.download_button(
            label="Download das Métricas de Clusterização",
            data=file,
            file_name=clustering_metrics_filename,
            mime="text/csv"
        )
    if btn:
        st.success("Métricas de clusterização baixadas com sucesso!")


def evaluate_image(model, image, classes):
    """
    Avalia uma única imagem e retorna a classe predita e a confiança.
    """
    model.eval()
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    class_idx = predicted.item()
    class_name = classes[class_idx]
    return class_name, confidence.item()


def visualize_activations(model, image, class_names, model_name, run_id):
    """
    Visualiza as ativações na imagem usando Grad-CAM.
    """
    model.eval()  # Coloca o modelo em modo de avaliação
    input_tensor = test_transforms(image).unsqueeze(0).to(device)

    # Verificar se o modelo é suportado
    if model_name.startswith('ResNet'):
        target_layer = 'layer4'
    elif model_name.startswith('DenseNet'):
        target_layer = 'features.denseblock4'
    else:
        st.error("Modelo não suportado para Grad-CAM.")
        return

    # Criar o objeto CAM usando torchcam
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

    # Ativar Grad-CAM
    with torch.set_grad_enabled(True):
        out = model(input_tensor)  # Faz a previsão
        probabilities = torch.nn.functional.softmax(out, dim=1)
        confidence, pred = torch.max(probabilities, 1)  # Obtém a classe predita
        pred_class = pred.item()

        # Gerar o mapa de ativação
        activation_map = cam_extractor(pred_class, out)

    # Converter o mapa de ativação para PIL Image
    activation_map = activation_map[0]
    result = overlay_mask(to_pil_image(input_tensor.squeeze().cpu()), to_pil_image(activation_map.squeeze(), mode='F'), alpha=0.5)

    # Converter a imagem para array NumPy
    image_np = np.array(image)

    # Exibir as imagens: Imagem Original e Grad-CAM
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Imagem original
    ax[0].imshow(image_np)
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')

    # Imagem com Grad-CAM
    ax[1].imshow(result)
    ax[1].set_title('Grad-CAM')
    ax[1].axis('off')

    plt.tight_layout()
    activation_filename = f'grad_cam_{model_name}_run{run_id}.png'
    fig.savefig(activation_filename)
    st.image(activation_filename, caption='Visualização de Grad-CAM')
    plt.close(fig)  # Fechar a figura para liberar memória

    # Disponibilizar para download
    with open(activation_filename, "rb") as file:
        btn = st.download_button(
            label="Download da Visualização de Grad-CAM",
            data=file,
            file_name=activation_filename,
            mime="image/png"
        )
    if btn:
        st.success("Visualização de Grad-CAM baixada com sucesso!")

    # Limpar os hooks após a visualização
    cam_extractor.clear_hooks()


def perform_anova(data, groups):
    """
    Realiza a análise ANOVA para comparar as médias entre diferentes grupos.
    """
    f_val, p_val = stats.f_oneway(*[data[groups == group] for group in np.unique(groups)])
    return f_val, p_val


def visualize_anova_results(f_val, p_val):
    """
    Visualiza os resultados da análise ANOVA.
    """
    st.write(f"**Valor F:** {f_val:.4f}, **Valor p:** {p_val:.4f}")
    if p_val < 0.05:
        st.write("Os resultados são estatisticamente significativos.")
    else:
        st.write("Os resultados não são estatisticamente significativos.")


def main():
    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        try:
            st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
            logging.info(f"Ícone {icon_path} carregado com sucesso.")
        except Exception as e:
            st.set_page_config(page_title="Geomaker", layout="wide")
            logging.warning(f"Erro ao carregar o ícone {icon_path}: {e}")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")

    # Layout da página
    if os.path.exists('capa.png'):
        try:
            st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_container_width=True)
        except UnidentifiedImageError:
            st.warning("Imagem 'capa.png' não pôde ser carregada ou está corrompida.")
    else:
        st.warning("Imagem 'capa.png' não encontrada.")

    # Carregar o logotipo na barra lateral
    if os.path.exists("logo.png"):
        try:
            st.sidebar.image("logo.png", width=200)
        except UnidentifiedImageError:
            st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")

    st.title("Classificação de Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar múltiplos modelos de classificação de imagens, aplicar algoritmos de clustering para análise comparativa e realizar avaliações estatísticas detalhadas.")
    st.write("As etapas são cuidadosamente documentadas para auxiliar na reprodução e análise científica.")

    # Inicializar 'all_model_metrics' no session_state se ainda não existir
    if 'all_model_metrics' not in st.session_state:
        st.session_state['all_model_metrics'] = []

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, step=1, key="num_classes")
    # Removido o selectbox de seleção de modelo para o treinamento múltiplo
    fine_tune = st.sidebar.checkbox("Fine-Tuning Completo", value=False, key="fine_tune")
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=500, value=200, step=1, key="epochs")
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.0001, key="learning_rate")
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2, key="batch_size")
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="train_split")
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05, key="valid_split")
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01, key="l2_lambda")
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1, key="patience")
    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada para Classes Desbalanceadas", value=False, key="use_weighted_loss")
    if os.path.exists("eu.ico"):
        try:
            st.sidebar.image("eu.ico", width=80)
        except UnidentifiedImageError:
            st.sidebar.text("Imagem 'eu.ico' não pôde ser carregada ou está corrompida.")
    else:
        st.sidebar.text("Imagem 'eu.ico' não encontrada.")

    st.sidebar.write("""
    **Produzido pelo:**

    Projeto Geomaker + IA 

    [DOI:10.5281/zenodo.13910277](https://doi.org/10.5281/zenodo.13910277)

    - **Professor:** Marcelo Claro.

    - **Contatos:** marceloclaro@gmail.com

    - **Whatsapp:** (88)981587145

    - **Instagram:** [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    """)

    # Verificar se a soma dos splits é válida
    if train_split + valid_split > 0.95:
        st.sidebar.error("A soma dos splits de treinamento e validação deve ser menor ou igual a 0.95.")

    # Adicionar uma seção para Treinamento Múltiplo de Modelos
    st.header("Treinamento de Múltiplos Modelos para Análise Estatística")
    st.write("Treine múltiplos modelos com diferentes configurações para avaliar estatisticamente o desempenho.")

    # Usando st.form para agrupar os widgets e garantir que as entradas sejam submetidas juntas
    with st.form(key='training_form'):
        # Configurações para múltiplos modelos
        runs_per_model = st.number_input("Número de Execuções por Modelo:", min_value=1, max_value=10, value=3, step=1, key="runs_per_model")

        # Checkboxes para seleção de modelos
        st.write("Selecione os modelos que deseja treinar:")
        model_resnet18 = st.checkbox('ResNet18', value=True, key='model_resnet18')
        model_resnet50 = st.checkbox('ResNet50', value=True, key='model_resnet50')
        model_densenet121 = st.checkbox('DenseNet121', value=True, key='model_densenet121')

        # Uploader de arquivo ZIP
        zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"], key="zip_file_uploader_main_multiple")

        # Botão para iniciar o treinamento múltiplo
        submit_button = st.form_submit_button(label='Iniciar Treinamento Múltiplo')

    if submit_button:
        # Capturar as seleções dos modelos
        model_selection = {
            'ResNet18': model_resnet18,
            'ResNet50': model_resnet50,
            'DenseNet121': model_densenet121
        }
        # Lista de modelos selecionados na ordem fixa
        model_list = [model_name for model_name in ['ResNet18', 'ResNet50', 'DenseNet121'] if model_selection[model_name]]

        if len(model_list) == 0:
            st.error("Por favor, selecione pelo menos um modelo para treinar.")
            return

        # Inicializar lista para armazenar modelos treinados
        trained_models = []

        if zip_file is None:
            st.error("Por favor, faça upload do arquivo ZIP com as imagens.")
        else:
            try:
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                data_dir = temp_dir

                for i, model_name in enumerate(model_list):
                    for run in range(1, runs_per_model + 1):
                        st.write(f"**Treinando Modelo {i+1}/{len(model_list)} ({model_name}) - Execução {run}/{runs_per_model}**")
                        model_id = f"model_{i+1}"
                        run_id = run
                        model_data = train_model(
                            data_dir, num_classes, model_name, fine_tune,
                            epochs, learning_rate, batch_size, train_split,
                            valid_split, use_weighted_loss, l2_lambda, patience,
                            model_id=model_id, run_id=run_id  # Passar o run_id para métricas distintas
                        )

                        if model_data is None:
                            st.error(f"Erro no treinamento do Modelo {i+1}, Execução {run}.")
                            continue

                        model, classes, metrics = model_data
                        st.session_state['all_model_metrics'].append(metrics)
                        st.success(f"Treinamento do Modelo {i+1} ({model_name}), Execução {run} concluído!")

                        # Armazenar o modelo treinado na lista
                        trained_models.append({
                            'model': model,
                            'model_name': model_name,
                            'run_id': run,
                            'classes': classes
                        })

                        # Salvar o modelo treinado
                        model_filename = f'{model_name}_run{run}.pth'
                        torch.save(model.state_dict(), model_filename)
                        st.write(f"Modelo salvo como `{model_filename}`")

                        # Salvar as classes em um arquivo
                        classes_data = "\n".join(classes)
                        classes_filename = f'classes_{model_name}_run{run}.txt'
                        with open(classes_filename, 'w') as f:
                            f.write(classes_data)
                        st.write(f"Classes salvas como `{classes_filename}`")

                        # Disponibilizar para download
                        with open(model_filename, "rb") as file:
                            btn = st.download_button(
                                label=f"Download do Modelo {model_name}_Execução_{run}",
                                data=file,
                                file_name=model_filename,
                                mime="application/octet-stream",
                                key=f"download_model_button_{model_name}_{run}"
                            )
                        if btn:
                            st.success(f"Modelo {model_name}_run{run} baixado com sucesso!")

                        with open(classes_filename, "rb") as file:
                            btn = st.download_button(
                                label=f"Download das Classes para {model_name}_Execução_{run}",
                                data=file,
                                file_name=classes_filename,
                                mime="text/plain",
                                key=f"download_classes_button_{model_name}_{run}"
                            )
                        if btn:
                            st.success(f"Classes {model_name}_run{run} baixadas com sucesso!")

                        # Salvar métricas em arquivo CSV
                        metrics_df = pd.DataFrame([metrics])
                        metrics_filename = f'metrics_{model_name}_run{run}.csv'
                        metrics_df.to_csv(metrics_filename, index=False)
                        st.write(f"Métricas salvas como `{metrics_filename}`")

                        # Disponibilizar para download
                        with open(metrics_filename, "rb") as file:
                            btn = st.download_button(
                                label=f"Download das Métricas para {model_name}_Execução_{run}",
                                data=file,
                                file_name=metrics_filename,
                                mime="text/csv",
                                key=f"download_metrics_button_{model_name}_{run}"
                            )
                        if btn:
                            st.success(f"Métricas {model_name}_run{run} baixadas com sucesso!")

                        # Limpar cache e liberar memória
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

                # Limpar o diretório temporário
                shutil.rmtree(temp_dir)

                # Exibir as métricas coletadas após o treinamento
                if len(st.session_state['all_model_metrics']) > 0:
                    st.header("Métricas de Desempenho de Todos os Modelos")
                    metrics_df = pd.DataFrame(st.session_state['all_model_metrics'])
                    st.dataframe(metrics_df)

                    # Salvar métricas coletadas em um único arquivo
                    all_metrics_filename = 'all_model_metrics.csv'
                    metrics_df.to_csv(all_metrics_filename, index=False)
                    st.write(f"Métricas de todos os modelos salvas como `{all_metrics_filename}`")

                    # Disponibilizar para download
                    with open(all_metrics_filename, "rb") as file:
                        btn = st.download_button(
                            label="Download das Métricas de Todos os Modelos",
                            data=file,
                            file_name=all_metrics_filename,
                            mime="text/csv"
                        )
                    if btn:
                        st.success("Métricas de todos os modelos baixadas com sucesso!")

                    # Calcular Intervalos de Confiança para Cada Métrica
                    st.subheader("Intervalos de Confiança para as Métricas de Desempenho")
                    confidence_level = 0.95
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
                        data = metrics_df[metric].dropna()
                        if len(data) > 1:
                            conf_interval = stats.t.interval(confidence_level, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
                            st.write(f"**{metric}:** Média = {np.mean(data):.4f}, Intervalo de Confiança de {int(confidence_level*100)}% = [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]")
                        elif len(data) == 1:
                            st.write(f"**{metric}:** Apenas uma observação disponível.")
                        else:
                            st.write(f"**{metric}:** Nenhum dado disponível.")

                    # Realizar ANOVA para Cada Métrica
                    st.subheader("Análise de Variância (ANOVA) para as Métricas de Desempenho")
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
                        data = metrics_df[['Model', metric]].dropna()
                        st.write(f"**Métrica: {metric}**")
                        # Verificar se há dados suficientes
                        group_sizes = data.groupby('Model').size()
                        st.write("**Tamanhos dos Grupos (Modelos):**")
                        st.write(group_sizes)
                        if len(group_sizes) >= 2 and (group_sizes >= 2).all():
                            # Preparar os dados para ANOVA
                            groups = data['Model'].values
                            metric_data = data[metric].values
                            f_val, p_val = perform_anova(metric_data, groups)
                            visualize_anova_results(f_val, p_val)
                        else:
                            st.write(f"**{metric}:** ANOVA não pode ser realizada. É necessário pelo menos dois modelos com pelo menos duas observações cada.")

                    # Realizar Teste Tukey HSD para Cada Métrica
                    st.subheader("Teste Post-Hoc Tukey HSD para as Métricas de Desempenho")
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
                        data = metrics_df[['Model', metric]].dropna()
                        group_sizes = data.groupby('Model').size()
                        if len(group_sizes) >= 2 and (group_sizes >= 2).all():
                            # Supondo que cada modelo seja um grupo distinto
                            tukey = pairwise_tukeyhsd(endog=data[metric], groups=data['Model'], alpha=0.05)
                            st.write(f"**{metric}:**")
                            st.text(tukey.summary())
                            # Salvar o resumo do Tukey
                            tukey_summary = tukey.summary().as_text()
                            tukey_filename = f'tukey_{metric}_{model_name}_run{run_id}.txt'
                            with open(tukey_filename, 'w') as f:
                                f.write(tukey_summary)
                            st.write(f"Resumo do Teste Tukey HSD salvo como `{tukey_filename}`")

                            # Disponibilizar para download
                            with open(tukey_filename, "rb") as file:
                                btn = st.download_button(
                                    label=f"Download do Resumo Tukey para {metric}",
                                    data=file,
                                    file_name=tukey_filename,
                                    mime="text/plain",
                                    key=f"download_tukey_{metric}_{model_name}_{run_id}"
                                )
                            if btn:
                                st.success(f"Resumo Tukey para {metric} baixado com sucesso!")
                        else:
                            st.write(f"**{metric}:** Teste Tukey HSD não pode ser realizado. É necessário pelo menos dois modelos com pelo menos duas observações cada.")

            except Exception as e:
                st.error(f"Erro durante o treinamento múltiplo: {e}")

    # Opções de carregamento do modelo
    st.header("Opções de Carregamento do Modelo")

    model_option = st.selectbox("Escolha uma opção:", ["Treinar um novo modelo", "Carregar um modelo existente"], key="model_option_main")
    if model_option == "Carregar um modelo existente":
        # Upload do modelo pré-treinado
        model_file = st.file_uploader("Faça upload do arquivo do modelo (.pt ou .pth)", type=["pt", "pth"], key="model_file_uploader_main")
        if model_file is not None and num_classes > 0:
            # Seleção do modelo
            model_name = st.selectbox("Modelo Pré-treinado:", options=['ResNet18', 'ResNet50', 'DenseNet121'], key="model_name_single_load")
            # Carregar o modelo
            model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
            if model is None:
                st.error("Erro ao carregar o modelo.")
                return

            # Carregar os pesos do modelo
            try:
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict)
                st.session_state['model'] = model
                st.session_state['trained_model_name'] = model_name  # Armazena o nome do modelo treinado
                st.success("Modelo carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
                return

            # Carregar as classes
            classes_file = st.file_uploader("Faça upload do arquivo com as classes (classes.txt)", type=["txt"], key="classes_file_uploader_main_load")
            if classes_file is not None:
                try:
                    classes = classes_file.read().decode("utf-8").splitlines()
                    st.session_state['classes'] = classes
                    st.write(f"Classes carregadas: {classes}")
                except Exception as e:
                    st.error(f"Erro ao carregar as classes: {e}")
            else:
                st.error("Por favor, forneça o arquivo com as classes.")

    elif model_option == "Treinar um novo modelo":
        # Seleção do modelo
        model_name = st.selectbox("Modelo Pré-treinado para Treinamento Único:", options=['ResNet18', 'ResNet50', 'DenseNet121'], key="model_name_single_train")
        # Upload do arquivo ZIP
        zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"], key="zip_file_uploader_single")
        if zip_file is not None and num_classes > 0 and train_split + valid_split <= 0.95:
            try:
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                data_dir = temp_dir

                st.write("Iniciando o treinamento supervisionado...")
                model_data = train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience, model_id="single_run", run_id=1)

                if model_data is None:
                    st.error("Erro no treinamento do modelo.")
                    shutil.rmtree(temp_dir)
                    return

                model, classes, metrics = model_data
                # O modelo e as classes já estão armazenados no st.session_state
                st.success("Treinamento concluído!")

                # Opção para baixar o modelo treinado
                model_filename = f'{model_name}_run1.pth'
                torch.save(model.state_dict(), model_filename)
                st.write(f"Modelo salvo como `{model_filename}`")

                # Salvar as classes em um arquivo
                classes_data = "\n".join(classes)
                classes_filename = f'classes_{model_name}_run1.txt'
                with open(classes_filename, 'w') as f:
                    f.write(classes_data)
                st.write(f"Classes salvas como `{classes_filename}`")

                # Disponibilizar para download
                with open(model_filename, "rb") as file:
                    btn = st.download_button(
                        label="Download do Modelo",
                        data=file,
                        file_name=model_filename,
                        mime="application/octet-stream",
                        key="download_model_button_single"
                    )
                if btn:
                    st.success("Modelo baixado com sucesso!")

                with open(classes_filename, "rb") as file:
                    btn = st.download_button(
                        label="Download das Classes",
                        data=file,
                        file_name=classes_filename,
                        mime="text/plain",
                        key="download_classes_button_single"
                    )
                if btn:
                    st.success("Classes baixadas com sucesso!")

                # Salvar métricas em arquivo CSV
                metrics_df = pd.DataFrame([metrics])
                metrics_filename = f'metrics_{model_name}_run1.csv'
                metrics_df.to_csv(metrics_filename, index=False)
                st.write(f"Métricas salvas como `{metrics_filename}`")

                # Disponibilizar para download
                with open(metrics_filename, "rb") as file:
                    btn = st.download_button(
                        label="Download das Métricas",
                        data=file,
                        file_name=metrics_filename,
                        mime="text/csv",
                        key="download_metrics_button_single"
                    )
                if btn:
                    st.success("Métricas baixadas com sucesso!")

                # Limpar o diretório temporário
                shutil.rmtree(temp_dir)

            except Exception as e:
                st.error(f"Erro durante o treinamento do modelo único: {e}")

    # Avaliação de uma imagem individual
    st.header("Avaliação de Imagem")
    evaluate = st.radio("Deseja avaliar uma imagem?", ("Sim", "Não"), key="evaluate_option")
    if evaluate == "Sim":
        # Verificar se o modelo já foi carregado ou treinado
        if 'model' not in st.session_state or 'classes' not in st.session_state:
            st.warning("Nenhum modelo carregado ou treinado. Por favor, carregue um modelo existente ou treine um novo modelo.")
            # Opção para carregar um modelo existente
            model_file_eval = st.file_uploader("Faça upload do arquivo do modelo (.pt ou .pth)", type=["pt", "pth"], key="model_file_uploader_eval")
            if model_file_eval is not None:
                num_classes_eval = st.number_input("Número de Classes:", min_value=2, step=1, key="num_classes_eval_eval")
                model_name_eval = st.selectbox("Modelo Pré-treinado:", options=['ResNet18', 'ResNet50', 'DenseNet121'], key="model_name_eval")
                model_eval = get_model(model_name_eval, num_classes_eval, dropout_p=0.5, fine_tune=False)
                if model_eval is None:
                    st.error("Erro ao carregar o modelo.")
                    return
                try:
                    state_dict = torch.load(model_file_eval, map_location=device)
                    model_eval.load_state_dict(state_dict)
                    st.session_state['model'] = model_eval
                    st.session_state['trained_model_name'] = model_name_eval  # Armazena o nome do modelo treinado
                    st.success("Modelo carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo: {e}")
                    return

                # Carregar as classes
                classes_file_eval = st.file_uploader("Faça upload do arquivo com as classes (classes.txt)", type=["txt"], key="classes_file_uploader_eval_load")
                if classes_file_eval is not None:
                    try:
                        classes_eval = classes_file_eval.read().decode("utf-8").splitlines()
                        st.session_state['classes'] = classes_eval
                        st.write(f"Classes carregadas: {classes_eval}")
                    except Exception as e:
                        st.error(f"Erro ao carregar as classes: {e}")
                else:
                    st.error("Por favor, forneça o arquivo com as classes.")
            else:
                st.info("Aguardando o upload do modelo e das classes.")
        else:
            model_eval = st.session_state['model']
            classes_eval = st.session_state['classes']
            model_name_eval = st.session_state.get('trained_model_name', 'ResNet18')  # Usa o nome do modelo armazenado

        eval_image_file = st.file_uploader("Faça upload da imagem para avaliação", type=["png", "jpg", "jpeg", "bmp", "gif"], key="eval_image_file_eval")
        if eval_image_file is not None:
            eval_image_file.seek(0)
            try:
                eval_image = Image.open(eval_image_file).convert("RGB")
            except Exception as e:
                st.error(f"Erro ao abrir a imagem: {e}")
                return

            st.image(eval_image, caption='Imagem para avaliação', use_container_width=True)

            if 'model' in st.session_state and 'classes' in st.session_state:
                class_name, confidence = evaluate_image(st.session_state['model'], eval_image, st.session_state['classes'])
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

                # Visualizar ativações
                visualize_activations(st.session_state['model'], eval_image, st.session_state['classes'], model_name_eval, run_id=1)
            else:
                st.error("Modelo ou classes não carregados. Por favor, carregue um modelo ou treine um novo modelo.")

    st.write("### Documentação dos Procedimentos")
    st.write("Todas as etapas foram cuidadosamente registradas. Utilize esta documentação para reproduzir o experimento e analisar os resultados.")

    # Encerrar a aplicação
    st.write("Obrigado por utilizar o aplicativo!")


if __name__ == "__main__":
    main()