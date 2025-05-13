import os
import zipfile
import random
import tempfile
import uuid
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
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
# from sklearn import metrics # Parece não ser usado diretamente, metrics é um alias comum para sklearn.metrics
from sklearn.utils import resample # resample não parece ser usado no código fornecido
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
import logging
import base64 # base64 não parece ser usado no código fornecido
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import io # io não parece ser usado no código fornecido
import warnings
from datetime import datetime # datetime não parece ser usado no código fornecido
from scipy import stats
import torchvision
import torchcam

# Supressão dos avisos relacionados ao torch.classes
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Definir as transformações
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
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Comentar se não for usar ImageNet mean/std
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Comentar se não for usar ImageNet mean/std
])

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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Funções movidas para o escopo global ---
def visualize_data(dataset, classes):
    st.write("Visualização de algumas imagens do conjunto de dados:")
    # Garantir que estamos lidando com o dataset ImageFolder ou similar, não um Subset.
    if isinstance(dataset, torch.utils.data.Subset):
        # Tentar acessar o dataset original e os índices
        original_dataset = dataset.dataset
        indices_to_sample_from = dataset.indices
        if not indices_to_sample_from: # Lista de índices vazia
             st.warning("Subconjunto de dados para visualização está vazio.")
             return
    else:
        original_dataset = dataset
        indices_to_sample_from = list(range(len(original_dataset)))


    num_samples = min(10, len(indices_to_sample_from))
    if num_samples == 0:
        st.warning("Não há imagens suficientes para visualização.")
        return

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1: # Matplotlib trata subplot(1,1,...) diferente
        axes = [axes]

    sampled_indices = random.sample(indices_to_sample_from, num_samples)

    for i, original_idx in enumerate(sampled_indices):
        try:
            image, label = original_dataset[original_idx] # Acessa a imagem original PIL
            if not isinstance(image, Image.Image): # Se já for tensor
                image = transforms.ToPILImage()(image)
            
            image_np = np.array(image)
            axes[i].imshow(image_np)
            axes[i].set_title(classes[label])
            axes[i].axis('off')
        except Exception as e:
            logging.error(f"Erro ao visualizar imagem no índice {original_idx}: {e}")
            if i < len(axes): # Evitar erro se o array de axes for menor
                axes[i].set_title("Erro")
                axes[i].axis('off')


    plt.tight_layout()
    visualize_data_filename = f"visualize_data_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(visualize_data_filename)
    st.image(visualize_data_filename, caption='Exemplos do Conjunto de Dados', use_container_width=True)

    with open(visualize_data_filename, "rb") as file:
        btn = st.download_button(
            label="Download da Visualização de Dados",
            data=file,
            file_name=visualize_data_filename,
            mime="image/png",
            key=f"download_visualize_data_{uuid.uuid4()}"
        )
    if btn:
        st.success("Visualização de dados baixada com sucesso!")
    os.remove(visualize_data_filename) # Limpar o arquivo após o uso


def plot_class_distribution(dataset, classes):
    # Extrair os rótulos das classes para todas as imagens no dataset
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset, 'targets'): # e.g. ImageFolder
        labels = dataset.targets
    elif hasattr(dataset, 'labels'): # e.g. algumas datasets do torchvision
         labels = dataset.labels
    else: # fallback se não conseguir extrair labels diretamente
        labels = [label for _, label in dataset]


    df = pd.DataFrame({'Classe': labels})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Classe', data=df, ax=ax, palette="Set2", hue='Classe', dodge=False, legend=False) # dodge=False, legend=False for newer seaborn

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right") # ha="right" for better alignment

    # Remover a legenda se existir (com hue='Classe', uma legenda é criada)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    class_counts = df['Classe'].value_counts().sort_index()
    for i, class_idx in enumerate(class_counts.index):
        count = class_counts[class_idx]
        # Encontrar a posição correta da barra para o texto
        # Se as classes forem numéricas de 0 a N-1, i está correto.
        # Se forem outros valores, precisamos mapear class_idx para a posição da barra.
        bar_position = list(range(len(classes))).index(class_idx) if class_idx in list(range(len(classes))) else i
        ax.text(bar_position, count, str(count), ha='center', va='bottom', fontweight='bold')


    ax.set_title("Distribuição das Classes (Quantidade de Imagens)")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")

    plt.tight_layout()
    class_distribution_filename = f"class_distribution_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(class_distribution_filename)
    st.image(class_distribution_filename, caption='Distribuição das Classes', use_container_width=True)

    with open(class_distribution_filename, "rb") as file:
        btn = st.download_button(
            label="Download da Distribuição das Classes",
            data=file,
            file_name=class_distribution_filename,
            mime="image/png",
            key=f"download_class_distribution_{uuid.uuid4()}"
        )
    if btn:
        st.success("Distribuição das classes baixada com sucesso!")
    os.remove(class_distribution_filename)


def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
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
        st.error(f"Modelo '{model_name}' não suportado.")
        logging.error(f"Modelo não suportado: {model_name}")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    else: # Se fine_tune é True, todas as camadas são treináveis
        for param in model.parameters():
            param.requires_grad = True


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
    # else: # Já tratado acima
    #     st.error("Modelo não suportado.")
    #     logging.error(f"Modelo não suportado na camada final: {model_name}")
    #     return None

    model = model.to(device)
    logging.info(f"Modelo {model_name} carregado e configurado para {num_classes} classes. Fine-tuning: {fine_tune}")
    return model


def apply_transforms_and_get_embeddings(dataset_subset, model, transform, batch_size=16):
    # 'dataset_subset' é um torch.utils.data.Subset
    def pil_collate_fn(batch): # Lida com imagens PIL
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    # Usar o dataset original do Subset para o DataLoader, pois o transform será aplicado manualmente
    temp_dataset_for_loader = torch.utils.data.Subset(dataset_subset.dataset, dataset_subset.indices)

    data_loader = DataLoader(temp_dataset_for_loader, batch_size=batch_size, shuffle=False, collate_fn=pil_collate_fn, num_workers=0) # num_workers=0 para evitar problemas com PIL e multiprocessing em alguns setups
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    augmented_images_list = [] # Armazena tensores já transformados

    # Remover a última camada do modelo para extrair os embeddings
    if isinstance(model, nn.Sequential) and len(list(model.children())) > 1 : # Se já for Sequential (provavelmente camada final modificada)
         model_embedding = nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, 'fc'): # ResNet
        model_embedding = nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, 'classifier'): # DenseNet
        model_embedding = nn.Sequential(*list(model.children())[:-1])
    else:
        st.error("Não foi possível determinar a camada de embedding do modelo.")
        return pd.DataFrame() # Retorna DataFrame vazio em caso de erro

    model_embedding.eval()
    model_embedding.to(device)

    with torch.no_grad():
        current_idx = 0
        for images_pil, labels_batch in data_loader: # images_pil é uma lista de PIL.Image
            images_transformed_tensors = [transform(img) for img in images_pil]
            images_batch_tensor = torch.stack(images_transformed_tensors).to(device)

            embeddings = model_embedding(images_batch_tensor)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()

            embeddings_list.extend(embeddings)
            labels_list.extend(labels_batch.numpy())
            # Guardar imagens augmentadas como tensores CPU (para economizar memória vs numpy arrays grandes)
            augmented_images_list.extend([img_tensor.cpu() for img_tensor in images_batch_tensor])

            # Obter file_paths
            batch_indices_in_subset = list(range(current_idx, current_idx + len(images_pil)))
            original_indices_in_full_dataset = [dataset_subset.indices[i] for i in batch_indices_in_subset]

            if hasattr(dataset_subset.dataset, 'samples'): # ImageFolder
                file_paths = [dataset_subset.dataset.samples[original_idx][0] for original_idx in original_indices_in_full_dataset]
                file_paths_list.extend(file_paths)
            else:
                file_paths_list.extend([f'N/A_idx_{original_idx}' for original_idx in original_indices_in_full_dataset])
            current_idx += len(images_pil)

    df = pd.DataFrame({
        'file_path': file_paths_list,
        'label': labels_list,
        'embedding': list(embeddings_list), # Embeddings como lista de arrays
        'augmented_image_tensor': augmented_images_list # Tensores
    })
    return df


def display_all_augmented_images(df, class_names, model_name, run_id, max_images=None):
    if 'augmented_image_tensor' not in df.columns:
        st.warning("Coluna 'augmented_image_tensor' não encontrada no DataFrame para visualização.")
        return

    display_df = df.copy()
    if max_images is not None:
        display_df = display_df.head(max_images)
        st.write(f"**Visualização das Primeiras {min(max_images, len(display_df))} Imagens após Data Augmentation:**")
    else:
        st.write("**Visualização de Todas as Imagens após Data Augmentation:**")

    num_images = len(display_df)
    if num_images == 0:
        st.write("Nenhuma imagem para exibir.")
        return

    cols_per_row = 5
    rows = (num_images + cols_per_row - 1) // cols_per_row

    for r_idx in range(rows):
        cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            idx = r_idx * cols_per_row + c_idx
            if idx < num_images:
                image_tensor = display_df.iloc[idx]['augmented_image_tensor']
                label = display_df.iloc[idx]['label']

                # Converter tensor para imagem PIL para salvar e exibir
                pil_image = to_pil_image(image_tensor.cpu()) # Garante que está na CPU

                with cols[c_idx]:
                    temp_image_filename = f'augmented_image_{idx}_{uuid.uuid4().hex[:6]}.png'
                    pil_image.save(temp_image_filename)
                    st.image(temp_image_filename, caption=class_names[label], use_container_width=True)

                    with open(temp_image_filename, "rb") as file_bytes:
                        btn = st.download_button(
                            label=f"Download Img {idx}",
                            data=file_bytes,
                            file_name=f"aug_img_{class_names[label]}_{idx}.png",
                            mime="image/png",
                            key=f"download_aug_img_{model_name}_run{run_id}_{idx}_{uuid.uuid4()}"
                        )
                    if btn: # Este if é mais para o backend, o Streamlit lida com o download
                        pass # st.success(f"Imagem {idx} pronta para download!") -> melhor não poluir
                    os.remove(temp_image_filename) # Limpa o arquivo temporário


def visualize_embeddings(df, class_names, model_name, run_id):
    if 'embedding' not in df.columns or df.empty:
        st.warning("DataFrame de embeddings vazio ou coluna 'embedding' ausente.")
        return

    embeddings = np.vstack(df['embedding'].values)
    labels = df['label'].values

    if embeddings.shape[0] < 2:
        st.warning("Não há embeddings suficientes para visualização com PCA (necessário >1).")
        return
    
    n_components = min(2, embeddings.shape[1]) # PCA não pode ter mais componentes que features
    if embeddings.shape[0] <= n_components: # PCA não pode ter mais componentes que amostras
        n_components = embeddings.shape[0] -1
    if n_components < 1:
        st.warning(f"Não é possível aplicar PCA com n_components={n_components}. Insuficientes amostras ou features.")
        return

    pca = PCA(n_components=n_components)
    try:
        embeddings_reduced = pca.fit_transform(embeddings)
    except Exception as e:
        st.error(f"Erro ao aplicar PCA: {e}")
        logging.error(f"Erro PCA: {e}, shape: {embeddings.shape}")
        return

    plot_df_data = {'label': labels}
    if n_components == 2:
        plot_df_data['PC1'] = embeddings_reduced[:, 0]
        plot_df_data['PC2'] = embeddings_reduced[:, 1]
        x_axis, y_axis = 'PC1', 'PC2'
    elif n_components == 1:
        plot_df_data['PC1'] = embeddings_reduced[:, 0]
        plot_df_data['PC2'] = np.zeros_like(embeddings_reduced[:, 0]) # Plotar em 1D vs zeros
        x_axis, y_axis = 'PC1', 'PC2'
        st.info("Visualização de embeddings com PCA (1 componente).")
    else: # n_components == 0
        st.warning("PCA resultou em 0 componentes. Não é possível visualizar.")
        return

    plot_df = pd.DataFrame(plot_df_data)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, hue='label', palette='Set2', legend='full')

    plt.title(f'Visualização dos Embeddings com PCA ({n_components}D) - {model_name} Run {run_id}')
    # Mapear labels numéricos para nomes de classes na legenda
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [class_names[int(text)] for text in sorted(plot_df['label'].unique())], title='Classes')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)' if n_components > 0 else 'Componente Principal 1')
    if n_components == 2:
        plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    else:
        plt.ylabel('')


    embeddings_pca_filename = f"embeddings_pca_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png"
    plt.tight_layout()
    plt.savefig(embeddings_pca_filename)
    st.image(embeddings_pca_filename, caption='Visualização dos Embeddings com PCA', use_container_width=True)

    with open(embeddings_pca_filename, "rb") as file:
        st.download_button(
            label="Download da Visualização dos Embeddings",
            data=file,
            file_name=embeddings_pca_filename,
            mime="image/png",
            key=f"download_embeddings_pca_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(embeddings_pca_filename)
    plt.close()


def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, model_name, run_id):
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title(f'Perda por Época - {model_name} (Execução {run_id})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title(f'Acurácia por Época - {model_name} (Execução {run_id})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    plt.tight_layout()
    plot_filename = f'loss_accuracy_final_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
    fig.savefig(plot_filename)
    st.image(plot_filename, caption='Perda e Acurácia Finais', use_container_width=True)

    with open(plot_filename, "rb") as file:
        st.download_button(
            label="Download Gráficos Perda/Acurácia",
            data=file,
            file_name=plot_filename,
            mime="image/png",
            key=f"download_final_loss_accuracy_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(plot_filename)
    plt.close(fig)


def compute_metrics(model, dataloader, classes, model_name, run_id):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    if not all_labels: # No data to compute metrics
        st.warning("Não há dados no dataloader para calcular métricas.")
        return {
            'Model': model_name, 'Run_ID': run_id, 'Accuracy': np.nan, 'Precision': np.nan,
            'Recall': np.nan, 'F1_Score': np.nan, 'ROC_AUC': np.nan
        }


    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.text("Relatório de Classificação:")
    st.dataframe(report_df) # Use st.dataframe for better table display

    report_filename = f'classification_report_{model_name}_run{run_id}.csv'
    report_df.to_csv(report_filename) # Keep index=True for pandas style
    with open(report_filename, "rb") as file:
        st.download_button(
            label="Download Relatório Classificação (CSV)",
            data=file,
            file_name=report_filename,
            mime="text/csv",
            key=f"download_clf_report_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(report_filename)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax_cm)
    ax_cm.set_xlabel('Predito')
    ax_cm.set_ylabel('Verdadeiro')
    ax_cm.set_title('Matriz de Confusão Normalizada')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
    fig_cm.savefig(cm_filename)
    st.image(cm_filename, caption='Matriz de Confusão Normalizada', use_container_width=True)
    with open(cm_filename, "rb") as file:
        st.download_button(
            label="Download Matriz de Confusão",
            data=file,
            file_name=cm_filename,
            mime="image/png",
            key=f"download_cm_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(cm_filename)
    plt.close(fig_cm)

    roc_auc = np.nan
    if len(classes) == 2:
        # Ensure all_probs has shape (n_samples, 2) for binary case, take prob of positive class
        probs_positive_class = np.array(all_probs)[:, 1] if np.array(all_probs).ndim == 2 else np.array(all_probs)
        fpr, tpr, _ = roc_curve(all_labels, probs_positive_class)
        roc_auc = roc_auc_score(all_labels, probs_positive_class)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('Taxa de Falsos Positivos')
        ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
        ax_roc.set_title('Curva ROC')
        ax_roc.legend(loc='lower right')
        plt.tight_layout()
        roc_filename = f'roc_curve_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
        fig_roc.savefig(roc_filename)
        st.image(roc_filename, caption='Curva ROC', use_container_width=True)
        with open(roc_filename, "rb") as file:
            st.download_button(
                label="Download Curva ROC",
                data=file,
                file_name=roc_filename,
                mime="image/png",
                key=f"download_roc_{model_name}_run{run_id}_{uuid.uuid4()}"
            )
        os.remove(roc_filename)
        plt.close(fig_roc)
    elif len(classes) > 2:
        binarized_labels = label_binarize(all_labels, classes=list(range(len(classes))))
        if binarized_labels.shape[1] == len(classes): # Check if binarization produced correct number of columns
            try:
                roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
                st.write(f"AUC-ROC Média Ponderada (OvR): {roc_auc:.4f}")
                
                auc_text_filename = f'auc_roc_multiclass_{model_name}_run{run_id}.txt'
                with open(auc_text_filename, 'w') as f:
                    f.write(f"AUC-ROC Média Ponderada (OvR): {roc_auc:.4f}\n")
                with open(auc_text_filename, "rb") as file_auc: # Use "rb" for download button
                    st.download_button(
                        label="Download AUC-ROC (Multiclasse)",
                        data=file_auc,
                        file_name=auc_text_filename,
                        mime="text/plain",
                        key=f"download_auc_multiclass_{model_name}_run{run_id}_{uuid.uuid4()}"
                    )
                os.remove(auc_text_filename)
            except ValueError as e:
                st.warning(f"Não foi possível calcular ROC AUC para multiclasse: {e}")
                logging.warning(f"ROC AUC multiclasse error for {model_name} run {run_id}: {e}")
        else:
            st.warning("Binarização de rótulos falhou para ROC AUC multiclasse.")


    metrics_summary = {
        'Model': model_name, 'Run_ID': run_id,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1_Score': report['weighted avg']['f1-score'],
        'ROC_AUC': roc_auc
    }
    metrics_df_summary = pd.DataFrame([metrics_summary])
    metrics_filename = f'metrics_summary_{model_name}_run{run_id}.csv'
    metrics_df_summary.to_csv(metrics_filename, index=False)
    with open(metrics_filename, "rb") as file:
        st.download_button(
            label="Download Sumário de Métricas (CSV)",
            data=file,
            file_name=metrics_filename,
            mime="text/csv",
            key=f"download_metrics_summary_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(metrics_filename)
    return metrics_summary


def error_analysis(model, dataloader, classes, model_name, run_id):
    model.eval()
    misclassified_images_tensors = []
    misclassified_true_labels = []
    misclassified_pred_labels = []
    original_images_pil = [] # Para mostrar a imagem original não transformada

    # Precisamos iterar sobre o dataset original para pegar as imagens PIL
    # e o dataloader para pegar as predições do modelo sobre imagens transformadas
    
    # Temporariamente criar um dataloader que NÃO aplica transformações para pegar as PIL originais
    # e um que aplica para predições. Isso é um pouco complexo.
    # Simplificação: usar as imagens transformadas do dataloader para visualização do erro.
    # O usuário entende que são as imagens como o modelo as viu.

    with torch.no_grad():
        for i, (inputs_transformed, labels) in enumerate(dataloader): # inputs_transformed já são tensores
            inputs_transformed, labels = inputs_transformed.to(device), labels.to(device)
            outputs = model(inputs_transformed)
            _, preds = torch.max(outputs, 1)

            incorrect_mask = preds != labels
            if incorrect_mask.any():
                misclassified_images_tensors.extend(inputs_transformed[incorrect_mask].cpu())
                misclassified_true_labels.extend(labels[incorrect_mask].cpu().numpy())
                misclassified_pred_labels.extend(preds[incorrect_mask].cpu().numpy())
            
            if len(misclassified_images_tensors) >= 5: # Limitar a 5 exemplos
                break
    
    if misclassified_images_tensors:
        st.write("Algumas imagens mal classificadas:")
        num_to_show = min(5, len(misclassified_images_tensors))
        fig, axes = plt.subplots(1, num_to_show, figsize=(3*num_to_show, 3))
        if num_to_show == 1: axes = [axes] # Para lidar com subplot(1,1,...)

        for i in range(num_to_show):
            img_tensor = misclassified_images_tensors[i]
            # Converter tensor para numpy para imshow (C, H, W) -> (H, W, C)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            # Desnormalizar se a normalização foi aplicada nas transformações
            # mean = np.array([0.485, 0.456, 0.406])
            # std = np.array([0.229, 0.224, 0.225])
            # img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1) # Garantir que está no range [0,1]

            axes[i].imshow(img_np)
            true_label_name = classes[misclassified_true_labels[i]]
            pred_label_name = classes[misclassified_pred_labels[i]]
            axes[i].set_title(f"V: {true_label_name}\nP: {pred_label_name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        misclassified_filename = f'misclassified_examples_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
        plt.savefig(misclassified_filename)
        st.image(misclassified_filename, caption='Exemplos de Erros de Classificação', use_container_width=True)

        with open(misclassified_filename, "rb") as file:
            st.download_button(
                label="Download Imagens Mal Classificadas",
                data=file,
                file_name=misclassified_filename,
                mime="image/png",
                key=f"download_misclassified_imgs_{model_name}_run{run_id}_{uuid.uuid4()}"
            )
        os.remove(misclassified_filename)
        plt.close(fig)
    else:
        st.write("Nenhuma imagem mal classificada encontrada nos lotes processados (ou limite de 5 atingido rapidamente).")


def perform_clustering(model, dataloader, classes, model_name, run_id):
    features_list = []
    true_labels_list = []

    # Configurar modelo para extração de features
    if hasattr(model, 'fc'): # ResNet
        model_feat_extractor = nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, 'classifier'): # DenseNet
        model_feat_extractor = nn.Sequential(*list(model.children())[:-1])
    else: # Caso genérico ou já é um extrator
        st.warning("Não foi possível identificar a camada de features automaticamente, usando o modelo como está.")
        model_feat_extractor = model
    
    model_feat_extractor.eval()
    model_feat_extractor.to(device)

    with torch.no_grad():
        for inputs, labels_batch in dataloader:
            inputs = inputs.to(device)
            output_features = model_feat_extractor(inputs)
            output_features = output_features.view(output_features.size(0), -1) # Flatten
            features_list.append(output_features.cpu().numpy())
            true_labels_list.extend(labels_batch.numpy())

    if not features_list:
        st.warning("Nenhuma feature extraída para clustering.")
        return

    features_np = np.vstack(features_list)
    true_labels_np = np.array(true_labels_list)

    if features_np.shape[0] < len(classes): # Menos amostras que clusters desejados
        st.warning(f"Número de amostras ({features_np.shape[0]}) é menor que o número de classes/clusters ({len(classes)}). Clustering pode não ser significativo.")
        # Pode-se optar por não prosseguir ou ajustar n_clusters, aqui vamos prosseguir com cautela
        # return 

    # Redução de dimensionalidade para visualização
    n_components_pca_cluster = min(2, features_np.shape[1], features_np.shape[0] -1 if features_np.shape[0] >1 else 1)
    if n_components_pca_cluster < 1 :
        st.warning(f"PCA para clustering não pode ser aplicado com {n_components_pca_cluster} componentes.")
        # Plotar sem redução pode ser uma opção, mas geralmente é muito denso
        # Aqui, vamos evitar plotar se a redução não for possível para 2D.
        # Se você quiser plotar as features originais, adicione essa lógica.
        features_2d_for_plot = features_np[:, :2] if features_np.shape[1] >=2 else np.hstack((features_np, np.zeros_like(features_np))) if features_np.shape[1]==1 else None
        if features_2d_for_plot is None:
            st.warning("Não foi possível obter features 2D para plotar clusters.")
            return
    else:
        pca = PCA(n_components=n_components_pca_cluster)
        features_2d_for_plot = pca.fit_transform(features_np)


    # KMeans
    # n_init='auto' é o padrão em versões mais recentes do scikit-learn e lida com avisos
    kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init='auto')
    clusters_kmeans = kmeans.fit_predict(features_np)

    # Agglomerative Clustering
    # n_clusters não pode ser maior que n_samples
    n_clusters_agglo = min(len(classes), features_np.shape[0]) if features_np.shape[0] > 0 else len(classes)

    if n_clusters_agglo > 0:
        agglo = AgglomerativeClustering(n_clusters=n_clusters_agglo)
        clusters_agglo = agglo.fit_predict(features_np)
    else:
        clusters_agglo = np.array([]) # Vazio se não puder rodar
        st.warning("Agglomerative Clustering não pode ser executado com 0 clusters.")


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # Gráfico KMeans
    if features_2d_for_plot is not None and clusters_kmeans.size > 0 :
        scatter_kmeans = ax[0].scatter(features_2d_for_plot[:, 0], features_2d_for_plot[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.6)
        # Para legenda de clusters, se desejar:
        # legend_kmeans = ax[0].legend(*scatter_kmeans.legend_elements(), title="KMeans Clusters")
        # ax[0].add_artist(legend_kmeans)
        ax[0].set_title('Clusterização com KMeans')
    else:
        ax[0].set_title('KMeans (sem dados/plot)')


    # Gráfico Agglomerative Clustering
    if features_2d_for_plot is not None and clusters_agglo.size > 0:
        scatter_agglo = ax[1].scatter(features_2d_for_plot[:, 0], features_2d_for_plot[:, 1], c=clusters_agglo, cmap='viridis', alpha=0.6)
        # legend_agglo = ax[1].legend(*scatter_agglo.legend_elements(), title="Agglo Clusters")
        # ax[1].add_artist(legend_agglo)
        ax[1].set_title('Clusterização Hierárquica')
    else:
        ax[1].set_title('Agglo. Clustering (sem dados/plot)')


    plt.tight_layout()
    clustering_filename = f'clustering_results_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(clustering_filename)
    st.image(clustering_filename, caption='Resultados da Clusterização', use_container_width=True)
    with open(clustering_filename, "rb") as file:
        st.download_button(
            label="Download Resultados Clusterização",
            data=file,
            file_name=clustering_filename,
            mime="image/png",
            key=f"download_clustering_plot_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(clustering_filename)
    plt.close(fig)

    # Métricas de Avaliação
    # Garantir que true_labels_np e clusters_kmeans/agglo não estão vazios e têm o mesmo tamanho
    clustering_metrics_data = {'Model': model_name, 'Run_ID': run_id}
    if true_labels_np.size > 0 and clusters_kmeans.size == true_labels_np.size:
        ari_kmeans = adjusted_rand_score(true_labels_np, clusters_kmeans)
        nmi_kmeans = normalized_mutual_info_score(true_labels_np, clusters_kmeans)
        st.write(f"**KMeans** - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
        clustering_metrics_data.update({'KMeans_ARI': ari_kmeans, 'KMeans_NMI': nmi_kmeans})
    else:
        st.write("**KMeans** - Métricas não calculadas (dados insuficientes ou incompatíveis).")
        clustering_metrics_data.update({'KMeans_ARI': np.nan, 'KMeans_NMI': np.nan})

    if true_labels_np.size > 0 and clusters_agglo.size == true_labels_np.size:
        ari_agglo = adjusted_rand_score(true_labels_np, clusters_agglo)
        nmi_agglo = normalized_mutual_info_score(true_labels_np, clusters_agglo)
        st.write(f"**Agglomerative Clustering** - ARI: {ari_agglo:.4f}, NMI: {nmi_agglo:.4f}")
        clustering_metrics_data.update({'Agglomerative_ARI': ari_agglo, 'Agglomerative_NMI': nmi_agglo})
    else:
        st.write("**Agglomerative Clustering** - Métricas não calculadas.")
        clustering_metrics_data.update({'Agglomerative_ARI': np.nan, 'Agglomerative_NMI': np.nan})

    clustering_metrics_df = pd.DataFrame([clustering_metrics_data])
    clustering_metrics_filename = f'clustering_metrics_{model_name}_run{run_id}.csv'
    clustering_metrics_df.to_csv(clustering_metrics_filename, index=False)
    with open(clustering_metrics_filename, "rb") as file:
        st.download_button(
            label="Download Métricas Clusterização (CSV)",
            data=file,
            file_name=clustering_metrics_filename,
            mime="text/csv",
            key=f"download_clustering_metrics_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(clustering_metrics_filename)


def evaluate_image(model, image_pil, classes, applied_transforms): # Passar as transformações usadas
    model.eval()
    # Aplicar as mesmas transformações usadas no teste/validação
    image_tensor = applied_transforms(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    class_name = classes[predicted_idx.item()]
    return class_name, confidence.item()


def visualize_activations(model, image_pil, class_names, model_name, run_id, applied_transforms):
    model.eval()
    input_tensor = applied_transforms(image_pil).unsqueeze(0).to(device)

    target_layer_name = None
    if model_name.startswith('ResNet'):
        # Common target layers for ResNets (last conv block)
        if hasattr(model, 'layer4'): target_layer_name = model.layer4 
        # Add more specific target layers if needed, e.g., model.layer4[-1].conv2 for a specific conv layer
    elif model_name.startswith('DenseNet'):
        if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
            target_layer_name = model.features.denseblock4
        elif hasattr(model, 'features') and hasattr(model.features, 'norm5'): # Last normalization layer in DenseNet features
            target_layer_name = model.features.norm5


    if target_layer_name is None:
        st.error(f"Camada alvo para Grad-CAM não encontrada/configurada para {model_name}.")
        logging.error(f"Grad-CAM target layer not found for {model_name}")
        # Try to find a generic last convolutional layer if possible as a fallback
        # This is complex and error-prone; better to define explicitly per architecture
        # For now, just return if not explicitly defined.
        return

    try:
        cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer_name)
    except Exception as e:
        st.error(f"Erro ao inicializar SmoothGradCAMpp para {model_name} na camada {target_layer_name}: {e}")
        logging.error(f"SmoothGradCAMpp init error for {model_name} layer {target_layer_name}: {e}")
        return


    with torch.set_grad_enabled(True): # Ensure gradients are enabled for CAM
        out = model(input_tensor)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        _, pred_idx = torch.max(probabilities, 1)
        predicted_class_index = pred_idx.item()
        
        try:
            # activation_map is a list of CAM masks (one per class specified, or all if None)
            # We want CAM for the predicted class
            activation_maps_list = cam_extractor(predicted_class_index, out) 
            if not activation_maps_list :
                st.error("Grad-CAM não retornou mapas de ativação.")
                cam_extractor.remove_hooks()
                return
            activation_map_tensor = activation_maps_list[0].squeeze(0) # Squeeze batch dim
        except Exception as e:
            st.error(f"Erro ao gerar mapa de ativação Grad-CAM: {e}")
            logging.error(f"Grad-CAM generation error: {e}")
            cam_extractor.remove_hooks() # Clean up hooks on error
            return


    # Limpar hooks explicitamente após o uso
    cam_extractor.remove_hooks()

    # Overlay mask
    # A imagem original para overlay deve ser a imagem *antes* da normalização (se houver),
    # mas *depois* do resize/crop, para corresponder ao tensor de entrada em dimensões.
    # test_transforms_no_norm = transforms.Compose([t for t in applied_transforms.transforms if not isinstance(t, transforms.Normalize)])
    # pil_image_for_overlay = test_transforms_no_norm(image_pil)

    # Simplificação: usar a imagem PIL original e o overlay pode ter um leve desalinhamento de cores se normalizada.
    # Ou, usar to_pil_image(input_tensor.squeeze().cpu()) que é a imagem como o modelo a viu (normalizada).
    # A segunda opção é mais fiel ao que o modelo processa.
    
    original_image_for_overlay = to_pil_image(input_tensor.squeeze().cpu()) # Imagem como o modelo viu
    activation_map_pil = to_pil_image(activation_map_tensor.cpu(), mode='F') # mode='F' for grayscale float

    result_overlay = overlay_mask(original_image_for_overlay, activation_map_pil, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_pil) # Imagem PIL original fornecida
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')

    ax[1].imshow(result_overlay)
    ax[1].set_title(f'Grad-CAM ({class_names[predicted_class_index]})')
    ax[1].axis('off')

    plt.tight_layout()
    activation_filename = f'grad_cam_viz_{model_name}_run{run_id}_{uuid.uuid4().hex[:8]}.png'
    fig.savefig(activation_filename)
    st.image(activation_filename, caption='Visualização de Grad-CAM', use_container_width=True)

    with open(activation_filename, "rb") as file:
        st.download_button(
            label="Download Visualização Grad-CAM",
            data=file,
            file_name=activation_filename,
            mime="image/png",
            key=f"download_grad_cam_viz_{model_name}_run{run_id}_{uuid.uuid4()}"
        )
    os.remove(activation_filename)
    plt.close(fig)


def perform_anova(data_array, groups_array): # data_array: metric values, groups_array: model names
    # Ensure there are at least two groups and each group has at least one observation
    unique_groups = np.unique(groups_array)
    if len(unique_groups) < 2:
        # logging.warning("ANOVA requires at least two groups.")
        return np.nan, np.nan # Not enough groups
    
    grouped_data = [data_array[groups_array == group] for group in unique_groups]
    # Ensure all groups have data
    grouped_data = [g for g in grouped_data if len(g) > 0]
    if len(grouped_data) < 2: # After filtering empty groups
        # logging.warning("ANOVA requires at least two non-empty groups.")
        return np.nan, np.nan

    try:
        f_val, p_val = stats.f_oneway(*grouped_data)
        return f_val, p_val
    except Exception as e:
        logging.error(f"Erro ao calcular ANOVA: {e}")
        return np.nan, np.nan


def visualize_anova_results(f_val, p_val, metric_name):
    if pd.isna(f_val) or pd.isna(p_val):
        st.write(f"**{metric_name}:** ANOVA não pôde ser calculada (dados insuficientes ou erro).")
        return

    st.write(f"**{metric_name} - ANOVA:** Valor F = {f_val:.4f}, Valor p = {p_val:.4f}")
    if p_val < 0.05:
        st.markdown(f"Para **{metric_name}**, há uma diferença estatisticamente significativa entre as médias dos grupos (p < 0.05).")
    else:
        st.markdown(f"Para **{metric_name}**, não há evidência de uma diferença estatisticamente significativa entre as médias dos grupos (p >= 0.05).")

# --- Fim das funções movidas ---


def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, 
                train_split, valid_split, use_weighted_loss, l2_lambda, patience, 
                model_id=None, run_id=None): # model_id e run_id para chaves únicas
    set_seed(42) # Garante reprodutibilidade para esta execução específica
    logging.info(f"Iniciando treinamento: Modelo {model_name}, Execução {run_id}, Fine-tune: {fine_tune}")

    st.subheader(f"Treinamento do {model_name} - Execução {run_id}")
    st.write(f"**Fine-Tuning Completo:** {'Sim' if fine_tune else 'Não'}")
    # ... (exibir mais configs)

    # Ajustar batch size (já estava lá, mas confirmar)
    if model_name in ['ResNet50', 'DenseNet121'] and batch_size > 8: # Ajustei o limite
        original_bs = batch_size
        batch_size = min(batch_size, 8)
        st.warning(f"Tamanho do lote para {model_name} ajustado de {original_bs} para {batch_size} para evitar OOM.")
        logging.info(f"Batch size para {model_name} ajustado para {batch_size}.")

    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
    except Exception as e:
        st.error(f"Erro ao carregar o dataset de {data_dir}: {e}")
        logging.error(f"Erro carregando ImageFolder de {data_dir}: {e}")
        return None, None, None # Model, classes, metrics

    if not full_dataset.classes or len(full_dataset.classes) < num_classes:
        st.error(f"Número de classes encontradas ({len(full_dataset.classes)}) é incompatível com o especificado ({num_classes}).")
        logging.error(f"Incompatibilidade de classes: {len(full_dataset.classes)} vs {num_classes}")
        return None, None, None
    
    # Visualização de dados inicial (antes de qualquer split)
    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)

    # Divisão dos dados
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices) # Shuffle antes de dividir

    train_end = int(train_split * dataset_size)
    valid_end = train_end + int(valid_split * dataset_size) # valid_split é sobre o total

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    if not train_indices or not valid_indices or not test_indices:
        st.error("Divisão de dados resultou em um ou mais conjuntos vazios. Ajuste os percentuais.")
        logging.error("Conjunto de dados vazio após divisão.")
        return None, None, None

    # Subconjuntos para cada fase (ainda com PIL Images)
    train_subset_pil = torch.utils.data.Subset(full_dataset, train_indices)
    valid_subset_pil = torch.utils.data.Subset(full_dataset, valid_indices)
    test_subset_pil = torch.utils.data.Subset(full_dataset, test_indices)

    # Modelo para extração de embeddings (sempre backbone congelado aqui, só para visualização)
    # Usar o mesmo dropout que o modelo de treinamento
    model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
    if model_for_embeddings is None: return None, None, None

    st.write("**Processando dados de TREINO para visualização (Data Augmentation e Embeddings)...**")
    # Para visualização de augmentations, usar train_transforms. Para embeddings, o transform não deveria ter augmentation aleatório.
    # No entanto, o código original usa train_transforms para embeddings também. Manterei por consistência,
    # mas idealmente, embeddings para visualização seriam de test_transforms.
    train_df_viz = apply_transforms_and_get_embeddings(train_subset_pil, model_for_embeddings, train_transforms, batch_size)
    
    # Limpar modelo de embeddings para liberar memória
    del model_for_embeddings 
    torch.cuda.empty_cache()
    
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if not train_df_viz.empty:
        train_df_viz['class_name'] = train_df_viz['label'].map(idx_to_class)
        st.write("**DataFrame de Amostra do Conjunto de Treinamento (com Augmentation e Embeddings):**")
        st.dataframe(train_df_viz.drop(columns=['augmented_image_tensor', 'embedding']).head()) # Mostrar só algumas colunas/linhas
        display_all_augmented_images(train_df_viz, full_dataset.classes, model_name, run_id, max_images=10) # Mostrar poucas imagens
        visualize_embeddings(train_df_viz, full_dataset.classes, model_name, run_id)
    else:
        st.warning("DataFrame de visualização do treino está vazio.")


    # Datasets finais com transformações para treinamento/avaliação
    train_dataset_transformed = CustomDataset(train_subset_pil, transform=train_transforms)
    valid_dataset_transformed = CustomDataset(valid_subset_pil, transform=test_transforms)
    test_dataset_transformed = CustomDataset(test_subset_pil, transform=test_transforms)

    g = torch.Generator()
    g.manual_seed(42) # Seed para o generator do DataLoader

    # Weighted Loss
    if use_weighted_loss:
        # Calcular pesos baseados na distribuição de classes no CONJUNTO DE TREINO
        train_labels_for_weights = [train_subset_pil.dataset.targets[i] for i in train_subset_pil.indices]
        class_counts = np.bincount(train_labels_for_weights, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6) # Adicionar epsilon para evitar divisão por zero
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        logging.info(f"Perda ponderada utilizada. Pesos: {class_weights_tensor.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logging.info("Perda não ponderada (CrossEntropyLoss padrão) utilizada.")

    num_workers_loader = min(4, os.cpu_count() // 2 if os.cpu_count() else 1) # Heurística para num_workers
    if num_workers_loader < 0: num_workers_loader = 0 # Garantir não negativo

    train_loader = DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, pin_memory=True, num_workers=num_workers_loader)
    valid_loader = DataLoader(valid_dataset_transformed, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, pin_memory=True, num_workers=num_workers_loader)
    test_loader = DataLoader(test_dataset_transformed, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, pin_memory=True, num_workers=num_workers_loader)

    # Carregar modelo para treinamento
    # Passar dropout_p da UI para o modelo
    dropout_p_config = 0.5 # Ou pegar de uma config se for variável
    model = get_model(model_name, num_classes, dropout_p=dropout_p_config, fine_tune=fine_tune)
    if model is None: return None, None, None

    # Otimizador (apenas para parâmetros que exigem gradiente)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    logging.info(f"Otimizador Adam: lr={learning_rate}, weight_decay={l2_lambda}")


    # Chaves para st.session_state
    train_losses_key = f"train_losses_{model_name}_{run_id}" # model_id não é necessário se model_name e run_id já são únicos
    valid_losses_key = f"valid_losses_{model_name}_{run_id}"
    train_accuracies_key = f"train_accuracies_{model_name}_{run_id}"
    valid_accuracies_key = f"valid_accuracies_{model_name}_{run_id}"

    # Inicializar/resetar listas no st.session_state
    st.session_state[train_losses_key] = []
    st.session_state[valid_losses_key] = []
    st.session_state[train_accuracies_key] = []
    st.session_state[valid_accuracies_key] = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    # Placeholders para gráficos dinâmicos e progresso
    plot_placeholder = st.empty()
    progress_bar = st.progress(0)
    epoch_text_status = st.empty()

    for epoch in range(epochs):
        set_seed(42 + epoch) # Variar seed por época para DataLoaders se shuffle=True tem efeito, mas já usamos generator
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1) # Predições
            
            loss.backward() # Backward pass
            optimizer.step() # Otimizar
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        st.session_state[train_losses_key].append(epoch_loss)
        st.session_state[train_accuracies_key].append(epoch_acc.item())

        # Validação
        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(valid_loader.dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_loader.dataset)
        st.session_state[valid_losses_key].append(valid_epoch_loss)
        st.session_state[valid_accuracies_key].append(valid_epoch_acc.item())

        # Atualizar UI (gráficos e progresso)
        # Plotar dinamicamente (opcional: apenas a cada N épocas para reduzir sobrecarga)
        if epoch % 5 == 0 or epoch == epochs -1: # Atualiza a cada 5 épocas ou na última
            fig_dyn, ax_dyn = plt.subplots(1, 2, figsize=(12, 4))
            ax_dyn[0].plot(st.session_state[train_losses_key], label='Treino Loss')
            ax_dyn[0].plot(st.session_state[valid_losses_key], label='Validação Loss')
            ax_dyn[0].legend()
            ax_dyn[0].set_title(f'Perda Época {epoch+1}')
            ax_dyn[1].plot(st.session_state[train_accuracies_key], label='Treino Acc')
            ax_dyn[1].plot(st.session_state[valid_accuracies_key], label='Validação Acc')
            ax_dyn[1].legend()
            ax_dyn[1].set_title(f'Acurácia Época {epoch+1}')
            plt.tight_layout()
            with plot_placeholder.container(): # Usar container para substituir o gráfico
                 st.pyplot(fig_dyn)
            plt.close(fig_dyn) # Fechar para liberar memória

        progress_bar.progress((epoch + 1) / epochs)
        epoch_text_status.text(f"Modelo: {model_name} | Run: {run_id} | Época {epoch+1}/{epochs} | "
                               f"Treino Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                               f"Val Loss: {valid_epoch_loss:.4f} Acc: {valid_epoch_acc:.4f}")

        # Early Stopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict().copy() # Salvar os melhores pesos
            logging.info(f"Epoch {epoch+1}: Melhor validação loss: {best_valid_loss:.4f}. Pesos salvos.")
        else:
            epochs_no_improve += 1
            logging.info(f"Epoch {epoch+1}: Sem melhoria na validação loss por {epochs_no_improve} épocas.")
            if epochs_no_improve >= patience:
                st.write(f"Early stopping na época {epoch+1}!")
                logging.info(f"Early stopping ativado na época {epoch+1}.")
                if best_model_wts:
                    model.load_state_dict(best_model_wts)
                break
    
    # Carregar melhores pesos se o early stopping foi ativado (ou ao final do treinamento se não)
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        logging.info("Melhores pesos carregados no modelo após treinamento/early stopping.")
    
    # Limpar placeholders do Streamlit
    plot_placeholder.empty()
    epoch_text_status.empty()
    progress_bar.empty()

    # Plotar métricas finais
    plot_metrics(st.session_state[train_losses_key], st.session_state[valid_losses_key],
                 st.session_state[train_accuracies_key], st.session_state[valid_accuracies_key],
                 model_name, run_id)

    st.write("**Avaliação Final no Conjunto de Teste**")
    final_metrics = compute_metrics(model, test_loader, full_dataset.classes, model_name, run_id)

    st.write("**Análise de Erros (Test Set)**")
    error_analysis(model, test_loader, full_dataset.classes, model_name, run_id)

    st.write("**Análise de Clusterização (Test Set Features)**")
    perform_clustering(model, test_loader, full_dataset.classes, model_name, run_id)
    
    # Retornar o modelo treinado, as classes e as métricas finais
    # Não deletar o modelo aqui, pois ele será usado/salvo na função `main`
    return model, full_dataset.classes, final_metrics


def main():
    icon_path = "logo.png"
    try:
        page_icon = Image.open(icon_path) if os.path.exists(icon_path) else "🤖"
    except UnidentifiedImageError:
        page_icon = "🤖"
        logging.warning(f"Ícone {icon_path} corrompido, usando fallback.")
    
    st.set_page_config(page_title="Geomaker IA", page_icon=page_icon, layout="wide")

    if os.path.exists('capa.png'):
        try:
            st.image('capa.png', caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_container_width=True)
        except UnidentifiedImageError:
            st.warning("Imagem 'capa.png' não pôde ser carregada ou está corrompida.")
    
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=150)

    st.title("Classificação de Imagens com Aprendizado Profundo")
    st.write("Treine múltiplos modelos, analise clusters e avalie estatisticamente.")

    if 'all_model_metrics' not in st.session_state:
        st.session_state['all_model_metrics'] = []
    if 'trained_models_info' not in st.session_state: # Para guardar info dos modelos para avaliação individual
        st.session_state['trained_models_info'] = []


    st.sidebar.title("Configurações do Treinamento")
    # Parâmetros de UI ... (mantidos como no original)
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, value=2, step=1, key="num_classes")
    fine_tune_all_models = st.sidebar.checkbox("Fine-Tuning Completo para Todos os Modelos", value=False, key="fine_tune_all")
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=100, value=10, step=1, key="epochs") # Reduzido para testes rápidos
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4, key="learning_rate")
    batch_size_ui = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32], index=1, key="batch_size_ui") # 8 default
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="train_split")
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05, key="valid_split") # Sobre o total
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.001, step=0.001, format="%.4f", key="l2_lambda")
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=20, value=5, step=1, key="patience")
    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada", value=True, key="use_weighted_loss")


    st.sidebar.markdown("---")
    if os.path.exists("eu.ico"):
        st.sidebar.image("eu.ico", width=80)
    st.sidebar.markdown("""
    **Produzido por:**
    Projeto Geomaker + IA 
    [DOI:10.5281/zenodo.13910277](https://doi.org/10.5281/zenodo.13910277)
    - **Professor:** Marcelo Claro.
    - **Contatos:** marceloclaro@gmail.com
    """)
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Dispositivo de Treinamento:** {str(device).upper()}")
    st.sidebar.write(f"PyTorch: {torch.__version__}, Torchvision: {torchvision.__version__}, TorchCAM: {torchcam.__version__}")


    if train_split + valid_split >= 1.0:
        st.sidebar.error("A soma dos splits de treinamento e validação deve ser < 1.0 para ter um conjunto de teste.")
        st.stop()

    st.header("Treinamento de Múltiplos Modelos")
    with st.form(key='training_form_multiple'):
        runs_per_model = st.number_input("Execuções por Config. de Modelo:", min_value=1, max_value=5, value=1, step=1, key="runs_per_model")
        
        st.write("Selecione os modelos para treinar:")
        # Usar uma lista de modelos disponíveis
        available_models = ['ResNet18', 'ResNet50', 'DenseNet121']
        selected_models_names = st.multiselect("Modelos:", available_models, default=available_models[:1])

        zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"], key="zip_file_main")
        submit_button_multiple = st.form_submit_button(label='Iniciar Treinamento Múltiplo')

    if submit_button_multiple:
        if not selected_models_names:
            st.error("Por favor, selecione pelo menos um modelo.")
            st.stop()
        if zip_file is None:
            st.error("Por favor, faça upload do arquivo ZIP.")
            st.stop()

        # Limpar métricas e modelos de execuções anteriores
        st.session_state['all_model_metrics'] = []
        st.session_state['trained_models_info'] = [] # Resetar infos de modelos
        
        # Diretório temporário para extrair o ZIP
        # Usar 'with' garante que o diretório temporário seja limpo após o uso
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, zip_file.name)
            with open(zip_path, "wb") as f:
                f.write(zip_file.getbuffer()) # getbuffer() é mais eficiente
            
            data_dir_extracted = os.path.join(temp_dir, "extracted_data")
            os.makedirs(data_dir_extracted, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir_extracted)
                logging.info(f"Arquivo ZIP extraído para {data_dir_extracted}")
                
                # Verificar se o diretório extraído contém subdiretórios de classes
                # ImageFolder espera uma estrutura como: extracted_data/class_a/img1.jpg, extracted_data/class_b/img2.jpg
                # Se o ZIP contiver um único diretório raiz com as classes dentro, precisamos ajustar data_dir_extracted
                extracted_items = os.listdir(data_dir_extracted)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(data_dir_extracted, extracted_items[0])):
                    # Assumir que o ZIP continha um diretório raiz, e as classes estão dentro dele
                    data_dir_for_imagefolder = os.path.join(data_dir_extracted, extracted_items[0])
                    logging.info(f"Ajustando data_dir para ImageFolder: {data_dir_for_imagefolder}")
                else:
                    data_dir_for_imagefolder = data_dir_extracted

                # Verificar se data_dir_for_imagefolder realmente contém subdiretórios (classes)
                class_dirs = [d for d in os.listdir(data_dir_for_imagefolder) if os.path.isdir(os.path.join(data_dir_for_imagefolder, d))]
                if not class_dirs:
                    st.error(f"Nenhum subdiretório de classe encontrado em '{data_dir_for_imagefolder}'. "
                             "Verifique a estrutura do seu arquivo ZIP. "
                             "Deve ser: seu_zip.zip -> (opcional: pasta_raiz) -> classe_A/imgs..., classe_B/imgs...")
                    logging.error(f"Nenhum diretório de classe em {data_dir_for_imagefolder}")
                    st.stop()


            except zipfile.BadZipFile:
                st.error("Arquivo ZIP inválido ou corrompido.")
                logging.error("BadZipFile ao extrair.")
                st.stop()
            except Exception as e:
                st.error(f"Erro ao processar arquivo ZIP: {e}")
                logging.error(f"Erro processando ZIP: {e}")
                st.stop()


            for model_idx, model_name_iter in enumerate(selected_models_names):
                for run_iter in range(1, runs_per_model + 1):
                    st.markdown(f"--- \n ### Treinando: {model_name_iter} - Execução {run_iter}/{runs_per_model}")
                    
                    # Aqui você pode adicionar lógica para variar fine_tune por modelo se desejar
                    # Por agora, usamos o global fine_tune_all_models
                    
                    trained_model_obj, classes_list, run_metrics = train_model(
                        data_dir=data_dir_for_imagefolder, # Usar o diretório extraído
                        num_classes=num_classes,
                        model_name=model_name_iter,
                        fine_tune=fine_tune_all_models, # Usar a configuração global da UI
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size_ui, # Usar o valor da UI
                        train_split=train_split,
                        valid_split=valid_split,
                        use_weighted_loss=use_weighted_loss,
                        l2_lambda=l2_lambda,
                        patience=patience,
                        model_id=f"{model_name_iter.lower()}", # model_id pode ser o nome do modelo
                        run_id=run_iter
                    )

                    if trained_model_obj and classes_list and run_metrics:
                        st.session_state['all_model_metrics'].append(run_metrics)
                        
                        # Salvar modelo e informações para avaliação posterior
                        model_filename = f'{model_name_iter}_run{run_iter}_{uuid.uuid4().hex[:8]}.pth'
                        torch.save(trained_model_obj.state_dict(), model_filename)
                        st.success(f"Modelo {model_name_iter} (Run {run_iter}) treinado e salvo como `{model_filename}`.")
                        logging.info(f"Modelo salvo: {model_filename}")
                        
                        st.session_state['trained_models_info'].append({
                            'name': f"{model_name_iter} Run {run_iter}",
                            'model_type': model_name_iter, # e.g. ResNet18
                            'path': model_filename,
                            'classes': classes_list,
                            'num_classes': len(classes_list),
                            'metrics': run_metrics # Guardar métricas desta execução
                        })

                        # Oferecer download do modelo salvo
                        with open(model_filename, "rb") as fp:
                            st.download_button(
                                label=f"Download Modelo {model_name_iter} Run {run_iter}",
                                data=fp,
                                file_name=model_filename,
                                mime="application/octet-stream",
                                key=f"dl_model_{model_name_iter}_{run_iter}_{uuid.uuid4()}"
                            )
                        # Não remover o arquivo aqui se quiser usá-lo na seção de avaliação individual.
                        # Os arquivos serão limpos quando o app Streamlit for fechado/reexecutado se não forem persistidos.
                        # Para persistência real, salvar em local não temporário.
                    else:
                        st.error(f"Falha no treinamento do modelo {model_name_iter}, Execução {run_iter}.")
                        logging.error(f"Falha treinamento: {model_name_iter}, Run {run_iter}")
                    
                    # Limpar memória da GPU após cada treinamento de modelo/run
                    if 'trained_model_obj' in locals() and trained_model_obj is not None: del trained_model_obj
                    torch.cuda.empty_cache()
        # Fim do `with tempfile.TemporaryDirectory()` - arquivos no temp_dir são removidos
        # Modelos .pth salvos fora do temp_dir (no diretório atual do script) persistirão
        # até o app ser fechado ou o contêiner/VM ser reiniciado.

        # Análise Estatística Agregada
        if st.session_state['all_model_metrics']:
            st.markdown("--- \n ## Análise Estatística Agregada dos Modelos")
            metrics_df_all = pd.DataFrame(st.session_state['all_model_metrics'])
            st.dataframe(metrics_df_all)

            # Salvar e oferecer download de todas as métricas
            all_metrics_filename = f'all_model_metrics_summary_{uuid.uuid4().hex[:8]}.csv'
            metrics_df_all.to_csv(all_metrics_filename, index=False)
            with open(all_metrics_filename, "rb") as fp:
                st.download_button(
                    label="Download Todas as Métricas (CSV)",
                    data=fp,
                    file_name=all_metrics_filename,
                    mime="text/csv",
                    key=f"dl_all_metrics_{uuid.uuid4()}"
                )
            # os.remove(all_metrics_filename) # Opcional: remover após download

            # ANOVA e Tukey
            for metric_col in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
                if metric_col in metrics_df_all.columns:
                    metric_data = metrics_df_all[['Model', metric_col]].copy() # Model é o nome do modelo
                    metric_data.dropna(subset=[metric_col], inplace=True) # Remover NaNs da métrica
                    
                    if metric_data.shape[0] > 1 and metric_data['Model'].nunique() > 1:
                        # Verificar se há dados suficientes por grupo para ANOVA/Tukey
                        group_counts = metric_data['Model'].value_counts()
                        valid_groups_for_stat = group_counts[group_counts >= 2].index # Grupos com pelo menos 2 observações
                        
                        if len(valid_groups_for_stat) >= 2: # Pelo menos 2 grupos válidos
                            filtered_metric_data = metric_data[metric_data['Model'].isin(valid_groups_for_stat)]
                            
                            f_val, p_val = perform_anova(filtered_metric_data[metric_col].values, filtered_metric_data['Model'].values)
                            visualize_anova_results(f_val, p_val, metric_col)

                            if p_val < 0.05: # Se ANOVA é significativa, rodar Tukey
                                try:
                                    tukey_results = pairwise_tukeyhsd(endog=filtered_metric_data[metric_col],
                                                                      groups=filtered_metric_data['Model'],
                                                                      alpha=0.05)
                                    st.text_area(f"Teste Tukey HSD para {metric_col}:", value=str(tukey_results.summary()), height=200,
                                                 key=f"tukey_summary_{metric_col}_{uuid.uuid4()}")
                                    
                                    tukey_filename = f'tukey_summary_{metric_col}_{uuid.uuid4().hex[:8]}.txt'
                                    with open(tukey_filename, 'w') as f_tukey:
                                        f_tukey.write(str(tukey_results.summary()))
                                    with open(tukey_filename, "rb") as fp_tukey:
                                        st.download_button(
                                            label=f"Download Resumo Tukey ({metric_col})",
                                            data=fp_tukey,
                                            file_name=tukey_filename,
                                            mime="text/plain",
                                            key=f"dl_tukey_{metric_col}_{uuid.uuid4()}"
                                        )
                                    # os.remove(tukey_filename)
                                except Exception as e_tukey:
                                    st.warning(f"Erro ao calcular Tukey HSD para {metric_col}: {e_tukey}")
                                    logging.warning(f"Tukey HSD error for {metric_col}: {e_tukey}")
                        else:
                            st.write(f"**{metric_col}:** Não há grupos suficientes com múltiplas observações para ANOVA/Tukey.")
                    else:
                        st.write(f"**{metric_col}:** Dados insuficientes para análise estatística entre modelos (precisa de >1 modelo com dados).")
            st.markdown("---")


    # Seção para Avaliação de Imagem Individual e Grad-CAM
    if st.session_state.get('trained_models_info'):
        st.header("Avaliar Imagem Individual e Visualizar Ativações (Grad-CAM)")
        
        model_options = {info['name']: info for info in st.session_state['trained_models_info']}
        selected_model_name_eval = st.selectbox("Escolha um modelo treinado para avaliação:", list(model_options.keys()), key="model_select_eval")

        uploaded_image_eval = st.file_uploader("Upload de uma imagem para avaliação", type=["png", "jpg", "jpeg"], key="img_eval_uploader")

        if uploaded_image_eval and selected_model_name_eval:
            model_info = model_options[selected_model_name_eval]
            
            # Carregar o modelo selecionado
            try:
                # Recriar a arquitetura do modelo
                # dropout_p usado no get_model deve ser o mesmo do treinamento se salvo apenas state_dict
                eval_model = get_model(model_info['model_type'], model_info['num_classes'], dropout_p=0.5, fine_tune=True) # Assumir fine_tune=True para carregar todos os pesos
                eval_model.load_state_dict(torch.load(model_info['path'], map_location=device))
                eval_model.to(device)
                eval_model.eval()
                st.success(f"Modelo '{model_info['name']}' carregado para avaliação.")
            except Exception as e:
                st.error(f"Erro ao carregar modelo '{model_info['name']}': {e}")
                logging.error(f"Erro carregando modelo {model_info['path']}: {e}")
                st.stop()

            pil_image = Image.open(uploaded_image_eval).convert("RGB")
            
            # Avaliar
            pred_class, pred_conf = evaluate_image(eval_model, pil_image, model_info['classes'], test_transforms) # Usar test_transforms
            st.write(f"**Predição:** {pred_class} (Confiança: {pred_conf:.2%})")
            
            # Grad-CAM
            st.write("**Visualização Grad-CAM:**")
            # O run_id aqui é para o nome do arquivo do Grad-CAM, pode ser um UUID simples
            visualize_activations(eval_model, pil_image, model_info['classes'], model_info['model_type'], 
                                  run_id=f"eval_{uuid.uuid4().hex[:4]}", applied_transforms=test_transforms) # Usar test_transforms

            del eval_model # Liberar memória
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
