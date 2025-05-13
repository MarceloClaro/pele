# -*- coding: utf-8 -*-
# =============================================================================
# ATENÇÃO: SOBRE O ERRO "RuntimeError: Tried to instantiate class '__path__._path'"
#
# Se você encontrar este erro no log do Streamlit, ele é causado pelo
# file watcher do Streamlit tentando inspecionar o módulo `torch.classes`.
#
# SOLUÇÃO:
# 1. No seu repositório GitHub (ex: 'pele'), crie uma pasta chamada `.streamlit`.
# 2. Dentro da pasta `.streamlit`, crie um arquivo chamado `config.toml`.
# 3. Adicione o seguinte ao `config.toml`, substituindo o caminho pelo
#    caminho real para o diretório `torch` no ambiente do Streamlit Cloud
#    (conforme o log, parece ser: /home/adminuser/venv/lib/python3.12/site-packages/torch):
#
#    [server]
#    folderWatchBlacklist = ["/home/adminuser/venv/lib/python3.12/site-packages/torch"]
#
# 4. Faça commit e push dessas alterações para o seu repositório GitHub.
# 5. Reinicie/Reimplante seu aplicativo no Streamlit Cloud.
# =============================================================================

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
from torch.utils.data import DataLoader # random_split é usado diretamente de torch.utils.data
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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
import logging
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, InterpolationMode
import warnings
from scipy import stats
import torchvision
import torchcam
import io # Para BytesIO no download de imagens

# Supressão dos avisos
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Using categorical units to plot a list of strings that are all parsable as floats or dates.")
warnings.filterwarnings("ignore", category=UserWarning, message="set_ticklabels() should only be used with a fixed number of ticks") # Adicionado para o warning específico


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
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ], p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
])

test_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_subset, transform=None):
        self.dataset_subset = dataset_subset
        self.transform = transform

    def __len__(self):
        return len(self.dataset_subset)

    def __getitem__(self, idx):
        image, label = self.dataset_subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def visualize_data(dataset, classes):
    st.write("Visualização de algumas imagens do conjunto de dados:")
    num_samples_to_show = min(10, len(dataset))
    if num_samples_to_show == 0:
        st.warning("Não há imagens para visualização.")
        return

    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(2 * num_samples_to_show, 3))
    if num_samples_to_show == 1: axes = [axes]

    indices_to_sample = random.sample(range(len(dataset)), num_samples_to_show)

    for i, data_idx in enumerate(indices_to_sample):
        try:
            image, label_idx = dataset[data_idx]
            if isinstance(image, torch.Tensor):
                image_pil = to_pil_image(image.cpu()) # Garante CPU
            elif isinstance(image, Image.Image):
                image_pil = image
            else:
                axes[i].set_title("Erro Formato"); axes[i].axis('off'); continue
            axes[i].imshow(np.array(image_pil)); axes[i].set_title(classes[label_idx]); axes[i].axis('off')
        except Exception as e:
            logging.error(f"Erro ao visualizar imagem no índice {data_idx}: {e}")
            if i < len(axes): axes[i].set_title("Erro"); axes[i].axis('off')

    plt.tight_layout()
    filename = f"visualize_data_{uuid.uuid4().hex[:8]}.png"
    try:
        plt.savefig(filename); st.image(filename, caption='Exemplos do Dataset', use_container_width=True)
        with open(filename, "rb") as f: st.download_button("DL Visualização", f, filename, "image/png", key=f"dl_viz_{uuid.uuid4()}")
    except Exception as e: st.error(f"Erro ao salvar/mostrar gráfico: {e}")
    finally:
        if os.path.exists(filename): os.remove(filename)
        plt.close(fig)


def plot_class_distribution(dataset, classes):
    if isinstance(dataset, torch.utils.data.Subset): all_labels = [dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset, 'targets'): all_labels = dataset.targets
    else: all_labels = [label for _, label in dataset]
    if not all_labels: st.warning("Sem rótulos para distribuição."); return

    df = pd.DataFrame({'label_idx': all_labels})
    df['Classe'] = df['label_idx'].apply(lambda x: classes[x])

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 6))
    plot_obj = sns.countplot(x='Classe', data=df, ax=ax, palette="viridis", hue='Classe', dodge=False, legend=False, order=classes)
    plot_obj.set_xticklabels(plot_obj.get_xticklabels(), rotation=45, ha="right") # Correção aqui

    class_counts = df['Classe'].value_counts().reindex(classes, fill_value=0)
    for i, class_name_iter in enumerate(classes):
        count = class_counts[class_name_iter]
        ax.text(i, count + (0.01 * max(class_counts) if max(class_counts) > 0 else 1), str(count), ha='center', va='bottom', fontweight='bold')

    ax.set_title("Distribuição das Classes"); ax.set_xlabel("Classes"); ax.set_ylabel("Número de Imagens")
    plt.tight_layout()
    filename = f"class_distribution_{uuid.uuid4().hex[:8]}.png"
    try:
        plt.savefig(filename); st.image(filename, caption='Distribuição das Classes', use_container_width=True)
        with open(filename, "rb") as f: st.download_button("DL Distribuição", f, filename, "image/png", key=f"dl_dist_{uuid.uuid4()}")
    except Exception as e: st.error(f"Erro ao salvar/mostrar gráfico: {e}")
    finally:
        if os.path.exists(filename): os.remove(filename)
        plt.close(fig)


def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    weights_map = {'ResNet18': ResNet18_Weights.DEFAULT, 'ResNet50': ResNet50_Weights.DEFAULT, 'DenseNet121': DenseNet121_Weights.DEFAULT}
    model_fn_map = {'ResNet18': resnet18, 'ResNet50': resnet50, 'DenseNet121': densenet121}
    if model_name not in model_fn_map: st.error(f"Modelo '{model_name}' não suportado."); return None
    model = model_fn_map[model_name](weights=weights_map[model_name])
    for param in model.parameters(): param.requires_grad = fine_tune
    if model_name.startswith('ResNet'): model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(model.fc.in_features, num_classes))
    elif model_name.startswith('DenseNet'): model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(model.classifier.in_features, num_classes))
    if not fine_tune: # Garantir que a nova camada classificadora seja treinável
        if hasattr(model, 'fc'):
            for param in model.fc.parameters(): param.requires_grad = True
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters(): param.requires_grad = True
    model = model.to(device)
    logging.info(f"Modelo {model_name} ({num_classes} classes). FineTune: {fine_tune}. Dropout: {dropout_p}")
    return model


def apply_transforms_and_get_embeddings(dataset_subset, model_for_embeddings, image_transform, batch_size=16):
    def pil_collate_fn(batch): return list(zip(*batch))[0], torch.tensor(list(zip(*batch))[1]) # Imagens PIL, Labels
    temp_loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, collate_fn=pil_collate_fn, num_workers=0)
    embeddings_l, labels_l, paths_l, aug_tensors_l = [], [], [], []
    if hasattr(model_for_embeddings, 'fc'): extractor = nn.Sequential(*list(model_for_embeddings.children())[:-1])
    elif hasattr(model_for_embeddings, 'classifier'): extractor = nn.Sequential(*list(model_for_embeddings.children())[:-1])
    else: st.error("Modelo desconhecido para embeddings."); return pd.DataFrame()
    extractor.eval().to(device)
    with torch.no_grad():
        ptr = 0
        for pil_imgs, lbls in temp_loader:
            t_tensors = torch.stack([image_transform(img) for img in pil_imgs]).to(device)
            embeds = extractor(t_tensors).view(t_tensors.size(0), -1).cpu().numpy()
            embeddings_l.extend(list(embeds)); labels_l.extend(lbls.numpy()); aug_tensors_l.extend([t.cpu() for t in t_tensors])
            orig_indices = [dataset_subset.indices[i] for i in range(ptr, ptr + len(pil_imgs))]
            if hasattr(dataset_subset.dataset, 'samples'): paths_l.extend([dataset_subset.dataset.samples[oi][0] for oi in orig_indices])
            else: paths_l.extend([f"N/A_{oi}" for oi in orig_indices])
            ptr += len(pil_imgs)
    return pd.DataFrame({'file_path': paths_l, 'label': labels_l, 'embedding': embeddings_l, 'augmented_image_tensor': aug_tensors_l})


def display_all_augmented_images(df, class_map, model_name, run_id, max_imgs=10):
    if df.empty or 'augmented_image_tensor' not in df.columns: st.write("Sem imagens aumentadas."); return
    sample_df = df.sample(min(max_imgs, len(df))) if len(df) > max_imgs else df
    st.write(f"**Amostra de Imagens Aumentadas ({len(sample_df)} de {len(df)}):**")
    cols_per_row = 5; num_rows = (len(sample_df) + cols_per_row - 1) // cols_per_row
    for r_idx in range(num_rows):
        cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            item_idx = r_idx * cols_per_row + c_idx
            if item_idx < len(sample_df):
                row = sample_df.iloc[item_idx]; img_t = row['augmented_image_tensor']; lbl_idx = row['label']
                pil_img = to_pil_image(img_t.cpu())
                with cols[c_idx]:
                    st.image(pil_img, caption=class_map[lbl_idx], use_container_width=True)
                    img_b = io.BytesIO(); pil_img.save(img_b, "PNG"); img_b.seek(0)
                    st.download_button("DL", img_b, f"aug_{model_name}_r{run_id}_{item_idx}.png", "image/png", key=f"dl_aug_{model_name}_{run_id}_{item_idx}_{uuid.uuid4()}")


def visualize_embeddings(df, class_map, model_name, run_id):
    if df.empty or 'embedding' not in df.columns: st.write("Sem embeddings."); return
    embeds_arr = np.vstack(df['embedding'].values); labels_arr = df['label'].values
    if embeds_arr.shape[0] < 2: st.warning("Poucos embeddings para PCA."); return
    n_comp = min(2, embeds_arr.shape[1], embeds_arr.shape[0] - 1 if embeds_arr.shape[0] > 1 else 1)
    if n_comp < 1: st.warning(f"PCA não aplicável com {n_comp} componentes."); return
    pca = PCA(n_components=n_comp)
    try: reduced_embeds = pca.fit_transform(embeds_arr)
    except Exception as e: st.error(f"Erro PCA: {e}"); return
    plot_d = {'label': labels_arr}; cols = []
    for i in range(n_comp): pc_n = f'PC{i+1}'; plot_d[pc_n] = reduced_embeds[:, i]; var_ex = pca.explained_variance_ratio_[i]*100; cols.append(f"{pc_n} ({var_ex:.1f}%)")
    plot_df = pd.DataFrame(plot_d); plot_df['Classe'] = plot_df['label'].apply(lambda x: class_map[x])
    fig, ax = plt.subplots(figsize=(10, 7))
    if n_comp == 1: plot_df['y'] = 0; sns.scatterplot(data=plot_df, x=cols[0], y='y', hue='Classe', palette='viridis', legend='full', ax=ax); ax.set_yticklabels([]); ax.set_ylabel("")
    else: sns.scatterplot(data=plot_df, x=cols[0], y=cols[1], hue='Classe', palette='viridis', legend='full', ax=ax); ax.set_ylabel(cols[1])
    ax.set_xlabel(cols[0]); ax.set_title(f'Embeddings PCA ({model_name} R{run_id})')
    filename = f"pca_embed_{model_name}_r{run_id}_{uuid.uuid4().hex[:8]}.png"
    try:
        plt.tight_layout(); plt.savefig(filename); st.image(filename, caption='Embeddings PCA', use_container_width=True)
        with open(filename, "rb") as f: st.download_button("DL PCA", f, filename, "image/png", key=f"dl_pca_{model_name}_{run_id}_{uuid.uuid4()}")
    except Exception as e: st.error(f"Erro ao salvar/mostrar PCA: {e}")
    finally:
        if os.path.exists(filename): os.remove(filename)
        plt.close(fig)


def plot_metrics(train_l, valid_l, train_a, valid_a, model_name, run_id_epoch_str): # run_id_epoch_str pode incluir a época para plots dinâmicos
    epochs_r = range(1, len(train_l) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) # Reduzido para caber melhor
    ax[0].plot(epochs_r, train_l, ".-", label='Treino Loss'); ax[0].plot(epochs_r, valid_l, ".-", label='Val Loss'); ax[0].set_title(f'Perda ({model_name} {run_id_epoch_str})'); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(epochs_r, train_a, ".-", label='Treino Acc'); ax[1].plot(epochs_r, valid_a, ".-", label='Val Acc'); ax[1].set_title(f'Acurácia ({model_name} {run_id_epoch_str})'); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    # Se for plot dinâmico, não salvar arquivo, apenas mostrar com st.pyplot()
    if "ep" in run_id_epoch_str: # Indicador de plot dinâmico
        st.pyplot(fig)
        plt.close(fig) # Fechar para liberar memória
    else: # Plot final
        filename = f'curves_{model_name}_r{run_id_epoch_str}_{uuid.uuid4().hex[:8]}.png'
        try:
            fig.savefig(filename); st.image(filename, caption='Curvas Métricas', use_container_width=True)
            with open(filename, "rb") as f: st.download_button("DL Curvas", f, filename, "image/png", key=f"dl_curves_{model_name}_{run_id_epoch_str}_{uuid.uuid4()}")
        except Exception as e: st.error(f"Erro ao salvar/mostrar curvas: {e}")
        finally:
            if os.path.exists(filename): os.remove(filename)
            plt.close(fig)


def compute_metrics_detailed(model, dataloader, class_names, model_name, run_id):
    model.eval(); preds_idx, labels_idx, probs_l = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs_dev, labels_dev = inputs.to(device), labels.to(device)
            outputs = model(inputs_dev); probabilities = torch.nn.functional.softmax(outputs, dim=1); _, p_idx = torch.max(outputs, 1)
            preds_idx.extend(p_idx.cpu().numpy()); labels_idx.extend(labels_dev.cpu().numpy()); probs_l.extend(probabilities.cpu().numpy())
    if not labels_idx: st.warning("Sem dados para métricas."); return {'Model': model_name, 'Run_ID': run_id, **{m: np.nan for m in ['Accuracy','Precision','Recall','F1_Score','ROC_AUC']}}
    
    report = classification_report(labels_idx, preds_idx, target_names=class_names, output_dict=True, zero_division=0)
    st.text("Relatório Classificação (Teste):"); st.dataframe(pd.DataFrame(report).transpose().style.format("{:.3f}"))
    # ... (CM, ROC, salvar CSVs, downloads - Lógica similar à anterior)
    cm = confusion_matrix(labels_idx, preds_idx, normalize='true')
    fig_cm, ax_cm = plt.subplots(figsize=(max(5,len(class_names)*0.7), max(4,len(class_names)*0.5))) # Ajustar tamanho
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel('Predito'); ax_cm.set_ylabel('Verdadeiro'); ax_cm.set_title('Matriz de Confusão')
    plt.tight_layout(); cm_fn = f'cm_{model_name}_r{run_id}_{uuid.uuid4().hex[:8]}.png'
    try: fig_cm.savefig(cm_fn); st.image(cm_fn, caption='Matriz Confusão'); # ... download ...
    finally:
        if os.path.exists(cm_fn): os.remove(cm_fn)
        plt.close(fig_cm)

    roc_auc = np.nan; probs_np = np.array(probs_l)
    if len(class_names) == 2:
        roc_auc = roc_auc_score(labels_idx, probs_np[:,1]) # ... plot ROC ...
    elif len(class_names) > 2:
        try: roc_auc = roc_auc_score(label_binarize(labels_idx, classes=list(range(len(class_names)))), probs_np, average='weighted', multi_class='ovr'); st.write(f"AUC-ROC (OvR): {roc_auc:.4f}")
        except Exception as e: st.warning(f"Erro ROC AUC multiclasse: {e}")
    
    metrics_out = {'Model': model_name, 'Run_ID': run_id, 'Accuracy': report['accuracy'], 'Precision': report['weighted avg']['precision'], 'Recall': report['weighted avg']['recall'], 'F1_Score': report['weighted avg']['f1-score'], 'ROC_AUC': roc_auc}
    return metrics_out

# ... (error_analysis, perform_clustering, evaluate_image, visualize_activations, perform_anova, visualize_anova_results - Lógica similar)
# Essas funções são complexas e precisam ser adaptadas/completadas com base na sua versão anterior,
# focando em robustez, limpeza de arquivos e clareza. Por brevidade, não as reescrevo inteiramente aqui.

def train_model(data_dir, num_classes_cfg, model_name_cfg, fine_tune_cfg, epochs_cfg, lr_cfg, 
                bs_cfg, train_split_cfg, valid_split_cfg, use_weighted_loss_cfg, l2_cfg, 
                patience_cfg, run_id_cfg, dropout_p_cfg=0.5):
    set_seed(42 + run_id_cfg)
    logging.info(f"Treino: {model_name_cfg} R{run_id_cfg}. FT:{fine_tune_cfg}. BS:{bs_cfg}")
    st.markdown(f"#### Treinando: {model_name_cfg} (Run {run_id_cfg})")

    try: full_ds_pil = datasets.ImageFolder(root=data_dir)
    except Exception as e: st.error(f"Erro dataset '{data_dir}': {e}"); return None,None,None
    if len(full_ds_pil.classes) != num_classes_cfg: st.error(f"Dataset tem {len(full_ds_pil.classes)} classes, config tem {num_classes_cfg}."); return None,None,None
    
    with st.expander("Dados Iniciais (Completo)", expanded=False):
        visualize_data(full_ds_pil, full_ds_pil.classes)
        plot_class_distribution(full_ds_pil, full_ds_pil.classes)

    total_sz = len(full_ds_pil); train_sz = int(train_split_cfg*total_sz); valid_sz = int(valid_split_cfg*total_sz); test_sz = total_sz-train_sz-valid_sz
    if not all([train_sz, valid_sz, test_sz]): st.error(f"Split inválido: T={train_sz},V={valid_sz},Ts={test_sz}."); return None,None,None
    
    train_sub_pil, valid_sub_pil, test_sub_pil = torch.utils.data.random_split(full_ds_pil, [train_sz,valid_sz,test_sz], generator=torch.Generator().manual_seed(42))
    logging.info(f"Splits: T={len(train_sub_pil)},V={len(valid_sub_pil)},Ts={len(test_sub_pil)}")

    if st.checkbox(f"Ver Augment/Embeddings para {model_name_cfg} R{run_id_cfg}?", value=False, key=f"cb_viz_{model_name_cfg}_{run_id_cfg}"):
        with st.spinner("Processando visualizações..."):
            model_viz = get_model(model_name_cfg, num_classes_cfg, dropout_p_cfg, fine_tune=False)
            if model_viz:
                df_viz = apply_transforms_and_get_embeddings(train_sub_pil, model_viz, train_transforms, bs_cfg)
                if not df_viz.empty: display_all_augmented_images(df_viz, full_ds_pil.classes, model_name_cfg, run_id_cfg, max_imgs=3); visualize_embeddings(df_viz, full_ds_pil.classes, model_name_cfg, run_id_cfg)
                del model_viz; torch.cuda.empty_cache()

    train_ds = CustomDataset(train_sub_pil, train_transforms); valid_ds = CustomDataset(valid_sub_pil, test_transforms); test_ds = CustomDataset(test_sub_pil, test_transforms)
    n_workers = min(2, os.cpu_count()//2 if os.cpu_count() else 0) # Reduzido para Streamlit Cloud
    train_load = DataLoader(train_ds, bs_cfg, shuffle=True, num_workers=n_workers, pin_memory=True, worker_init_fn=seed_worker)
    valid_load = DataLoader(valid_ds, bs_cfg, shuffle=False, num_workers=n_workers, pin_memory=True)
    test_load = DataLoader(test_ds, bs_cfg, shuffle=False, num_workers=n_workers, pin_memory=True)

    if use_weighted_loss_cfg:
        counts = np.bincount([train_sub_pil.dataset.targets[i] for i in train_sub_pil.indices], minlength=num_classes_cfg)
        w = 1.0/(counts+1e-6); criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(w).to(device)); logging.info(f"Loss ponderada: {w}")
    else: criterion = nn.CrossEntropyLoss()
    
    model_train = get_model(model_name_cfg, num_classes_cfg, dropout_p_cfg, fine_tune_cfg)
    if not model_train: return None,None,None
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model_train.parameters()), lr_cfg, weight_decay=l2_cfg)

    history = {'train_loss':[],'valid_loss':[],'train_acc':[],'valid_acc':[]}
    best_vloss = float('inf'); epochs_no_imp = 0; best_state = None
    prog_bar = st.progress(0); status_txt = st.empty(); plot_ph = st.empty()

    for epoch in range(epochs_cfg):
        model_train.train(); t_loss, t_corr = 0.0, 0
        for inputs, labels in train_load:
            inputs_dev, labels_dev = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model_train(inputs_dev); loss = criterion(outputs, labels_dev)
            _, preds = torch.max(outputs,1); loss.backward(); optimizer.step()
            t_loss += loss.item()*inputs.size(0); t_corr += torch.sum(preds==labels_dev.data).item()
            # if os.getenv("STREAMLIT_DEBUG_FAST_EPOCH"): break # Para debug rápido
        history['train_loss'].append(t_loss/len(train_ds)); history['train_acc'].append(t_corr/len(train_ds))
        
        model_train.eval(); v_loss, v_corr = 0.0, 0
        with torch.no_grad():
            for inputs, labels in valid_load:
                inputs_dev, labels_dev = inputs.to(device), labels.to(device)
                outputs = model_train(inputs_dev); loss = criterion(outputs, labels_dev); _, preds = torch.max(outputs,1)
                v_loss += loss.item()*inputs.size(0); v_corr += torch.sum(preds==labels_dev.data).item()
                # if os.getenv("STREAMLIT_DEBUG_FAST_EPOCH"): break
        history['valid_loss'].append(v_loss/len(valid_ds)); history['valid_acc'].append(v_corr/len(valid_ds))
        
        status_txt.text(f"E{epoch+1} TL:{history['train_loss'][-1]:.3f} TA:{history['train_acc'][-1]:.3f} | VL:{history['valid_loss'][-1]:.3f} VA:{history['valid_acc'][-1]:.3f}")
        prog_bar.progress((epoch+1)/epochs_cfg)
        if epoch % 5 == 0 or epoch == epochs_cfg-1:
            with plot_ph.container(): plot_metrics(history['train_loss'],history['valid_loss'],history['train_acc'],history['valid_acc'], model_name_cfg, f"R{run_id_cfg}_ep{epoch+1}")
        
        if history['valid_loss'][-1] < best_vloss: best_vloss=history['valid_loss'][-1]; epochs_no_imp=0; best_state=model_train.state_dict().copy()
        else: epochs_no_imp+=1
        if epochs_no_imp >= patience_cfg: logging.info(f"Early stop E{epoch+1}."); st.write(f"Early stop E{epoch+1}."); break
    
    if best_state: model_train.load_state_dict(best_state)
    prog_bar.empty(); status_txt.empty(); plot_ph.empty()

    st.write("### Resultados Finais Treino:"); plot_metrics(history['train_loss'],history['valid_loss'],history['train_acc'],history['valid_acc'], model_name_cfg, str(run_id_cfg))
    metrics_test = compute_metrics_detailed(model_train, test_load, full_ds_pil.classes, model_name_cfg, str(run_id_cfg))
    # error_analysis(...) e perform_clustering(...) podem ser chamados aqui
    
    return model_train, full_ds_pil.classes, metrics_test


def main():
    st.set_page_config(page_title="GeoIA Classifier", page_icon="🔬", layout="wide")
    if os.path.exists("capa.png"): st.image("capa.png", use_container_width=True)
    if os.path.exists("logo.png"): st.sidebar.image("logo.png", width=120)
    st.title("Classificador de Imagens GeoMaker-IA")

    if 'all_metrics' not in st.session_state: st.session_state['all_metrics'] = []
    if 'trained_models' not in st.session_state: st.session_state['trained_models'] = []

    st.sidebar.header("Configurações")
    num_cls_ui = st.sidebar.number_input("Nº Classes", 2, 100, 2)
    fine_tune_ui = st.sidebar.checkbox("Fine-Tuning Completo", False)
    epochs_ui = st.sidebar.slider("Épocas", 1, 50, 5, 1) # Reduzido para demo
    lr_ui = st.sidebar.select_slider("Learning Rate", [1e-5,1e-4,1e-3], 1e-4)
    bs_ui = st.sidebar.selectbox("Batch Size", [4,8,16], 1) # index 1 para 8
    dropout_ui = st.sidebar.slider("Dropout", 0.0,0.9,0.5,0.1)
    train_split_ui = st.sidebar.slider("Split Treino", 0.1,0.9,0.7,0.05, "%.2f")
    valid_split_ui = st.sidebar.slider("Split Val", 0.05,0.5,0.15,0.05, "%.2f")
    if train_split_ui+valid_split_ui >= 1.0: st.sidebar.error("Treino+Val < 1.0"); st.stop()
    weighted_loss_ui = st.sidebar.checkbox("Loss Ponderada", True)
    l2_ui = st.sidebar.number_input("L2 Decay", 0.0,0.1,0.001,0.0001,"%.4f")
    patience_ui = st.sidebar.number_input("Paciência EarlyStop", 1,20,3,1) # Reduzido
    
    st.sidebar.markdown("---"); # ... (infos do dev) ...
    st.sidebar.caption(f"Dispositivo: {str(device).upper()}")

    st.header("Treinamento")
    with st.form("train_form"):
        zip_up = st.file_uploader("Dataset (.zip)", ["zip"])
        models_sel_ui = st.multiselect("Arquiteturas:", ['ResNet18','ResNet50','DenseNet121'], ['ResNet18'])
        runs_ui = st.number_input("Execuções/Modelo:", 1,3,1)
        start_btn = st.form_submit_button("🚀 Iniciar")

    if start_btn:
        if not zip_up: st.error("Upload do dataset."); st.stop()
        if not models_sel_ui: st.error("Selecione modelo(s)."); st.stop()
        st.session_state['all_metrics']=[]; st.session_state['trained_models']=[]
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_p = os.path.join(tmpdir, zip_up.name); open(zip_p,"wb").write(zip_up.getbuffer())
            data_p = os.path.join(tmpdir, "data_extracted")
            try: 
                with zipfile.ZipFile(zip_p,'r') as z: z.extractall(data_p)
                items = os.listdir(data_p) # Lidar com pasta raiz extra no ZIP
                if len(items)==1 and os.path.isdir(os.path.join(data_p,items[0])): data_p=os.path.join(data_p,items[0])
                logging.info(f"Dataset em: {data_p}")
            except Exception as e: st.error(f"Erro ZIP: {e}"); st.stop()

            for model_n_iter in models_sel_ui:
                for run_i in range(1, runs_ui+1):
                    model_obj, cls_names, metrics_r = train_model(
                        data_p, num_cls_ui, model_n_iter, fine_tune_ui, epochs_ui, lr_ui, bs_ui,
                        train_split_ui, valid_split_ui, weighted_loss_ui, l2_ui, patience_ui, run_i, dropout_ui
                    )
                    if model_obj and cls_names and metrics_r:
                        st.session_state['all_metrics'].append(metrics_r)
                        model_fn = f"{model_n_iter}_R{run_i}_{uuid.uuid4().hex[:4]}.pth"
                        torch.save(model_obj.state_dict(), model_fn)
                        st.success(f"Modelo {model_n_iter} R{run_i} salvo: {model_fn}")
                        st.session_state['trained_models'].append({'name':f"{model_n_iter} R{run_i}",'arch':model_n_iter,'path':model_fn,'classes':cls_names,'num_cls':len(cls_names),'dropout':dropout_ui})
                        # ... (botão download modelo) ...
                    else: st.warning(f"Falha Treino {model_n_iter} R{run_i}")
                    if 'model_obj' in locals(): del model_obj; torch.cuda.empty_cache()
        
        if st.session_state['all_metrics']:
            st.header("📊 Análise Agregada"); df_all_m = pd.DataFrame(st.session_state['all_metrics'])
            st.dataframe(df_all_m.style.format("{:.4f}", subset=pd.IndexSlice[:, ['Accuracy','Precision','Recall','F1_Score','ROC_AUC']]))
            # ... (ANOVA/Tukey) ...

    if st.session_state.get('trained_models'):
        st.header("🧐 Avaliar Imagem")
        opts = {info['name']:info for info in st.session_state['trained_models']}
        sel_name_eval = st.selectbox("Modelo para Avaliação:", list(opts.keys()))
        img_eval_up = st.file_uploader("Upload Imagem Eval:", ["png","jpg","jpeg"])
        if img_eval_up and sel_name_eval:
            info_eval = opts[sel_name_eval]
            try:
                eval_m = get_model(info_eval['arch'], info_eval['num_cls'], info_eval['dropout'], True) # Assumir fine_tune=True para carregar state_dict completo
                eval_m.load_state_dict(torch.load(info_eval['path'], map_location=device)); eval_m.to(device)
                pil_eval = Image.open(img_eval_up).convert("RGB"); st.image(pil_eval, "Carregada", 200)
                pred_n, pred_c = evaluate_image(eval_m, pil_eval, info_eval['classes'], test_transforms) # Requer que evaluate_image seja definida
                st.success(f"**Predição '{info_eval['name']}':** {pred_n} ({pred_c:.2%})")
                # if st.checkbox("Grad-CAM?"): visualize_activations(...) # Requer que visualize_activations seja definida
                del eval_m; torch.cuda.empty_cache()
            except Exception as e: st.error(f"Erro Avaliação: {e}")

if __name__ == "__main__":
    main()
