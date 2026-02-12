##export PYTHONPATH=$(pwd)/VideoMAE:$(pwd)
##python -m evaluation.zero_shot
##Citer Videomae = partie du code = run_class_finetuning.py

from .dataset import create_test_dataset, TestingVideoClsDataset
import torch
from timm.models import create_model
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from .metrics import hit_at_k
import VideoMAE.modeling_pretrain #Register videomae models into timm

anno_path = "/teamspace/studios/this_studio/PFR-ViTCow/pretraining_dataset/test.csv"
checkpoint_path="/teamspace/studios/this_studio/PFR-ViTCow/results/test_1/checkpoint-0.pth"
model_name="pretrain_videomae_base_patch16_224"
patch_size=16
num_frames=16
img_size = 224
device="cpu"
batch_size=2
num_workers=0
pin_mem=torch.cuda.is_available()

def load_videomae_encoder(
    checkpoint_path: str,
    model_name: str,
    device: str,
    ):
    """
    Charge un encodeur VideoMAE pré-entraîné (sans tête).

    Returns:
        torch.nn.Module: modèle prêt pour extraction d'embeddings
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=0, #Pas de tête de classif
         )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, num_frames

def make_no_mask(batch_size, num_patches, device):
    return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

def find_neighbours(
    all_representations: List[Tuple[np.ndarray, int]],
    distance_type: str = "cosine",
    ) -> Dict[int, Dict]:

    embeddings = np.stack([r[0] for r in all_representations])  # (N, D)
    labels = np.array([r[1] for r in all_representations])      # (N,)

    dists = cdist(embeddings, embeddings, metric=distance_type)

    all_neighbours = {}
    for i in range(len(embeddings)):
        order = np.argsort(dists[i]) 
        order = order[order != i]     # retire soi-même
        all_neighbours[i] = {
            "label": labels[i],
            "neighbours": labels[order].tolist()
        }

    return all_neighbours

print("Loading model")
model, num_frames = load_videomae_encoder(checkpoint_path=checkpoint_path,
                                model_name=model_name,
                                device="cpu")
print("Model loaded")
#dataset = create_test_dataset()
dataset = TestingVideoClsDataset(anno_path=anno_path,
                                clip_len=num_frames)
test_dataloader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False)

print("Extracting representations")
all_representations: List[Tuple[np.ndarray, int]] = []
i=0
with torch.no_grad():
    for videos, labels, *_ in test_dataloader:
        videos = videos.to(device)
        mask = make_no_mask(videos.shape[0], patch_size, device)
        embeddings = model(videos, mask=mask)          # (B, D)
        embeddings = embeddings.cpu().numpy()

        for emb, label in zip(embeddings, labels):
            all_representations.append((emb, int(label)))
        print(f"Done: batch {i}/{len(test_dataloader)}")
        i+=1
all_data = find_neighbours(all_representations, distance_type="cosine")

for k in [1, 2, 3, 5, 10]:
    print(f"Hit@{k}: {hit_at_k(all_data, k)}")