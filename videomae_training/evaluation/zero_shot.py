"""
This project reuses and adapts the `VideoClsDataset` class originally introduced
in the official VideoMAE implementation (see kinetics.py).

Tong, Z., Song, Y., Wang, J., & Wang, L. (2022).
VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.
Advances in Neural Information Processing Systems (NeurIPS 2022).
arXiv preprint arXiv:2203.12602.

"""

from .dataset import TestingVideoClsDataset
import torch
from timm.models import create_model
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from .metrics import hit_at_k
import VideoMAE.modeling_pretrain #Register videomae models into timm
import json
import os
import argparse
from tqdm import tqdm
from config import TEST_FOLDER, FARM_NAMES
from collections import defaultdict
import random

def load_videomae_encoder(checkpoint_path: str, model_name: str, device: str):
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

    return model

def make_no_mask(batch_size, model, device):
    num_patches = model.encoder.patch_embed.num_patches

    return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

from collections import defaultdict
import random
import numpy as np


def split_intra_domain_stratified(
    all_representations,
    ratio,
    domain_names,
    seed=42,
    ):
    """
    Stratified intra-domain split.

    For each domain and label:
        - at least one sample in query
        - at least (1 - ratio) of total label occurrences in index
    """
    index_data = []
    query_data = []
    rng = random.Random(seed)
    
    # Group by domain
    domains = {name: [] for name in domain_names}
    for r in all_representations:
        path = r[2]
        matched = False
        for domain_name in domain_names:
            if domain_name in path:
                domains[domain_name].append(r)
                matched = True
                break
        if not matched:
            raise ValueError(f"No declared domain found in path: {path}")

    # Global label counts
    total_per_label = defaultdict(int)
    for r in all_representations:
        total_per_label[r[1]] += 1

    test_per_label = defaultdict(int)

    # Stratified split
    for samples in domains.values():
        if not samples:
            continue

        by_label = defaultdict(list)
        for r in samples:
            by_label[r[1]].append(r)

        for label, label_samples in by_label.items():
            rng.shuffle(label_samples)

            n_total_label = total_per_label[label]
            n_domain_label = len(label_samples)

            n_test = max(1, int(n_domain_label * ratio))

            max_allowed_test_global = n_total_label - int(np.ceil((1 - ratio) * n_total_label))
            remaining_capacity = n_total_label - test_per_label[label]
            n_test = min(n_test, max(0, remaining_capacity))

            if test_per_label[label] == 0 and n_test == 0:
                n_test = 1

            test_samples = label_samples[:n_test]
            train_samples = label_samples[n_test:]

            query_data.extend(test_samples)
            index_data.extend(train_samples)

            test_per_label[label] += len(test_samples)

    return index_data, query_data
    
def find_neighbours(
    all_representations: List[Tuple[np.ndarray, int, str]], #(embedding, label, path)
    test_folder: str,
    distance_type: str = "euclidean",
    without_domain_shift: bool = False,
    ratio=1/5,
    )-> Dict[int, Dict]:
    """
    Construit le dictionnaire des voisins pour une évaluation retrieval.
    
    Modes :
        - without_domain_shift = False :
            Les échantillons dont le chemin contient `test_folder` sont utilisés comme 
            ensemble de requêtes (query set), les autres comme ensemble d’index.
        - without_domain_shift = True :
            Les données sont d’abord regroupées par domaine (via FARM_NAMES). 
            Pour chaque domaine, une fraction `ratio` des échantillons est 
            utilisée comme requêtes, le reste constituant l’index (split intra-domaine).

    Args:
        all_representations: Liste de tuples (embedding, label, path).
        test_folder: Nom du folder utilisé comme ensemble de test (query).
        distance_type: Métrique utilisée par `scipy.spatial.distance.cdist` (ex: "euclidean", "cosine").
        without_domain_shift: Active le split intra-domaine.
        ratio: Proportion (0 < ratio < 1) d’échantillons par domaine utilisée 
               comme requêtes lorsque without_domain_shift=True.
        
    Returns:
        Dict[int, Dict]:
            Dictionnaire indexé par id de requête (0..N_query-1) :
                {
                    i: {
                        "label": int,
                        "neighbours": List[int]  # labels triés par distance croissante
                    }
                }
            Format compatible avec `hit_at_k`.
    """
    if not without_domain_shift:
        # Query = test_folder, Index = everything else
        index_data = [r for r in all_representations if test_folder not in r[2]]
        query_data = [r for r in all_representations if test_folder in r[2]]
    else:
        # Query et index en fonction du ratio par domaine
        index_data, query_data = split_intra_domain_stratified(all_representations, ratio, FARM_NAMES.keys())
        print(len(query_data))
    if len(query_data) == 0 or len(index_data) == 0:
        raise ValueError("Query or index set is empty.")

    index_embeddings = np.stack([r[0] for r in index_data])
    index_labels = np.array([r[1] for r in index_data])

    query_embeddings = np.stack([r[0] for r in query_data])
    query_labels = np.array([r[1] for r in query_data])

    # distances computed between test and train
    dists = cdist(query_embeddings, index_embeddings, metric=distance_type)

    all_neighbours = {}

    for i in range(len(query_embeddings)):
        order = np.argsort(dists[i])
        all_neighbours[i] = {
            "label": query_labels[i],
            "neighbours": index_labels[order].tolist()
        }

    return all_neighbours

def compute_and_save_embeddings(
    test_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str,
    output_json: str,
    mapping: Dict,
    ) -> List[Tuple[np.ndarray, int]]:
    """
    Extract embeddings and aggregate per video.

    - Intra-clip pooling: mean over tokens.
    - Inter-clip pooling: mean over all segments/crops of the same video.

    Returns:
        List[(video_embedding, label)]
    """

    video_dict = {}  # path -> {"embeddings": [], "label": int}

    with torch.no_grad():
        for videos, labels, video_paths, *_ in tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc="Extracting embeddings"
            ):

            videos = videos.to(device)
            mask = make_no_mask(videos.shape[0], model, device)

            embeddings = model(videos, mask=mask)          # (B, N, D)
            embeddings = embeddings.mean(dim=1)            # (B, D)
            embeddings = embeddings.cpu().numpy()

            for emb, label, path in zip(embeddings, labels, video_paths):
                if path not in video_dict:
                    video_dict[path] = {
                        "embeddings": [],
                        "label": int(label)
                    }
                video_dict[path]["embeddings"].append(emb)

    all_entries = []
    all_representations: List[Tuple[np.ndarray, int]] = []
    inv_mapping = {encoded: decoded for decoded, encoded in mapping.items()}

    for path, data in video_dict.items():
        stacked = np.stack(data["embeddings"], axis=0)
        video_embedding = stacked.mean(axis=0)  # inter-clip average pooling

        all_representations.append((video_embedding, data["label"], path))

        all_entries.append({
            "path": path,
            "encoded_label": data["label"],
            "decoded_label": inv_mapping[data["label"]],
            "embedding": video_embedding.tolist()
        })

    with open(output_json, "w") as f:
        json.dump(all_entries, f)

    print(f"Saved {len(all_entries)} video-level embeddings to {output_json}")

    return all_representations

def load_embeddings(json_path: str) -> List[Tuple[np.ndarray, int]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    all_representations = [
        (np.array(entry["embedding"], dtype=np.float32), int(entry["encoded_label"]), str(entry["path"]))
        for entry in data
    ]

    return all_representations



def main(args):
    """
    Run zero-shot video retrieval evaluation.

    Pipeline:
        1. Optionally compute and store video embeddings (JSON).
        2. Load embeddings if already computed.
        3. Compute nearest neighbours (Euclidean distance).
        4. Evaluate Hit@K.
        5. Save Hit@K results to CSV.

    Outputs (under output_dir/experiment_name):
        - all_embeddings.json
        - hit_at_k.csv
    """

    experiment_name = (args.experiment_name 
                        if args.experiment_name != None 
                        else os.path.basename(os.path.dirname(args.checkpoint_path)))

    exp_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    output_json = os.path.join(exp_dir, "all_embeddings.json")
    suffix = f"_without_domain_shift" if args.without_domain_shift else ""
    output_csv = os.path.join(exp_dir, f"hit_at_k{suffix}.csv")

    if args.compute_embeddings:
        print("Loading model")
        model = load_videomae_encoder(checkpoint_path=args.checkpoint_path,
                                        model_name=args.model_name,
                                        device="cpu")
        print("Model loaded")
        dataset = TestingVideoClsDataset(anno_path=args.anno_path,
                                        clip_len=args.num_frames,
                                        sep=args.sep,
                                        test_num_segment=args.test_num_segment,
                                        test_num_crop=args.test_num_crop,
                                        frame_sample_rate=args.frame_sample_rate,
                                        crop_size=args.crop_size)

        test_dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False)
        print("Now extracting embeddings")
        all_representations = compute_and_save_embeddings(test_dataloader, model, args.device, output_json, dataset.label_mapping)
    else:
        all_representations = load_embeddings(output_json)

    all_data = find_neighbours(all_representations, 
                               test_folder=TEST_FOLDER, 
                               distance_type="euclidean",
                               without_domain_shift=args.without_domain_shift,
                               ratio=args.ratio)

    results = []
    print(f"Computing Hit@k results with {TEST_FOLDER} as test")
    for k in args.k:
        micro_value = hit_at_k(all_data, k, average="micro")
        macro_value = hit_at_k(all_data, k, average="macro")
        print(f"Hit@{k}: micro = {micro_value}, macro = {macro_value}")
        results.append((k, micro_value, macro_value))

    with open(output_csv, "w") as f:
        f.write("k,hit_at_k_micro,hit_at_k_macro\n")
        for k, micro_value, macro_value in results:
            f.write(f"{k},{micro_value},{macro_value}\n")

    print(f"Saved metrics to {output_csv}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--compute_embeddings", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 10], help="List of K values for Hit@K evaluation")
    parser.add_argument("--without_domain_shift", action="store_true", help="Split the data between indexing and test without separating depending on the farm")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of samples per domain used as queries when without_domain_shift=True \
                                                                  (remaining samples form the index set).")
    
    # For emebedding extraction : 
    ## Dataset parameters
    parser.add_argument("--anno_path", type=str, default="./pretraining_dataset/test.csv")
    parser.add_argument("--sep", type=str, default=";", help="Separator for the annotation csv")
    parser.add_argument("--test_num_segment", type=int, default="1", help="Number of temporal segments")
    parser.add_argument("--frame_sample_rate", type=int, default="1", help="Temporal stride between frames")
    parser.add_argument("--test_num_crop", type=int, default="1", help="Number of spatial crops")
    parser.add_argument("--crop_size", type=int, default="224", help="Spatial crop size before final resize")

    ## Model and dataloader parameters
    parser.add_argument("--checkpoint_path", type=str, default="./results/final_pretrain/checkpoint-best.pth")
    parser.add_argument("--output_dir", type=str, default="./zero_shot_results")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="pretrain_videomae_base_patch16_224")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_mem", type=bool, default=torch.cuda.is_available())


    args = parser.parse_args()

    main(args)
