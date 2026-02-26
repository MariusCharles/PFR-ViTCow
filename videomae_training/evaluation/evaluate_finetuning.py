import torch
import numpy as np
from tqdm import tqdm
from timm.models import create_model
import VideoMAE.modeling_finetune  # enregistre les modèles dans timm
from .metrics import classification_metrics, compute_and_save_confusion_matrix
from .dataset import TestingVideoClsDataset
import pandas as pd
import argparse
import os

def load_finetuned_model(checkpoint_path, model_name, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):
    all_preds = []
    all_targets = []
    all_scores = []
    all_paths = []

    for videos, targets, video_paths, *_ in tqdm(dataloader):

        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(videos)                  # (B, num_classes)
        probs = torch.softmax(logits, dim=1)    # scores

        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_scores.append(probs.cpu().numpy())
        all_paths.extend(video_paths)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_scores)

    return all_paths, y_pred, y_true, y_scores

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model")
    model = load_finetuned_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        device=device
    )
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


    all_paths, y_pred, y_true, y_scores = evaluate(model, test_dataloader, device)
    
    df_dict = {
        "video_path": all_paths,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    for c in range(args.num_classes):
        df_dict[f"score_class_{c}"] = y_scores[:, c]

    df = pd.DataFrame(df_dict)

    output_csv = os.path.join(args.output_dir, "predictions_full.csv")
    df.to_csv(output_csv, index=False)

    print(f"Saved full predictions to {output_csv}")

    # Calcul
    metrics_macro = classification_metrics(
        y_pred=y_pred,
        y_true=y_true,
        y_scores=y_scores,
        average_type="macro"
    )

    metrics_micro = classification_metrics(
        y_pred=y_pred,
        y_true=y_true,
        y_scores=y_scores,
        average_type="micro"
    )

    # Ajout d'un identifiant de type d'average
    metrics_macro["average"] = "macro"
    metrics_micro["average"] = "micro"

    # Fusion en DataFrame
    df = pd.DataFrame([metrics_macro, metrics_micro])
    print(df)

    # Réorganisation colonnes (average en premier)
    cols = ["average"] + [c for c in df.columns if c != "average"]
    df = df[cols]

    # Sauvegarde
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"{args.output_dir}/classification_results.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")

    with open(args.class_mapping, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    cm_raw = compute_and_save_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
    suffix="raw",
    output_dir=args.output_dir,
    normalize=None
    )

    cm_norm = compute_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        suffix="norm",
        output_dir=args.output_dir,
        normalize="true"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--class_mapping", type=str, default="./training_dataset/class_mapping.txt")

    parser.add_argument("--checkpoint_path", type=str, default="./results/train_35_epochs/checkpoint-best.pth")
    parser.add_argument("--output_dir", type=str, default="./testing_results/train_35_epochs")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--anno_path", type=str, default="./training_dataset/test.csv")
    parser.add_argument("--sep", type=str, default=" ", help="Separator for the annotation csv")
    parser.add_argument("--test_num_segment", type=int, default="1", help="Number of temporal segments")
    parser.add_argument("--frame_sample_rate", type=int, default="1", help="Temporal stride between frames")
    parser.add_argument("--test_num_crop", type=int, default="1", help="Number of spatial crops")
    parser.add_argument("--crop_size", type=int, default="224", help="Spatial crop size before final resize")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_mem", type=bool, default=torch.cuda.is_available())

    args = parser.parse_args()

    main(args)