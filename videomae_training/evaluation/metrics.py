import numpy as np
from typing import List, Dict, Optional, Sequence, Literal
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score
)

def hit_at_k(
    all_data: Dict[int, Dict[str, List[int]]],
    k: int,
    average: Literal["micro", "macro"] = "micro"
) -> float:
    """
    Calcule Hit@k pour un problème de retrieval.

    Hit@k = 1 si au moins un des k plus proches voisins
    partage le label de la requête, sinon 0.

    Args:
         Args:
        all_data: dictionnaire indexé par sample_id.
            all_data[i]["label"] : int
            all_data[i]["neighbours"] : List[int]
                Labels des voisins triés par proximité croissante.
        k: nombre de voisins considérés.
        average: 
            - "micro" : moyenne globale sur toutes les requêtes.
            - "macro" : moyenne des Hit@K calculés par classe.
    Returns:
        float: Hit@K agrégé selon le mode choisi.
    """
    if average not in {"micro", "macro"}:
        raise ValueError("average must be 'micro' or 'macro'")

    # MICRO
    if average == "micro":

        hits = []
        for sample in all_data.values():
            y = sample["label"]
            neighbours = sample["neighbours"][:k]
            hits.append(int(y in neighbours))
    
        return float(np.mean(hits))

    # MACRO
    class_hits = {}
    for sample in all_data.values():
        y = sample["label"]
        neighbours = sample["neighbours"][:k]
        hit = int(y in neighbours)

        if y not in class_hits:
            class_hits[y] = []
        class_hits[y].append(hit)

    per_class_scores = [np.mean(hits) for hits in class_hits.values()]
    return float(np.mean(per_class_scores))

def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_scores: Optional[np.ndarray] = None,
    beta: float = 1.0,
    average_type: str = "micro",
) -> Dict[str, float]:
    """
    Calcule les métriques standards de classification.

    Args :
    - y_true : Labels ground truth, taille (N,).
    - y_pred : Prédictions discrètes du modèle, taille (N,).
    - y_scores requis pour mAP : Scores ou probabilités du modèle, shape (N, C).
            Requis uniquement pour le calcul de la mAP.
    - beta : Paramètre du F-beta score.
            beta > 1 favorise le recall, beta < 1 favorise la precision.

    - average_type : Type d'averaging utilisé pour precision, recall et F-beta.
            Valeurs possibles : "micro", "macro", "weighted".
    Returns:
        Dict contenant accuracy, precision, recall, 
        fbeta, mAP micro (optionnel), mAP macro (optionnel)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average_type),
        "recall": recall_score(y_true, y_pred, average=average_type),
        f"f{beta}": fbeta_score(y_true, y_pred, beta=beta, 
                                average=average_type),
    }

    if y_scores is not None:
        metrics["mAP_macro"] = average_precision_score(
            y_true, y_scores, average="macro"
        )
        metrics["mAP_micro"] = average_precision_score(
            y_true, y_scores, average="micro"
        )

    return metrics
