# README – Prétraining VideoMAE v1

## 1. Cloner le repository

```bash
git clone https://github.com/MCG-NJU/VideoMAE.git
```

## 2. Préparer les datasets de prétraining
Le script génère automatiquement :
- `pretrain_dataset/`
  - `train/`
  - `test/`
- Les fichiers `.csv` nécessaires

Le dossier utilisé comme `test` est configurable dans les fichiers de configuration.

### Télécharger les données et créer les csv
Si les données n'ont pas déjà été téléchargées :
```bash
python create_pretrain_dataset.py --download
```
Si les données sont déjà en local :
```bash
python create_pretrain_dataset.py
```

## 3. Lancer le prétraining
Des exemples de scripts bash sont disponibles dans `pretrain_scripts/`.
Exécution typique :
```bash
./pretraining_scripts/pretrain_template.sh
```
