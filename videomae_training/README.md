# ViTCow Project – Pretraining VideoMAE v1

## 0. Install Required Dependencies

```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements.txt
```

## 1. Clone the Repository

```bash
git clone https://github.com/MCG-NJU/VideoMAE.git
```

## 2. Prepare Pretraining Datasets
The script automatically generates:
- `pretrain_dataset/`
  - `train/`
  - `test/`
- Required `.csv` files

The folder used as `test` is configurable in the configuration files.

### Download Data and Create CSV Files
If the data has not already been downloaded:
```bash
python create_pretrain_dataset.py --download
```
If the data is already local:
```bash
python create_pretrain_dataset.py
```

## 3. Launch Pretraining
Example bash scripts are available in `pretrain_scripts/`.
Typical execution:
```bash
./pretraining_scripts/pretrain_template.sh
```
