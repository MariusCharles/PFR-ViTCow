# ViTCow Project – Pretraining VideoMAE v1

## 0. Install Required Dependencies

```bash
sudo apt update
sudo apt install ffmpeg
sudo apt install -y libgl1
pip install -r requirements_videomae.txt
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

## 4. Evaluate Pretraining
Evaluation code is in the evaluation folder. 
To execute a zero shot evaluation (KNN-like on the embeddings directly from the pretraining model) : 

```bash
export PYTHONPATH=$(pwd)/VideoMAE:$(pwd)
python -m evaluation.zero_shot [--compute_embeddings]
```
- Use `--compute_embeddings` to extract embeddings.
- Extraction can be time-consuming.
- Embeddings are saved as a `.json` file in `--output_dir`.
- If the `.json` already exists and neither the checkpoint nor dataset changed, recomputation is unnecessary.

### Architecture Consistency (Critical)

The following arguments **must match** the pretrained checkpoint configuration:

- `--model_name`
- `--patch_size`
- `--num_frames`
- `--img_size`

Any mismatch will prevent correct weight loading or produce invalid embeddings.

### Evaluation parameters
- `--k` defines the Hit@K values (default: `1 2 3 5 10`).
- Results are saved as a `.csv` file in `--output_dir`.
