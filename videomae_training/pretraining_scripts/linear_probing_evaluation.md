Perfect — since your SSL training is already running and stable, we’ll design something **clean, minimal, and publication-grade**, without breaking your current pipeline.

---

# 🎯 OBJECTIVE

Evaluate representation quality of your SSL ViT using:

* Linear probing
* Clean train/val/test protocol
* Repeatable checkpoints
* Metrics adapted to behavior classification (macro-F1)

---

# 🧭 OVERVIEW OF THE PLAN

You will add **4 components** to your existing codebase:

1. 🔹 Labeled dataset split
2. 🔹 Feature extraction pipeline
3. 🔹 Linear probe trainer
4. 🔹 Periodic evaluation hook during SSL training

---

# 1️⃣ DATASET SPLIT (DO THIS ONCE AND FREEZE IT)

You said you have ~2000 annotated videos.

## Split once, save indices

```python
train: 70%
val:   15%
test:  15%
```

Important:

* Stratified split by behavior class
* Save the split to disk (JSON or pickle)
* NEVER reshuffle later

This ensures:

* Reproducibility
* No leakage
* Publication-grade protocol

---

# 2️⃣ FEATURE EXTRACTION MODULE

Create a standalone script:

```
extract_features.py
```

### Functionality

* Load encoder checkpoint
* Freeze encoder
* Put in eval mode
* Loop through labeled dataset
* Save embeddings + labels

### Example skeleton

```python
encoder.eval()
encoder.to(device)

features = []
labels = []

with torch.no_grad():
    for video, label in dataloader:
        video = video.to(device)
        embedding = encoder(video)  # shape: [B, D]
        features.append(embedding.cpu())
        labels.append(label)

features = torch.cat(features)
labels = torch.cat(labels)

torch.save({
    "features": features,
    "labels": labels
}, "features_epoch_60.pt")
```

Now your probe training becomes extremely fast.

---

# 3️⃣ LINEAR PROBE TRAINER

Create:

```
train_linear_probe.py
```

### Model

```python
classifier = nn.Linear(D, num_classes)
```

No dropout. No hidden layers.

---

### Training config (safe default)

* Optimizer: Adam
* LR: 1e-3
* Epochs: 50
* Batch size: 64
* Weight decay: 0

---

### Training loop

Only train classifier:

```python
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for epoch in range(50):
    classifier.train()
    ...
```

---

# 4️⃣ METRICS (IMPORTANT FOR YOUR DOMAIN)

Since this is animal behavior:

### Track

* Accuracy
* Macro-F1  ← critical
* Per-class F1
* Confusion matrix

Use:

```python
from sklearn.metrics import f1_score
```

Macro-F1:

```python
f1_score(y_true, y_pred, average="macro")
```

---

# 5️⃣ AUTOMATE DURING SSL TRAINING

Modify your SSL training loop.

Every N epochs (e.g., every 20):

```python
if epoch % 20 == 0:
    save_encoder_checkpoint()
    run_feature_extraction()
    train_linear_probe()
    log_metrics()
```

You can even make this asynchronous if needed.

---

# 📈 6️⃣ WHAT YOU SHOULD PLOT

You want 2 curves:

1. SSL pretext loss
2. Linear probe macro-F1

Plot:

```
SSL Epoch  →  Linear Probe F1
```

This figure = paper-quality.

---

# 🔬 7️⃣ TEST SET USAGE (VERY IMPORTANT)

During development:

* Only use validation set

At the very end:

* Run probe once on test set
* Report final performance

Never tune hyperparameters on test.

---

# 🧠 8️⃣ OPTIONAL BUT VERY STRONG (for a PhD)

Repeat probe training 3 times with different seeds.

Report:

```
Mean ± std
```

Reviewers love this.

---

# 🏗 Suggested Project Structure

```
project/
│
├── ssl_training/
│
├── probing/
│   ├── extract_features.py
│   ├── train_linear_probe.py
│   └── splits.json
│
├── checkpoints/
│
└── logs/
```

---

# 🚀 9️⃣ Minimal Viable Version (if you want fast implementation)

If you don’t want feature saving:

You can directly:

1. Freeze encoder
2. Attach linear layer
3. Train on labeled data

But extracting features once is cleaner and faster.

---

# 🧭 10️⃣ How to Interpret Results

If you see:

### Case A

SSL loss ↓
Probe accuracy ↑ steadily

→ Good representation learning

### Case B

SSL loss ↓
Probe accuracy plateaus

→ Representation saturated

### Case C

Probe accuracy drops

→ Over-regularization or too long training

---

# 🎓 What This Gives You for Your PhD

This allows you to claim:

> "Self-supervised pretraining improves downstream behavior classification by X% compared to training from scratch."

That is publishable.

---

# 💬 Final Question For You

Do you want this to evaluate:

* 🐄 Per-video behavior classification?
* 🎥 Per-frame behavior classification?
* Or temporal sequence modeling?

Because the probing protocol differs slightly depending on that.

And that choice actually affects your scientific conclusions.
