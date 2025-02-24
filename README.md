# Relation Extraction with BERT

This project trains a BERT-based model for relation extraction using the `train_wiki.json` dataset. The dataset is preprocessed, tokenized, and used to fine-tune BERT for classification.

## Installation & Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch transformers scikit-learn datasets
```

Alternatively, install all dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset (`train_wiki.json`) contains relation-labeled sentences. Entities are marked in the text as `[E1] entity1 [/E1] [E2] entity2 [/E2]`.

## Usage

### 1. Preprocess Data

```python
from preprocess import preprocess_data
train_data = load_data("train_wiki.json")
samples = preprocess_data(train_data)
```

### 2. Train Model

```python
from model import BERTRE, train
model = BERTRE(num_labels=len(relations)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    loss, acc = train(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
```

### 3. Evaluate Model

```python
from evaluation import evaluate
acc, report = evaluate(model, val_loader, device)
print(f"Test Accuracy: {acc:.4f}")
print(report)
```

## Saving & Loading Model

To save the trained model:

```python
torch.save(model.state_dict(), "bert_re_model.pth")
```

To load and use the model later:

```python
model.load_state_dict(torch.load("bert_re_model.pth"))
model.eval()
```

## Notes
- The dataset is split into **80% training** and **20% validation**.
- The model is fine-tuned using BERT (`bert-base-uncased`).
- Training and evaluation are performed using PyTorch and Hugging Face Transformers.

## License
MIT License
