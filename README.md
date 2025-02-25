# Relation Extraction with BERT

This project trains a BERT-based model for relation extraction using the `train_wiki.json` dataset. The dataset is preprocessed, tokenized, and used to fine-tune BERT for classification.

## Installation & Requirements

Ensure you have the following dependencies installed:

Python 3.8+

PyTorch

Transformers (Hugging Face)

scikit-learn

json (built-in)

## Dataset

### 1️. Dataset Overview
This project uses a structured dataset for **Relation Extraction (RE)**, where each instance consists of:
- A **text sequence** containing two entities.
- **Entity positions** within the text.
- The **relation label** between the two entities.

The dataset follows a format similar to **FewRel** and is stored as a JSON file.

---
###  2️.  Dataset Files
#### 🔹 `train_wiki.json`
- Used for training the relation extraction model.
- Consists of multiple relation types with corresponding sentence instances.

#### 🔹 `pid2name.json`
- Used for validation/testing.
- Provides a mapping of relation identifiers (e.g., P2384) to their descriptions

### 3️. Dataset Structure
The dataset file is a JSON object where:
- **Keys** represent relation labels.
- **Values** are lists of instances corresponding to that relation.

#### 🔹 Example Format:
```json
{
  "capital_of": [
    {
      "tokens": ["The", "capital", "of", "France", "is", "Paris", "."],
      "h": ["France", "Q142", [[3, 3]]], 
      "t": ["Paris", "Q90", [[5, 5]]]
    }
  ]
}
```
## Preprocessing Steps

Before training the model, the dataset needs to be preprocessed to prepare it for BERT-based relation extraction. The following steps are performed:

### 1. **Loading the Data**
   - The dataset (`train_wiki.json`) is loaded and parsed into a structured format.
   - The relation descriptions are mapped using `pid2name.json` to provide meaningful labels.

### 2. **Tokenization and Entity Marking**
   - Sentences are tokenized using the `BertTokenizer`.
   - Entities are marked using special tokens:
     ```
     [E1] entity1 [/E1] [E2] entity2 [/E2]
     ```
   - This helps BERT understand entity boundaries and improves relation extraction performance.

### 3. **Label Encoding**
   - Unique relation types are extracted and encoded into numerical labels using `LabelEncoder`.

### 4. **Data Splitting**
   - The preprocessed dataset is split into:
     - **Training set (80%)**: Used for model training.
     - **Validation set (20%)**: Used for model evaluation.

### 5. **DataLoader Preparation**
   - The processed dataset is wrapped into a `PyTorch Dataset` class (`REDataset`).
   - A `DataLoader` is created to enable efficient batch training and evaluation.

After preprocessing, the dataset is ready to be used for training the BERT-based relation extraction model.


## Training  

The model is trained using **BERT-based Relation Extraction (BERT-RE)**. The training process involves fine-tuning a pre-trained **BERT-base-uncased** model on the extracted relation-labeled sentences.  

1. **Model Architecture**  
   - A **BERT-based classifier** is used, which extracts the `[CLS]` token embedding and passes it through:  
     - A **Dropout layer (0.3 probability)** for regularization  
     - A **fully connected layer** that maps to the number of relation labels  

2. **Optimization & Loss Function**  
   - **Loss Function:** Cross-Entropy Loss  
   - **Optimizer:** AdamW with a learning rate of **2e-5**  
   - **Batch Size:** 16  
   - **Epochs:** 3
   - 
## Running the Inference Method
To run the relation extraction method in real-time, use the `infer` function, which takes the following inputs:

### Input Format
- **Sentence (str):** A natural language sentence containing two entities.
- **h_pos (tuple):** A tuple `(start, end)` representing the position of the head entity in the tokenized sentence.
- **t_pos (tuple):** A tuple `(start, end)` representing the position of the tail entity in the tokenized sentence.
- **model:** A trained BERT-based relation extraction model.
- **tokenizer:** A BERT tokenizer for encoding sentences.
- **label_encoder:** A label encoder used to decode predicted relation indices.
- **device:** The hardware device (CPU/GPU) to run the model.

### Example Usage
```python
sentence = "Barack Obama was born in Honolulu, Hawaii."
h_pos = (0, 1)  # "Barack Obama"
t_pos = (5, 5)  # "Honolulu"

predicted_relation = infer(model, tokenizer, label_encoder, sentence, h_pos, t_pos, device)
print(f"Predicted Relation: {pid2name[predicted_relation]}")
```
Predicted Relation: ['residence', 'the place where the person is or has been, resident']

## Saving & Loading Model

To save the trained model:

```python
torch.save(model, PATH)
```

To load and use the model later:

```python
model=torch.load(PATH)
model.eval()
```
