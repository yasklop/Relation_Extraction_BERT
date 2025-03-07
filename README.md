# Relation Extraction with SVM and BERT

 This project trains two models, traditional machine learning model SVM (Support Vector Machine) and deep learning model BERT for relation extraction using the `train_wiki.json` dataset. The dataset is preprocessed, tokenized, and used to fine-tune BERT for classification.


  

## Installation & Requirements

  

Ensure you have the following dependencies installed:

  

PyTorch

  

Transformers (Hugging Face)

  

scikit-learn

 SpaCy

json (built-in)

  

## Dataset

  

### 1️. Dataset Overview

Source: https://github.com/thunlp/FewRel

  

This project uses a structured dataset for **Relation Extraction (RE)**, where each instance consists of:

- A **text sequence** containing two entities.

- **Entity positions** within the text.

- The **relation label** between the two entities.

  

---

### 2️. Dataset Files

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

  

Before training the models, the dataset needs to be preprocessed to prepare it for relation extraction. The following steps are performed:

  

### 1. **Loading the Data**

- The dataset (`train_wiki.json`) is loaded and parsed into a structured format.

- The relation descriptions are mapped using `pid2name.json` to provide meaningful labels.

  

### 2. **Tokenization and Entity Marking**

- Sentences are tokenized using the `BertTokenizer`. (In SVM's part, it is not necessary)

- Entities are marked using special tokens:

```

[E1] entity1 [/E1] [E2] entity2 [/E2]

```

- This helps SVM and BERT understand entity boundaries and improves relation extraction performance.
  

### 3. **Label Encoding**

- Unique relation types are extracted and encoded into numerical labels using `LabelEncoder`.
- In SVM part, `LabelEncoder` is replaced with capturing linguistic features to enrich the text. Additionally, `TfidfVectorizer` is used to represent the importance of a word.

### 4. **Data Splitting**

- The preprocessed dataset is split into:

- **Training set (80%)**: Used for model training.

- **Validation set (20%)**: Used for model evaluation.

  

### 5. **DataLoader Preparation**

- The processed dataset is wrapped into a `PyTorch Dataset` class (`REDataset`).

- A `DataLoader` is created to enable efficient batch training and evaluation.

  

After preprocessing, the dataset is ready to be used for training the BERT-based relation extraction model.

  
  

## Training
### - SVM
Traditional machine learning algorithm **SVM** is less complicated than deep learning BERT. The number of hyperparameters are relatively small. 

The main parameters:
· `C`: **float, default=1.0**

· `kernel`: **{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’**

· `gamma`: **{‘scale’, ‘auto’} or float, default=’scale’**


If you would like to find out more, you can check the official document (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

According to hyperparameter tuning, `GridSearchCV` is a great choice to apply to find the **best combination of hyperparameters** for a given machine learning model (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). `GridSearchCV` uses **cross-validation** to select the **best parameters**. As a result, it will take longer to train the model. 

  
### - BERT 
The model is trained using **BERT-based Relation Extraction (BERT-RE)**. The training process involves fine-tuning a pre-trained **BERT-base-uncased** model on the extracted relation-labeled sentences.
https://drive.google.com/file/d/1W5_Gx4qKyOdQazr9kBh7NH_tPnf6YRa9/view?usp=sharing
  

1. **Model Architecture**

- A **BERT-based classifier** is used, which extracts the `[CLS]` token embedding and passes it through:

- A **Dropout layer (0.3 probability)** for regularization

- A **fully connected layer** that maps to the number of relation labels

  

2. **Optimization & Loss Function**

- **Loss Function:** Cross-Entropy Loss

- **Optimizer:** AdamW with a learning rate of **2e-5**

- **Batch Size:** 16

- **Epochs:** 3



## Evaluation

### - SVM

### Test Accuracy: 0.68

#### Categories with Poor Performance

The following labels have both low precision and recall, indicating that the model struggles to correctly classify these relationships:
  

| Label  | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| P551  | 0.40      | 0.33   | 0.36     |
| P800   | 0.40      | 0.42   | 0.41     |
| P527   | 0.29      | 0.31   | 0.30     |


 
These results suggest that the model frequently misclassifies these relationship types.

Overall, SVM predicts stable results. The difference between `Precision` and `Recall` is not very large. Here some good results are demonstrated.

| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| P105    | 0.98      | 0.95   | 0.96     |
| P1435   | 0.98      | 0.99   | 0.99     |
| P931   | 0.90      | 0.88   | 0.89     |
| P1411   | 0.91      | 0.96   | 0.93     |




### - BERT

### Test Accuracy: 0.89

#### Categories with Poor Performance

The following labels have both low precision and recall, indicating that the model struggles to correctly classify these relationships:

  

| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| P127    | 0.65      | 0.56   | 0.60     |
| P58   | 0.90      | 0.60   | 0.72     |
| P551   | 0.76      | 0.64   | 0.70     |


  

These results suggest that the model frequently misclassifies these relationship types.




  

#### High Recall but Low Precision

The following labels have high recall but low precision, meaning the model tends to over-predict these relationships, which may affect overall reliability:

  


| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| P17   | 0.66      | 0.96   | 0.78     |
| P159   | 0.66      | 0.83   | 0.74     |

  

This indicates that the model often falsely predicts these labels, leading to many false positives.

  

#### High Precision but Low Recall

The following labels have high precision but low recall, meaning the model rarely predicts these relationships, possibly due to data imbalance or weak feature representation:

  


| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| P58   | 0.90      | 0.60   | 0.72     |
| P407   | 0.98      | 0.79   | 0.88     |


 

This suggests that these relationships are under-predicted, which might be caused by insufficient training examples or unclear features in the dataset.

  

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

h_pos = (0, 1) # "Barack Obama"

t_pos = (5, 5) # "Honolulu"

  

predicted_relation = infer(model, tokenizer, label_encoder, sentence, h_pos, t_pos, device)

print(f"Predicted Relation: {pid2name[predicted_relation]}")

```

Predicted Relation: ['residence', 'the place where the person is or has been, resident']

(In SVM part, the process of prediction is similar with that of BERT part. You can just call the custom function `infer`).
 

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

## Citation

If you use this project or its datasets, please cite the following sources:

  

### FewRel: Large-Scale Supervised Few-Shot Relation Classification

Han, Xu, et al. (2018)

[*FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation*](https://www.aclweb.org/anthology/D18-1514)

**Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing**

DOI: [10.18653/v1/D18-1514](https://doi.org/10.18653/v1/D18-1514)

  

```bibtex

@inproceedings{han-etal-2018-fewrel,

title = "{F}ew{R}el: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation",

author = "Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong",

booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",

month = oct # "-" # nov,

year = "2018",

address = "Brussels, Belgium",

publisher = "Association for Computational Linguistics",

url = "https://www.aclweb.org/anthology/D18-1514",

doi = "10.18653/v1/D18-1514",

pages = "4803--4809"

}
