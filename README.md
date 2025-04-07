# 🧠 WiDS Datathon 2025: Sex Patterns in ADHD - Team Brainwave

---

### **👥 Team Members**

| Name            | GitHub Handle     | Contributions                                                                 |
|-----------------|-------------------|-------------------------------------------------------------------------------|
| Livia John      | [@liviajohn](https://github.com/liviajohn)         | Tuned XGBoost model, improved cross-validation strategy, evaluated multi-output performance |
| Annie Zhang     | [@anniezhang27](https://github.com/anniezhang27)   | Built Graph Convolutional Network, implemented analysis of quantitative brain imaging data |
| Akshitha Nagaraj| [@akshithanagaraj](https://github.com/akshithanagaraj) | Built Graph Convolutional Network, linked GCN embeddings to XGBoost model, preprocessed FCM data |
| Harini Anand    | [@merlinMorgan16](https://github.com/merlinMorgan16) | Optimized XGBoost model, tuned model for qualitative features                 |

---

## **🎯 Project Highlights**

- Built a hybrid model combining GCN embeddings with XGBoost to predict both ADHD diagnosis and biological sex using fMRI brain imaging data and metadata  
- Achieved **84% accuracy for ADHD** and **97% accuracy for Sex_F** on validation set  
- Utilized a custom Graph Convolutional Network (GCN) to extract embeddings from functional connectivity matrices (FCMs)  
- Performed hyperparameter tuning via `RandomizedSearchCV` with 5-fold cross-validation  
- ROC-AUC scores reached **0.999 (ADHD)** and **0.998 (Sex_F)**  
- Integrated multi-output classification and evaluated performance using precision, recall, F1-score, and confusion matrices  

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**To run the project locally or in Colab:**

```bash
# Clone the repository
git clone https://github.com/anniezhang27/WiDS-AI-Studio--Team-Brainwave.git
cd WiDS-AI-Studio--Team-Brainwave

```

- Use Python 3.9+ in Jupyter or Google Colab  
- Download the dataset from Kaggle and place it in the `data/` directory  
- Open and run:
  - `notebooks/EDA.ipynb`
  - `notebooks/Modeling.ipynb`

---

## **🏗️ Project Overview**

This project was developed as part of the **Spring 2025 Break Through Tech AI Studio** and submitted to the **WiDS Datathon 2025** on Kaggle.  
We were tasked with building a model that uses **fMRI data** to predict:

1. ADHD diagnosis (binary)
2. Biological sex (binary)

The model supports neuroscience research by identifying **functional connectome patterns**, particularly those affecting underrepresented groups such as females with ADHD.

---

## **📊 Data Exploration**

**Dataset Sources:**
- Functional Connectivity Matrices (FCMs)
- Quantitative & Categorical Metadata
- ADHD & Sex_F Labels

**Exploration Highlights:**
- Visualized distributions for ADHD and Sex_F classes
- Noted **class imbalance** in ADHD cases, especially among females
- Merged data across all training sources
- Detected and imputed missing values
- Standardized numeric features and encoded categorical ones

**Challenges:**
- High dimensionality (19,901 FCM features)
- Multi-output task
- Imbalanced target labels

---

## **🧠 Model Development**

### Graph Convolutional Network (GCN):
- Built with **PyTorch Geometric**
- Extracted graph-based embeddings from fMRI FCMs

### XGBoost Classifier:
- Trained separately for **ADHD_Outcome** and **Sex_F**
- Wrapped in `MultiOutputClassifier`
- Applied `scale_pos_weight` to balance ADHD classes

### Pipeline:
- Preprocessing via `ColumnTransformer`
- Grid tuning with `RandomizedSearchCV` over 5 folds
- Best CV score: **0.614**

---

## **📈 Results & Key Findings**

| Target        | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------|----------|-----------|--------|----------|---------|
| ADHD_Outcome  | 83.5%    | 80.7%     | 100%   | 89.3%    | 0.999   |
| Sex_F         | 96.7%    | 91.2%     | 100%   | 95.4%    | 0.998   |

- **Strong predictive performance**, especially for sex classification  
- Confusion matrices and ROC curves confirm robust discrimination  
- Multi-output classification architecture proved effective on this biomedical task

---

## **🖼️ Impact Narrative**

**What brain activity patterns are associated with ADHD? Are they different between males and females?**  
Using GCNs on FCMs allowed us to learn **graph-based embeddings** that captured nuanced ADHD-related brain connectivity patterns. Our results suggest **detectable differences** across sex-based subgroups, indicating unique neural signatures for ADHD in females.

**How could your work help ADHD research or clinical care?**  
By enabling **early detection** of ADHD via noninvasive imaging data, such models can assist clinicians in identifying cases that may otherwise be overlooked — particularly among **female populations**, where underdiagnosis is common.

---

## **🚀 Next Steps & Future Improvements**

- Expand the GCN with deeper layers or **attention mechanisms**
- Incorporate **temporal dynamics** from time-series fMRI (if available)
- Address overfitting via **dropout**, **data augmentation**, or **regularization**
- Evaluate **model fairness** and test **adversarial debiasing techniques**

---

## **📄 References & Additional Resources**

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Break Through Tech AI Studio curriculum
- Zhao et al. (2022), _A dynamic graph convolutional neural network framework reveals new insights into connectome dysfunctions in ADHD_  
  [DOI: 10.1016/j.neuroimage.2021.118774](https://doi.org/10.1016/j.neuroimage.2021.118774)
- 📹 [Statistical Approaches on Vectorized Connectomes for Brain-Behavior Mapping](https://youtu.be/jbIsfVxuMWM?si=4n6Ghe9Eoh5lO1eL)  
- 📹 [Geometric Approaches for Processing Brain Connectomes](https://youtu.be/vtHBOBOcn6E?si=Q0FuLhRJAHxqRcPx)  
- 📹 [Graph Neural Networks to Process Brain Connectomes](https://youtu.be/OkE3776GfWU?si=u1q_45MKaRzue70d)

---
