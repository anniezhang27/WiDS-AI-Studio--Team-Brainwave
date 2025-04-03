# WiDS-AI-Studio--Team-Brainwave

---

### **👥 Team Members**

| Livia John | @liviajohn | Tuned XGBoost model, improved cross-validation strategy, evaluated multi-output performance |
| Annie Zhang | @anniezhang27 | Built Graph Convolutional Network, implemented analysis of quantitative brain imaging data |
| Akshitha Nagaraj | @akshithanagaraj | Also built Graph Convolutional Network, linked up GCN embeddings to XGBoost model, preprocessed FCM data |
| Harini Anand | Optimized XGBoost model, tuned model for qualitative features

---

## **🎯 Project Highlights**

Built a hybrid model combining GNN embeddings with XGBoost to predict both ADHD diagnosis and sex using brain imaging and metadata


Achieved 84% accuracy for ADHD and 97% accuracy for Sex_F on validation set


Utilized a Graph Convolutional Network (GCN) to extract embeddings from fMRI connectome matrices


Performed hyperparameter tuning via RandomizedSearchCV with 5-fold cross-validation


ROC-AUC scores reached 0.999 for ADHD and 0.998 for Sex_F


Integrated multi-output classification and evaluated performance via precision, recall, F1, and confusion matrices


🔗 WiDS Datathon 2025 | Kaggle Competition Page


---

## **👩🏽‍💻 Setup & Execution**

Clone the repository:
git clone https://github.com/anniezhang27/WiDS-AI-Studio--Team-Brainwave.git
cd WiDS-AI-Studio--Team-Brainwave
Install dependencies:
pip install -r requirements.txt
Set up environment: Use a Jupyter or Colab notebook environment with Python 3.9+
Access the dataset: Download the dataset from the Kaggle competition page and place it in the data/ directory
Run the notebooks: Open and run the notebooks/Modeling.ipynb and notebooks/EDA.ipynb notebooks step by step



---

## **🏗️ Project Overview**

This project was completed as part of the Spring 2025 Break Through Tech AI Studio and submitted to the WiDS Datathon 2025 Kaggle competition.
We were challenged to build a machine learning model using fMRI data to predict two outputs: a person’s biological sex and whether they have ADHD. This work supports ongoing neuroscience research by identifying patterns in brain activity associated with ADHD, especially in female subjects.


---

## **📊 Data Exploration**

Dataset:  fMRI functional connectivity matrices (FCM), demographic/clinical categorical + quantitative features, and target labels
Exploration:
Visualized distribution of ADHD vs non-ADHD and male vs female
Identified skew in ADHD distribution, especially within female subgroups
Detected and handled missing values in select features
Preprocessing:
Merged TRAIN_QUANTITATIVE, TRAIN_CATEGORICAL, and TRAIN_FCM
Engineered GNN embeddings using PyTorch Geometric
Standardized numeric features, encoded categorical variables
Challenges:
High-dimensional FCM data (19,901 features)
Multi-output targets (ADHD, Sex)
Class imbalance

---

## **🧠 Model Development**
GNN: Custom Graph Convolutional Network (GCN) with PyTorch Geometric for embedding fMRI matrices
ML Model: XGBoost (separate estimators for ADHD and Sex)
Pipeline:
Preprocessing with ColumnTransformer
MultiOutputClassifier wrapper with scale_pos_weight for class imbalance
Hyperparameter tuning via RandomizedSearchCV
Cross-validation score: 0.614

---

## **📈 Results & Key Findings**
Target
Accuracy
Precision
Recall
F1 Score
ROC-AUC
ADHD_Outcome
83.5%
80.7%
100%
89.3%
0.999
Sex_F
96.7%
91.2%
100%
95.4%
0.998

Strong classification performance, especially for Sex prediction
Confusion matrices and ROC curves confirm high discriminative power


* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

What brain activity patterns are associated with ADHD; are they different between males and females?
 Using a GNN on functional connectivity matrices, we learned graph embeddings that helped capture ADHD-specific neural patterns. The classifier's performance suggests there are detectable and distinct patterns in how ADHD manifests across sexes.


How could your work help ADHD research or clinical care?
 With early prediction capability, such models could support screening and personalized treatment planning, reducing misdiagnoses and improving care especially for underrepresented groups like females with ADHD.

---

## **🚀 Next Steps & Future Improvements**

Expand GNN architecture with deeper layers and attention mechanisms


Incorporate time-series dynamics from fMRI (if available)


Address overfitting by using data augmentation or dropout regularization


Apply fairness metrics and adversarial debiasing techniques to enhance model equity


---

## **📄 References & Additional Resources**

PyTorch Geometric Documentation


XGBoost Documentation


Break Through Tech AI Studio curriculum


NeuroImage (2022): A dynamic graph convolutional neural network framework reveals new insights into connectome dysfunctions in ADHD by Zhao et al.
 DOI: 10.1016/j.neuroimage.2021.118774


📹 Statistical Approaches on Vectorized Connectomes for Brain-Behavior Mapping


📹 Geometric Approaches for Processing Brain Connectomes


📹 Graph Neural Networks to Process Brain Connectomes




---

