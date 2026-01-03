# Dataracing 2023 – Credit Default Prediction

This repository contains my solution for the **Dataracing 2023** competition, which focused on predicting domestic loan defaults using data provided by the **Magyar Nemzeti Bank (MNB)**.

## 📌 Project Overview

The challenge involved analyzing anonymized and distorted Hungarian lending data to predict the probability of a borrower defaulting on a loan within the two-year period following the observation window.

* **Competition Link:** [Dataracing 2023 Overview](https://eval.dataracing.hu/web/challenges/challenge-page/5/overview)
* **Dataset:** [Google Drive Link](https://drive.google.com/drive/folders/1xPFJ_Ln0lo12oJZk7GN03e_vvCl8R6m9)

## 🎯 Objective

The primary goal was to provide a probability-based prediction for loan defaults. Submissions were evaluated using the **Log Loss (sklearn.metrics.log_loss)** metric, which emphasizes the accuracy of the predicted probabilities rather than just the final classification.

## 🛠 Methodology: Logistic Regression with WoE Binning

For this task, I implemented a robust, industry-standard credit scoring approach in **R**:

1. **Data Preprocessing:** Handling missing values and preparing for the model.
2. **Weight of Evidence (WoE) Transformation:** * Applied optimal binning to capture non-linear relationships between predictors and the target variable.


3. **Modeling:**
* Developed a **Logistic Regression** model using the WoE-transformed variables.
* This approach ensures high interpretability while maintaining strong predictive power.


4. **Evaluation:**
* The model was fine-tuned to minimize Log Loss, aligning with the competition's evaluation criteria.



## 💻 Tech Stack

* **Language:** R


## 📂 Repository Content

* `analysis.R`: The full R script containing data loading, WoE binning, model training, and prediction generation.
