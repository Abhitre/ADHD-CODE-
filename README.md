 # 1 ADHD-CODE-(Machine learning Part)
The dataset consists of EEG readings of children diagnosed with ADHD and a control group. The goal is to preprocess the data, train multiple classification models, and evaluate their performance.
-Installation & Dependencies
Ensure you have Python installed along with the following libraries:
pip install pandas numpy seaborn matplotlib scikit-learn

-Data Acquisition & Description
The dataset consists of EEG recordings from ADHD and control subjects.
The ADHD dataset is compiled from multiple CSV files (v1p.csv, v3p.csv, v6p.csv).
The control dataset is compiled from (v41p.csv, v42p.csv, v43p.csv).
Both datasets are combined and saved as combined_ADHD.csv and combined_Control.csv, respectively.

-Data Preprocessing
Combining Data:
ADHD and control datasets are merged separately before being combined into a single dataset (combined_data.csv).
Shuffling:
The data is shuffled to remove biases.
Feature Engineering:
EEG readings are used as features, and a result column is added: 0 for ADHD, 1 for control.
Feature Scaling:
StandardScaler is applied to normalize the dataset for better model performance.
Data Visualization
A correlation heatmap is generated to examine feature relationships.
A bar plot is created to visualize the count of ADHD vs. control subjects.

-Model Development & Evaluation
Machine Learning Models Implemented:
Logistic Regression
Accuracy: 60.25%
K-Nearest Neighbors (KNN)
Accuracy: 92.01%
Gaussian Naïve Bayes (GNB)
Accuracy: 63.39%
Decision Tree Classifier
Accuracy: 84.43%
Random Forest Classifier
Accuracy: 89.70%
Each model's performance is evaluated using a confusion matrix and accuracy score.

-Implementation Steps
Clone the repository and navigate to the project directory.
Install dependencies using the provided command.
Ensure that combined_data.csv is in the correct directory.

-Run the preprocessing steps.
Train the models and evaluate their accuracy.

-Results & Conclusion
KNN achieved the highest accuracy at 92.01%.
Logistic Regression and Naïve Bayes underperformed due to data complexity.
Decision Trees and Random Forests provided strong performance, indicating the dataset’s nonlinear patterns.


# 2 .ADHD-CODE-XExplainable AI (XAI) for ADHD Prediction
To enhance transparency in ADHD prediction using machine learning models, we implemented two explainable AI (XAI) techniques:
SHAP (SHapley Additive Explanations) with XGBoost
LIME (Local Interpretable Model-Agnostic Explanations) with Random Forest
These methods provide insights into how models make predictions, helping both researchers and healthcare professionals understand feature importance.

Model 1: SHAP with XGBoost
Overview
SHAP (SHapley Additive Explanations) is a model interpretation technique based on cooperative game theory. It assigns a contribution value to each feature, explaining how much it influences a model’s output. We applied SHAP to an XGBoost classifier, a powerful ensemble learning method, to identify key features in ADHD detection.

Implementation Steps
Model Training & Evaluation:
We trained an XGBoost classifier on standardized ADHD data.
The model achieved an accuracy of 92.69%, with the confusion matrix:
[[18214   824]
 [ 1485 11080]]

SHAP Explanation:
We used SHAP to compute feature importance and generate interpretability plots.
The SHAP summary plot visualizes the most influential features, showing how they impact model predictions.
Insights from SHAP
Features with high SHAP values are the strongest indicators of ADHD.
SHAP allows both global  and local (explanations.
This helps clinicians and researchers understand which biological or behavioral markers contribute most to ADHD detection.

Model 2: LIME with Random Forest
LIME (Local Interpretable Model-Agnostic Explanations) explains individual predictions by approximating black-box models with simpler, interpretable models. We applied LIME to a Random Forest classifier to generate local explanations for ADHD detection.

Implementation Steps
Model Training & Evaluation:
A Random Forest classifier was trained and evaluated.
The model achieved an accuracy of 92.34%, with the confusion matrix:
[[18405   633]
 [ 1785 10780]]

LIME Explanation:
We converted the dataset back into a DataFrame for interpretability.
LIME generated instance-based explanations, highlighting the top 5 features influencing a specific prediction.
A sample instance was analyzed, and LIME provided a breakdown of feature importance.

Insights from LIME
LIME explains why a specific patient was classified as ADHD or non-ADHD.
Unlike SHAP (which provides a global explanation), LIME focuses on individual cases.
This technique is valuable for personalized medicine, where doctors need case-specific insights.

Conclusion
SHAP provides a global understanding of ADHD detection by highlighting the most important features in the dataset.
LIME provides local explanations, making predictions transparent at the individual level.
The combination of these techniques ensures that AI-driven ADHD prediction models are not only accurate but also explainable and trustworthy.




# 3 .ADHD-CODE-Hybrid Model:1. Neural Network + Random Forest
To improve performance, we implement a hybrid model that combines a Neural Network (MLPClassifier()) with Random Forest using a Stacking Classifier (StackingClassifier()).
The Neural Network (MLP) acts as a feature extractor, capturing complex patterns in the data, while Random Forest serves as the final decision-making layer. This combination leverages the strengths of deep learning and ensemble methods, balancing feature learning with interpretability.

Enhances predictive performance by stacking models with complementary strengths.
Random Forest as the final estimator provides robust classification and feature importance analys
Hybrid Model (NN + RF) Accuracy: 0.9339619656361737
Confusion Matrix: 
[[18122   916]
 [ 1171 11394]]

Hybrid Model: Neural Network + Decision Tree
A second hybrid approach replaces the Random Forest with a Decision Tree (DecisionTreeClassifier()) as the final classifier. The MLP Neural Network still performs feature extraction, but the Decision Tree makes the final classification decision.
This approach improves interpretability compared to Random Forest because Decision Trees are naturally explainable, providing a structured breakdown of how decisions are made.
Hybrid Model (NN + DT) Accuracy: 0.9355124513495554
Confusion Matrix: 
[[18122   916]
 [ 1122 11443]]


# 4. Deep learning Part 
In addition to traditional machine learning models and hybrid approaches, we explore deep learning models, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, for ADHD prediction.Data Preparation: Input data is reshaped for compatibility with deep learning models (X_train_cnn.shape[1], 1).
Model Definition:
CNN extracts spatial features from the input data.
LSTM captures temporal dependencies and sequential relationships.
Training: Each model is trained for 30 epochs with a batch size of 16 using the Adam optimizer and binary_crossentropy loss.
Evaluation: Predictions are generated and assessed using accuracy and confusion matrices.
The CNN model was trained for 30 epochs, leading to an accuracy of 95.79%.
The LSTM model was also trained for 30 epochs, achieving 94.23% accuracy, slightly lower than CNN but still highly effective.



 
 


 
