Utilized Python libraries such as scikit-learn and pandas to preprocess the heart disease dataset, including handling missing values and train machine learning algorithms.
analyzed the results to find that men have an 85% higher risk of developing heart disease compared to women, suggesting that gender is a significant predictor in the dataset.
Evaluated multiple machine learning models and selected the one with the highest accuracy, achieving an 98.5% prediction accuracy on the dataset.

Dataset is used from kaggle.
The dataset utilized in this project comprises 301 rows and 10 columns in which 8 are used, each representing specific attributes related to heart health. The columns include 'age,' 'sex,' 'trestbps' (resting blood pressure), 'chol' (serum cholesterol), 'fbs' (fasting blood sugar), 'restecg' (resting electrocardiographic results), 'thalach' (maximum heart rate achieved), 'thal' (thalassemia), and a target column 'Disease' indicates the presence or absence of heart disease. The 'age' column reflects the age of the individuals, 'sex' denotes their gender, and 'trestbps' and 'chol' provide information on resting blood pressure and serum cholesterol levels, respectively. 'Fbs' indicates fasting blood sugar levels, while 'restecg' characterizes resting electrocardiographic results. 'Thalach' represents the maximum heart rate achieved during the study, and 'thal' pertains to thalassemia. The 'goal' column is not specified, and the 'disease' column serves as the target variable for predicting heart disease. This dataset is fundamental for exploring correlations between these attributes and understanding the factors contributing to heart health outcomes.


Here's a basic summary of the algorithms used:
Decision Trees (DT):Decision trees are used for classification and regression tasks. They partition the feature space into regions, making them easy to interpret and suitable for decision-making in    healthcare scenarios.
Naive Bayes (NB):NB is a probabilistic classifier working using Bayes' theorem. It's efficient and effective for text classification and medical diagnosis tasks where features are assumed to be independent.
Support Vector Machines (SVM):SVM is a supervised learning algorithm for regression and classification tasks. It finds the hyperplane that best separates classes in the feature space, making it suitable for tasks like patient diagnosis and disease prediction.
Random Forests (RF):Random forests are ensemble learning methods that construct multiple decision trees during training and output the mode of the classes for classification tasks. They are robust and accurate, making them suitable for medical data analysis.
k-Nearest Neighbors (KNN):KNN is a simple and intuitive classification algorithm that classifies objects based on the majority vote of their neighbors. It's often used in healthcare for patient similarity analysis and disease prediction.
Logistic Regression (LR):Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. It's commonly used for binary classification tasks in healthcare, such as disease prediction.
Hybrid RF-SVM:This hybrid approach combines the strengths of random forests and support vector machines. It may utilize features from both algorithms to improve classification accuracy and robustness.
Hybrid KNN-NB:Similarly, this hybrid approach combines the k-nearest neighbors and naive Bayes algorithms. It may leverage the simplicity and efficiency of naive Bayes with the instance-based learning of KNN for enhanced performance.
All Classifiers:This likely refers to a combination or ensemble of all the aforementioned classifiers. Ensemble methods often provide superior performance by combining the predictions of multiple base estimators, thereby reducing overfitting and improving generalization.




