'''TASK 03 (Basic)'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==== Program Initialization and Data Loading ==== #
print("------------ 03. Implement K-Nearest Neighbors (KNN) Classifier -----------")

Gul3 = "e:/Desktop/Internship/codveda technology/My work/Data Set For Task/1) iris.csv" # Adjusted to Gul3
df_3 = pd.read_csv(Gul3) 

#---- Initial inspection of the loaded DataFrame ----#
print("\nInitial inspection of '" + Gul3 + "':")
print(df_3.head())
print(df_3.info())

#---- Separate features (X) and target (y) variable ----#
X_3 = df_3.drop('species', axis=1)
y_3 = df_3['species']

# ==== Data Splitting and Feature Scaling ==== #

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.3, random_state=42, stratify=y_3) 

scaler_3 = StandardScaler() 
X_train_scaled_3 = scaler_3.fit_transform(X_train_3) 
X_test_scaled_3 = scaler_3.transform(X_test_3) 

# ==== KNN Model Training and Evaluation for Different K Values ==== #
# Define different K values to test for the KNN classifier
k_values = [3, 5, 7]

results_3 = {}

for k in k_values:
    print("\n--- Training KNN with k = " + str(k) + " ---")
    # --> Initialize and train the KNN model with the current K value
    knn_model_3 = KNeighborsClassifier(n_neighbors=k)
    knn_model_3.fit(X_train_scaled_3, y_train_3)

    # --> Make predictions on the scaled test set
    y_pred_3 = knn_model_3.predict(X_test_scaled_3)

    # Evaluate the model using various metrics
    accuracy = accuracy_score(y_test_3, y_pred_3)
    conf_matrix = confusion_matrix(y_test_3, y_pred_3)

    # --->'average=weighted' is used for multiclass classification to acc for class imbalance
    precision = precision_score(y_test_3, y_pred_3, average='weighted') 
    recall = recall_score(y_test_3, y_pred_3, average='weighted') 
    f1 = f1_score(y_test_3, y_pred_3, average='weighted') 

    # --> Store results for the current K value
    results_3[k] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# --> Print evaluation metrics for the current K value
    print("Accuracy: {:.2f}".format(accuracy))
    print("Confusion Matrix:\n", conf_matrix)
    print("Precision (weighted): {:.2f}".format(precision))
    print("Recall (weighted): {:.2f}".format(recall))
    print("F1-Score (weighted): {:.2f}".format(f1))

# ==== Comparison and Best K Value Determination ==== #
print("\n==== Comparison of KNN Models with Different K Values ==== ")
for k, metrics in results_3.items():
    print("\nK = " + str(k) + ":")
    print("  Accuracy: {:.2f}".format(metrics['accuracy']))
    print("  Precision: {:.2f}".format(metrics['precision']))
    print("  Recall: {:.2f}".format(metrics['recall']))
    print("  F1-Score: {:.2f}".format(metrics['f1_score']))

# ---- Determine the best K value based on accuracy ---- #
best_k = max(results_3, key=lambda k_val: results_3[k_val]['accuracy']) 
print("\nBest K value based on accuracy: " + str(best_k) + " (Accuracy: {:.2f})".format(results_3[best_k]['accuracy']))
