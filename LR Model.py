'''TASK 02 (Basic)'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==== Program Initialization and Data Loading ==== #
print("------------ 02. Build a Simple LINEAR REGRESSION Model -----------")

Gul2 = "e:/Desktop/Internship/codveda technology/My work/Data Set For Task/4) house Prediction Data Set.csv" 
df_2 = pd.read_csv(Gul2, sep=r'\s+', header=None) 

column_names_2 = [ 
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'Price'
]
df_2.columns = column_names_2 

# --> Separate features (X) and target (y) variable
X_2 = df_2.drop('Price', axis=1) 
y_2 = df_2['Price'] 

# ==== Data Preprocessing (Replicated for Standalone Execution) ==== #
# --> Identify numerical and categorical columns
numerical_cols_2 = X_2.select_dtypes(include=['int64', 'float64']).columns 
categorical_cols_2 = X_2.select_dtypes(include=['object', 'bool']).columns 

# --> Handle missing data for numerical columns using mean imputation
if not numerical_cols_2.empty:
    imputer_numerical_2 = SimpleImputer(strategy='mean') 
    X_2[numerical_cols_2] = imputer_numerical_2.fit_transform(X_2[numerical_cols_2]) 

# --> Handle missing data for categorical columns using most frequent imputation
# and then encode them
if not categorical_cols_2.empty:
    imputer_categorical_2 = SimpleImputer(strategy='most_frequent')
    X_2[categorical_cols_2] = imputer_categorical_2.fit_transform(X_2[categorical_cols_2]) 
    
    # --> Encode categorical variables using One-Hot Encoding
    encoder_2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
    encoded_features_2 = encoder_2.fit_transform(X_2[categorical_cols_2]) 
    
    # --> Create a DataFrame from the encoded features
    encoded_df_2 = pd.DataFrame(encoded_features_2, columns=encoder_2.get_feature_names_out(categorical_cols_2), index=X_2.index)
    
    # --> Drop original categorical columns and concatenate encoded ones
    X_2 = X_2.drop(columns=categorical_cols_2) 
    X_2 = pd.concat([X_2, encoded_df_2], axis=1) 

# --> Normalize numerical features using StandardScaler
numerical_cols_after_encoding_2 = X_2.select_dtypes(include=['int64', 'float64']).columns 
if not numerical_cols_after_encoding_2.empty:
    scaler_2 = StandardScaler() 
    X_2[numerical_cols_after_encoding_2] = scaler_2.fit_transform(X_2[numerical_cols_after_encoding_2])

# --> Split the dataset into training and testing sets
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)


# ==== Linear Regression Model Training and Interpretation ==== #
# --> Initialize and train the Linear Regression model
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2)

print("\nLinear Regression Model trained successfully.")

# --> Interpret model coefficients
print("\nModel Coefficients (Feature importance based on magnitude):")
feature_names_2 = X_train_2.columns 
coefficients_df = pd.DataFrame({'Feature': feature_names_2, 'Coefficient': model_2.coef_})
print(coefficients_df.sort_values(by='Coefficient', ascending=False))

# ==== Model Prediction and Evaluation ==== #
# --> Make predictions on the test set
y_pred_2 = model_2.predict(X_test_2)

mse_2 = mean_squared_error(y_test_2, y_pred_2)
r2_2 = r2_score(y_test_2, y_pred_2)

print("\n==== Model Evaluation ==== ")
print("Mean Squared Error (MSE): {:.2f}".format(mse_2))
print("R-squared (R2): {:.2f}".format(r2_2))