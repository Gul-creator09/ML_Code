'''Task 01  (Basic)'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("------------ 01. DATA PROCESSING FOR ML -----------")

#---Giving the file path ----#
Gul1 = "e:/Desktop/Internship/codveda technology/My work/Data Set For Task/4) house Prediction Data Set.csv"

df_1 = pd.read_csv(Gul1, sep= r'\s+', header = None)

#----Assigning column names (assume it's a Boston House Price dataset structure)----#
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'Price'
]
df_1.columns = column_names

print("Initial inspection of 'e:/Desktop/Internship/codveda technology/My work/Data Set For Task/4) house Prediction Data Set.csv' with corrected loading:")
print(df_1.head())
print(df_1.info()) # --> This line will print 'None' after the info, as df.info() returns None

# --> Separate features (X) and target (y)
X_1 = df_1.drop('Price', axis=1)
y_1 = df_1['Price']

# ==== Handle Missing Data ==== #
# Identify numerical and categorical columns
numerical_cols_1 = X_1.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_1 = X_1.select_dtypes(include=['object', 'bool']).columns

# --> Handle missing data for numerical columns (imputation with mean)
if not numerical_cols_1.empty:
    imputer_numerical_1 = SimpleImputer(strategy='mean')
    X_1[numerical_cols_1] = imputer_numerical_1.fit_transform(X_1[numerical_cols_1])
    print("\nNumerical columns imputed.") # Added back this print

# --> Handle missing data for categorical columns (imputation with most frequent) and encode them
if not categorical_cols_1.empty:
    imputer_categorical_1 = SimpleImputer(strategy='most_frequent')
    X_1[categorical_cols_1] = imputer_categorical_1.fit_transform(X_1[categorical_cols_1])

 # -==== Encode Categorical Variables ==== #
    # --> Encode categorical variables (One-Hot Encoding)
    encoder_1 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features_1 = encoder_1.fit_transform(X_1[categorical_cols_1])
    encoded_df_1 = pd.DataFrame(encoded_features_1, columns=encoder_1.get_feature_names_out(categorical_cols_1), index=X_1.index)

    # --> Drop original categorical columns and concatenate encoded ones
    X_1 = X_1.drop(columns=categorical_cols_1)
    X_1 = pd.concat([X_1, encoded_df_1], axis=1)
    print("Categorical columns imputed and encoded.") # Added back this print
else:
    print("No categorical columns found for encoding.") # Retained this print for clarity

# ==== Normalize or Standardize Numerical Features ==== #
# --> Normalize numerical features (Standard Scaling)
numerical_cols_after_encoding_1 = X_1.select_dtypes(include=['int64', 'float64']).columns
if not numerical_cols_after_encoding_1.empty:
    scaler_1 = StandardScaler()
    X_1[numerical_cols_after_encoding_1] = scaler_1.fit_transform(X_1[numerical_cols_after_encoding_1])
    print("Numerical columns scaled.") # Added back this print


# ==== Split Dataset into Training and Testing Sets ==== #

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

# ==== Final Output and Verification ==== #
print("\n==== Preprocessing complete ==== ")
print("Shape of X_train:", X_train_1.shape)
print("Shape of X_test:", X_test_1.shape)
print("Shape of y_train:", y_train_1.shape)
print("Shape of y_test:", y_test_1.shape)