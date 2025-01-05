from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

file_path = '/content/drive/My Drive/dataset.csv'

# Load the dataset
data = pd.read_csv(file_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.preprocessing import LabelEncoder


# Convert to DataFrame (if not already)
data = pd.DataFrame(data)

# Print the data
print("data")
print(data)

label_encoder = LabelEncoder()

# Features and target column
datainX = data.iloc[:, 1:20]  # Features (X)
datainY = data.iloc[:, 20]    # Target (Y)

# Print column names and target data
print(datainX.columns)
print("debug y")
print(datainY)

# Encode the target column (datainY)
datainY = label_encoder.fit_transform(datainY)

# Print the encoded target column
print("Encoded datainY:")
print(datainY)

# Specify the columns to encode in datainX
columns_to_encode = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
columns_to_remove=[0,5,6,7,12,13]

# Encode specified columns in datainX
for col_index in columns_to_encode:
    column_name = datainX.columns[col_index]
    datainX[column_name] = label_encoder.fit_transform(datainX[column_name])

# Print the encoded features and target
print("Encoded datainX:")
print(datainX.shape)
print("Encoded datainY:")
print(datainY.shape)

datainX = datainX.drop(columns=datainX.columns[columns_to_remove])



# Convert datainY to DataFrame for concatenation
datainY = pd.DataFrame(datainY)

# Concatenate datainX and datainY along columns (axis=1)

# Check the data types of columns in datainX
datainX['TotalCharges'].replace(' ', '0.0', inplace=True)

# Convert the column to float
datainX['TotalCharges'] = datainX['TotalCharges'].astype(float)

# Check the result
print(datainX['TotalCharges'])
for_corr = pd.concat([datainX, datainY], axis=1)
corr_matrix=for_corr.corr()
print(corr_matrix)


print(datainX.columns)
cols=datainX.columns

import itertools as iteration
combinations_prepared=list(iteration.combinations(cols,5))
# print(combinations_prepared)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt

dt=dt(max_depth=15)


X_train,X_test,Y_train,Y_test=train_test_split(datainX,datainY, test_size=0.3,shuffle=True)
dt.fit(X_train,Y_train)

test_predictions=[]

for combo in combinations_prepared:
    X_tr=X_train[list(combo)]
    X_te=X_test[list(combo)]
    dt.fit(X_tr,Y_train)

    predict=dt.predict(X_te)
    test_predictions.append(predict)

test_predictions=pd.DataFrame(test_predictions).T

test_predictions=test_predictions.mean(axis=1)
test_predictions = (test_predictions > 0.5).astype(int)

from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, accuracy_score

# Assuming `test_predictions` contains binary predictions (0 or 1)
# and `Y_train` contains the true labels for comparison

print(test_predictions)
# Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, test_predictions)
print(f"Mean Squared Error (MSE): {mse}")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, test_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Classification Report
print("Classification Report:")
print(classification_report(Y_test, test_predictions))

# Accuracy Score
accuracy = accuracy_score(Y_test, test_predictions)
print(f"Accuracy Score: {accuracy}")







