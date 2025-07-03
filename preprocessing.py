import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/Ashrae_Buildings.csv')

# Check the missing values
missing_values = df.isnull().sum()
print(missing_values)

# Percentage of missing values.
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

data1=df[df['Cooling startegy_building level']=='Air Conditioned']
data2=df[df['Cooling startegy_building level']=='Naturally Ventilated']
data3=df[df['Cooling startegy_building level']=='Mixed Mode']
data4=df[df['Cooling startegy_building level']=='Mechanically Ventilated']

df=pd.concat([data1,data2,data3,data4],axis=0)

# Making the range from [-3,3] to [-2,2]
df['Thermal sensation'] = df['Thermal sensation'].apply(lambda x: -2 if x <= -2 else x)
df['Thermal sensation'] = df['Thermal sensation'].apply(lambda x: 2 if x >= 2 else x)

# Rounding off the values to make it categorical in nature
df['Thermal sensation'] = df['Thermal sensation'].apply(lambda x: np.round(x))

# Handling NaNs before converting to integer
df = df.dropna(subset=['Thermal sensation'])  # Remove rows where 'Thermal sensation' is NaN

# Convert 'Thermal sensation' to integer type
df['Thermal sensation'] = df['Thermal sensation'].astype(int)

# Removing NaN values from 'Sex' column as well
df = df.dropna(subset=['Sex'])

# Checking unique values
print(df['Thermal sensation'].unique())

duplicates_count = df.duplicated().sum()

df = df.drop_duplicates()

# Define numerical as float64 and categorical as int64 or object
numerical_columns = df.select_dtypes(include=['float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['int64', 'object']).columns.tolist()

# Print results
print("Numerical Columns (float64):", numerical_columns)
print("Categorical Columns (int64 & object):", categorical_columns)

from sklearn.impute import SimpleImputer

# Initialize SimpleImputer with strategy='most_frequent'
imputer = SimpleImputer(strategy='most_frequent')

# Apply imputer to categorical columns
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

from sklearn.impute import KNNImputer

# Initialize KNN Imputer
num_imputer = KNNImputer()


# Apply KNN Imputer only on numerical columns
df[numerical_columns] =num_imputer.fit_transform(df[numerical_columns])

# Combine numerical and encoded categorical features
df = pd.concat([df[numerical_columns], df[categorical_columns]], axis=1)

# Initialize StandardScaler
scaler = StandardScaler()

# Apply StandardScaler only on numerical columns (Ensure they are numeric)
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Ensure categorical column names are valid
categorical_columns = [col for col in categorical_columns if col in df.columns]

# Convert categorical columns to string and fill NaN with 'Unknown'
df[categorical_columns] = df[categorical_columns].astype(str).fillna('Unknown')

# Initialize LabelEncoders dictionary
label_encoders = {}

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future decoding if needed

# Print success message
print("Data preprocessing completed successfully!")
