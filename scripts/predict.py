# Data Loading

import pandas as pd

# Load the test data
new_data = pd.read_csv('/Users/quantum/Engine/test_FD001.txt', sep=' ', header=None)

# Check for missing values in columns 26 and 27
print("Missing values in columns 26 and 27 before dropping:")
print(new_data[[26, 27]].isnull().sum())


# Data Preprocessing

# Drop columns 26 and 27 as they contain missing values and are not needed
new_data = new_data.drop([26, 27], axis=1)

# Assign column names to the test data, similar to what was done during training
columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
new_data.columns = columns


# Feature Engineering

# Calculate RUL for each engine
max_cycles = new_data.groupby('engine_id')['cycle'].max()
new_data['RUL'] = new_data.apply(lambda row: max_cycles[row['engine_id']] - row['cycle'], axis=1)

# Rolling means and standard deviations for each sensor
for sensor in [f'sensor_{i}' for i in range(1, 22)]:
    new_data[f'{sensor}_mean'] = new_data.groupby('engine_id')[sensor].rolling(window=5).mean().reset_index(0, drop=True)
    new_data[f'{sensor}_std'] = new_data.groupby('engine_id')[sensor].rolling(window=5).std().reset_index(0, drop=True)

# Lag features for each sensor
for sensor in [f'sensor_{i}' for i in range(1, 22)]:
    new_data[f'{sensor}_lag'] = new_data.groupby('engine_id')[sensor].shift(1)

# Drop any NaN values created by rolling or lagging
new_data_cleaned = new_data.dropna()

# Drop unnecessary columns (RUL, engine_id, and cycle) as they are not part of the features used for prediction
new_data_cleaned = new_data_cleaned.drop(['RUL', 'engine_id', 'cycle'], axis=1)

# Print the shape to ensure the test data matches the training data
print(new_data_cleaned.shape)


import joblib
# Load the saved model
model = joblib.load('models/best_model.pkl')

# Make predictions on the cleaned test data
predictions = model.predict(new_data_cleaned)

# Output predictions
print(predictions)


