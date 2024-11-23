import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load and preprocess dataset
data = pd.read_csv('C:\\Users\\bhavi\\OneDrive\\Desktop\\medical_insurance_prediction\\dataset.csv')

data['gender'] = data['gender'].map({'male': 1, 'female': 0})

data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

X = data[['age', 'gender', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
