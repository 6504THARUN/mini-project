from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'male' else 0
    bmi = float(request.form['bmi'])
    
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0
    region = request.form['region']
    region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_mapping[region]
    
    # Make prediction
    features = np.array([[age, gender, bmi, children, smoker, region]])
    prediction = model.predict(features)[0]
    
    return render_template('result.html', prediction=f"${prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
