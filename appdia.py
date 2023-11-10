# Import necessary libraries
from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('Diabetes_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('MinMaxScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for receiving input and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    input_data = []

    # Get user input for the specified columns
    for column in ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                   'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                   'Income']:
        input_data.append(float(request.form[column]))

    # Scale the input data using the MinMaxScaler
    input_data = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_data)

    # Make a prediction using the pre-trained model
    prediction = model.predict(scaled_input)

    return f'Predicted Diabetes Probability: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
