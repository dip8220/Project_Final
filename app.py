from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained models
models = {
    "Linear Regression": pickle.load(open("linear_regression.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "KNN": pickle.load(open("knn.pkl", "rb")),
    "SVR": pickle.load(open("svr.pkl", "rb")),
    "ANN": load_model("ann.h5"),  # Load the ANN model from .h5 file
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    Cs = float(request.form['Cs'])
    FA = float(request.form['FA'])
    MA = float(request.form['MA'])
    Cl = float(request.form['Cl'])
    Br = float(request.form['Br'])
    I = float(request.form['I'])
    model_name = request.form['model']
    
    # Prepare data for prediction
    inputs = np.array([[Cs, FA, MA, Cl, Br, I]])
    
    # Get the selected model and make prediction
    model = models[model_name]
    prediction = model.predict(inputs)[0]
    
    # Render the results page
    return render_template(
        'results.html', 
        Cs=Cs, FA=FA, MA=MA, Cl=Cl, Br=Br, I=I, 
        model=model_name, prediction=round(prediction, 3)
    )

if __name__ == '__main__':
    app.run(debug=True)
