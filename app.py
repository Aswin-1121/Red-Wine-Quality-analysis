from flask import Flask, render_template, request
import pickle
import numpy as np

# Creating the Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
model = pickle.load(open('redwine.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Collecting input data from form
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        # Preparing data for prediction
        input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
        input_data_as_array = np.asarray(input_data)
        input_data_reshape = input_data_as_array.reshape(1, -1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data_reshape)

        # Making prediction
        prediction = model.predict(input_data_scaled)[0]  # Binary classification: 1 (good), 0 (bad)

        # Interpret prediction
        if prediction == 1:
            result = "Good"
            customer_decision = "The customer will buy it."
        else:
            result = "Bad"
            customer_decision = "The customer will not buy it."

        return render_template("output.html", result=result, customer_decision=customer_decision)
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return "404-An error occurred, please try again later."

if __name__ == '__main__':
    app.run(debug=True)
