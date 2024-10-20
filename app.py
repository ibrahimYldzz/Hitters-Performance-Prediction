import numpy as np
from flask import Flask, request, render_template
import pickle 

# Path to the saved model file
model_pickle = 'models/regression_model.pkl'
template_path = 'template.html'

# Initializing the Flask application
app = Flask(__name__) 

# Loading the trained model using pickle
model = pickle.load(open(model_pickle, 'rb')) 


# Defining the home route to render the HTML template
@app.route('/')
def home():
    return render_template(template_path) 


# Defining the route to handle form submission and predict the output
@app.route('/predict', methods = ['POST'])
def predict():
    # Extracting features from the form, converting them to integers
    int_features = [int(x) for x in request.form.values()]    
    
    # Converting features to a NumPy array format as expected by the model
    final_features = [np.array(int_features)]
    
    # Making predictions using the loaded model
    prediction = model.predict(final_features)

    # Rounding the prediction to 2 decimal places
    predicted_y = np.round(prediction, 2)

    # Rendering the result on the HTML template
    return render_template(template_path, prediction_text='Predicted Salary: ${}'.format(predicted_y[0]))


# Running the Flask app
if __name__ == "__main__":
    app.run(debug=True)
