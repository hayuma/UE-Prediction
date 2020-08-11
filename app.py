import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    input_features_to_array = [np.array(input_features)]
    output_feature = model.predict(input_features_to_array)
    predicted_ue = round(output_feature[0], 2)
    return render_template('index.html', prediction_text='Predicted UE Marks is   {}%'.format(predicted_ue))
if __name__ == "__main__":
    app.run(debug=True)