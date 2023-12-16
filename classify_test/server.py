from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('./model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Receive data from the web form
    student_data = request.form.getlist('student_data[]')  # Adjust this to match your form input fields
    # Preprocess the input data
    input_data = np.array(student_data, dtype=np.int32)
    input_data = input_data.astype(int)
    input_data = tf.expand_dims(input_data, axis=0)

    # Make predictions
    recommended_schedule = model.predict(input_data)

    # Get the top 10 recommended courses
    top_ten_indices = np.argsort(recommended_schedule[0])[::-1][:10]

    # Return the recommendations as JSON
    recommendations = [{'course': int(index) + 1, 'probability': round(float(recommended_schedule[0][index]),3)} for index in top_ten_indices]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
