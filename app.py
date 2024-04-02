from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained Random Forest model
rf_classifier = joblib.load('rf_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the HTML form
    cgpa = float(request.form['cgpa'])
    internships = int(request.form['internships'])
    projects = int(request.form['projects'])
    workshops_certifications = int(request.form['workshops_certifications'])
    aptitude_test_score = int(request.form['aptitude_test_score'])
    soft_skills_rating = float(request.form['soft_skills_rating'])
    extracurricular_activities = request.form['extracurricular_activities'].lower()
    placement_training = request.form['placement_training'].lower()
    ssc_marks = int(request.form['ssc_marks'])
    hsc_marks = int(request.form['hsc_marks'])

    # Create a DataFrame with user input
    sample_input = pd.DataFrame({
        'CGPA': [cgpa],
        'Internships': [internships],
        'Projects': [projects],
        'Workshops/Certifications': [workshops_certifications],
        'AptitudeTestScore': [aptitude_test_score],
        'SoftSkillsRating': [soft_skills_rating],
        'ExtracurricularActivities': [1 if extracurricular_activities == 'yes' else 0],
        'PlacementTraining': [1 if placement_training == 'yes' else 0],
        'SSC_Marks': [ssc_marks],
        'HSC_Marks': [hsc_marks]
    })

    # Make predictions on the sample input data
    prediction = rf_classifier.predict(sample_input)

     # Define messages based on the prediction result
    if prediction[0] == 1:
        message = "You are good to go! All the best for your placements!"
    else:
        message = "You need to work harder to improve your chances. Best of luck!"

    # Render the prediction result along with the message
    return render_template('result.html', prediction=prediction[0], message=message)

if __name__ == '__main__':
    app.run(debug=True)
