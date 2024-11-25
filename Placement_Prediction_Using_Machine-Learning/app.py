import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="templates")

# Load the models
placement_model = pickle.load(open('model.pkl', 'rb'))
salary_model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def h():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Fetch input data
    cgpa = request.args.get('cgpa', '0')
    projects = request.args.get('projects', '0')
    workshops = request.args.get('workshops', '0')
    mini_projects = request.args.get('mini_projects', '0')
    skills = request.args.get('skills', '')
    communication_skills = request.args.get('communication_skills', '0')
    internship = request.args.get('internship', '0')
    hackathon = request.args.get('hackathon', '0')
    tw_percentage = request.args.get('tw_percentage', '0')
    te_percentage = request.args.get('te_percentage', '0')
    backlogs = request.args.get('backlogs', '0')
    name = request.args.get('name', 'Candidate')

    # Count skills
    skill_count = skills.count(',') + 1 if skills else 0

    # Create input array for placement prediction
    arr = np.array([cgpa, projects, workshops, mini_projects, skill_count, 
                    communication_skills, internship, hackathon, 
                    tw_percentage, te_percentage, backlogs], dtype=float)

    # Predict placement
    placement_output = placement_model.predict([arr])[0]
    placement_status = '1' if placement_output == 'Placed' else '0'

    # Create input array for salary prediction
    arr1 = np.append(arr, [placement_status])
    salary = salary_model.predict([arr1])[0]
    formatted_salary = f"{int(salary):,}"

    # Generate response
    if placement_output == 'Placed':
        out = f'Congratulations {name}!! You have high chances of getting placed!'
        out2 = f'Your expected salary will be INR {formatted_salary} per annum.'
    else:
        out = f'Sorry {name}!! You have low chances of getting placed. All the best!'
        out2 = 'Improve your skills...'
    
    return render_template('output.html', output=out, output2=out2)

if __name__ == "__main__":
    app.run(debug=True)
