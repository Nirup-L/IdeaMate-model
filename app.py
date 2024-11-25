from flask import Flask, request, jsonify, render_template
from summarization import process_idea
from classification import classify_idea
from feasibility import check_feasibility
from stemming import lemmatize_paragraph
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add CORS support for Flutter app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print('Request received....')
    data = request.json
    if not data or 'idea' not in data or not data['idea'].strip():
        return jsonify({"error": "Please provide a valid idea description."}), 400

    idea = data['idea']
    print('summarizing the input.....')
    try:
        summary = process_idea(idea)
    except Exception as e:
        summary = idea[0:180:1]
        print(f"Summarization error: {e}")
        
    print('almost end of summarization.....')    
    try:
        idea = lemmatize_paragraph(idea)
    except Exception as e:
        idea = idea
        print("Exception occured in stemming")
        
    print('classifying the input.....')
    try:
        classification = classify_idea(idea)
    except Exception as e:
        classification = {"type": "Unknown", "domain": "Unknown"}
        print(f"Classification error: {e}")
    
    print('feasibility check.')
    try:
        feasibility = check_feasibility(idea)
    except Exception as e:
        feasibility = "Error in feasibility analysis"
        print(f"Feasibility error: {e}")

    print('Sending response....')
    response = {
        "summary": summary,
        "classification": classification,
        "feasibility": feasibility
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
