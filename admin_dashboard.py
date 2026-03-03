from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64

app = Flask(__name__)

# Load model and data
model = load_model('models/chatbot_model.h5')
with open('data/intents.json') as file:
    intents = json.load(file)

# Analytics data storage
analytics = {
    "queries": [],
    "responses": [],
    "accuracy": 0,
    "response_times": []
}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/update_faqs', methods=['POST'])
def update_faqs():
    data = request.json
    # Update intents.json with new FAQs
    with open('data/intents.json', 'w') as file:
        json.dump(data, file)
    return jsonify({"status": "success"})

@app.route('/analytics')
def get_analytics():
    # Calculate accuracy
    if len(analytics["queries"]) > 0:
        correct = sum(1 for i in range(len(analytics["queries"])) 
                    if analytics["queries"][i] == analytics["responses"][i])
        analytics["accuracy"] = correct / len(analytics["queries"])
    
    # Generate confusion matrix
    if len(analytics["queries"]) > 10:
        cm = confusion_matrix(analytics["queries"], analytics["responses"])
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cm_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
    else:
        cm_image = None
    
    return jsonify({
        "accuracy": analytics["accuracy"],
        "response_time": np.mean(analytics["response_times"]),
        "confusion_matrix": cm_image,
        "top_queries": get_top_queries()
    })

def get_top_queries():
    # Count frequency of each query type
    query_count = {}
    for query in analytics["queries"]:
        query_count[query] = query_count.get(query, 0) + 1
    return sorted(query_count.items(), key=lambda x: x[1], reverse=True)[:5]

if __name__ == '__main__':
    app.run(debug=True)