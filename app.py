#Task 2 and Task 5 : Build Web Service to host model with log function



#Creation of flask instance
app = Flask(__name__)


# Set up logging configurations
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Load the NER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dipteshkanojia/roberta-large-finetuned-ner")
model = AutoModelForTokenClassification.from_pretrained("dipteshkanojia/roberta-large-finetuned-ner")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


#Function ensures that all elements in the NER results are JSON serializable
def make_serializable(results):
    # Convert the result to a JSON serializable format
    for result in results:
        for key in result:
            if isinstance(result[key], np.float32):
                result[key] = float(result[key])
    return results


#Function defined to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400

    text = request.json['text']
    results = ner_pipeline(text)
    serializable_results = make_serializable(results)
    
    # Log the input and output
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": text,
        "output": serializable_results
    }
    logging.info(json.dumps(log_entry))
    
    return jsonify(serializable_results)


#Flask app is selected to run on the given port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
