from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Initialize transformer model and tokenizer for hardware/software classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create a classifier using transformers pipeline
nlp = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

def classify_hardware_software(idea):
    idea_lower = idea.lower()
    classification = {"type": "unknown", "score": 0.0}
    
    # Step 1: Pre-filter classification based on keywords
    hardware_keywords = ["microcontroller", "robot", "sensor", "physical", "arduino"]
    software_keywords = ["web", "app", "AI", "software", "program"]
    
    # Initial score based on keyword matching
    hardware_score = sum(1 for keyword in hardware_keywords if keyword in idea_lower)
    software_score = sum(1 for keyword in software_keywords if keyword in idea_lower)
    
    # Compare pre-filtering scores
    if hardware_score > software_score:
        classification['type'] = "hardware"
        classification['score'] = hardware_score
    elif software_score > hardware_score:
        classification['type'] = "software"
        classification['score'] = software_score
    
    # Step 2: Use NLP classification pipeline if pre-filtering gives no clear result
    if classification['score'] == 0.0:
        type_classification = nlp(idea_lower, candidate_labels=["hardware", "software"])
        # Assign the classification with the higher score between pre-filtering and NLP
        if type_classification['scores'][0] > type_classification['scores'][1]:
            classification['type'] = type_classification['labels'][0]
            classification['score'] = type_classification['scores'][0]
        else:
            classification['type'] = type_classification['labels'][1]
            classification['score'] = type_classification['scores'][1]
    
    return classification
