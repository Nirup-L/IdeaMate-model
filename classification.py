from hardware_software_classifier import classify_hardware_software
from domain_classifier import classify_domain

def classify_idea(idea):
    # Classify hardware/software
    classification = {"type" : "Both hardware and software" , "domain" : "General"}
    hw_sw_classification = classify_hardware_software(idea)
    
    # Classify domain
    domain_classification = classify_domain(idea)
    classification["type"] = hw_sw_classification['type']
    classification["domain"]=domain_classification
    return classification

# Example usage
idea = "A new IoT solution for smart agriculture using sensors and Raspberry Pi."
classification_result = classify_idea(idea)
print(classification_result)
