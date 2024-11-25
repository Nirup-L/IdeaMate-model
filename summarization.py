from transformers import pipeline

# Function to expand the input idea with related keywords (Text-to-Text Generation)
def expand_idea(idea):
    # Load a text generation pipeline (GPT-3-like model or T5 for text generation)
    text_generator = pipeline("text-generation", model="gpt2")  # You can use a more powerful model if needed
    
    # Expand the input idea by adding more related content
    expanded_idea = text_generator(idea, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    # Print and return the expanded idea
    print("Expanded Idea:", expanded_idea)
    return expanded_idea

# Function to summarize the expanded idea
def summarize_idea(idea):
    # Load the pretrained BART model for summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Summarize the expanded idea
    summary = summarizer(idea, max_length=100, min_length=30, do_sample=False)
    
    # Print and return the summary
    print("Summary:", summary[0]['summary_text'])
    return summary[0]['summary_text']

# Main function that integrates the expansion and summarization
def process_idea(idea):
    # Step 1: Expand the idea with related keywords (Text-to-Text generation)
    expanded_idea = expand_idea(idea)
    
    # Step 2: Summarize the expanded idea
    final_summary = summarize_idea(expanded_idea)
    
    # Return the final summary
    return final_summary

# Example usage
