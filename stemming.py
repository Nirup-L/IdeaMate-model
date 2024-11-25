import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Ensure you have the necessary resources
nltk.download('punkt')
nltk.download('wordnet')

def lemmatize_paragraph(paragraph):
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the paragraph into words
    words = word_tokenize(paragraph)
    
    # Lemmatize each word in the paragraph
    lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in words]  # 'n' is for noun
    
    # Join the lemmatized words back into a paragraph
    lemmatized_paragraph = ' '.join(lemmatized_words)
    print(lemmatized_paragraph)
    return lemmatized_paragraph

# Example usage

