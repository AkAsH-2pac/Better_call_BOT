import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Sample dataset (replace this with your own data)
data = [
    "What is chemistry?",
    "Tell me about chemical elements.",
    "How does a chemical reaction work?",
    "Explain the periodic table.",
    "What are some common compounds?",
    "What is a molecule?",
    "How do atoms bond to form molecules?",
    "Define chemical reaction.",
    "What is the importance of the periodic table?",
    "List some examples of chemical compounds.",
]

# Preprocess the data
lemmatizer = WordNetLemmatizer()
corpus = []

for sentence in data:
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    corpus.append(" ".join(words))

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Initialize buffer memory as a dictionary
buffer_memory = {}


# Function to add a conversation to buffer memory
def add_to_buffer_memory(user_input, response):
    if user_input in buffer_memory:
        buffer_memory[user_input].append(response)
    else:
        buffer_memory[user_input] = [response]


# Function to get the most similar response from buffer memory
def get_response(user_input):
    user_input = lemmatizer.lemmatize(user_input.lower())
    user_vector = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)
    most_similar_index = np.argmax(cosine_similarities)
    response = data[most_similar_index]
    return response


# Slang responses
slang_responses = [
    "Chatbot: Yo, I gotchu! Here's the deal: ",
    "Chatbot: No worries, fam! Check it out: ",
    "Chatbot: Ayy, here's the scoop, bro: ",
    "Chatbot: Lemme drop some knowledge on ya: ",
    "Chatbot: Sure thing, my G! Here's the 411: ",
]

# Chatbot loop
print("Chatbot: Hey yo, shoot the questions, homie!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Peace out, amigo!")
        break

    # Check buffer memory for a relevant response
    for user_msg, responses in buffer_memory.items():
        if user_msg == user_input:
            print("Chatbot: You asked me this before, here's what I said:")
            for response in responses:
                print(f"Chatbot: {response}")
            break
    else:
        # If not found in buffer memory, generate a new response
        response = get_response(user_input)
        slang_response = random.choice(slang_responses) + response
        print(slang_response)

    # Add the conversation to buffer memory
    add_to_buffer_memory(uer_input, response)

# Import necessary libraries and download NLTK data

# Sample dataset (replace this with your own data)
data = [
    # Your existing questions and responses here
]

# Slang responses
slang_responses = {
    "What is chemistry?": "Yo, chemistry is all about mixin' stuff and seein' what happens.",
    "Tell me about chemical elements.": "Chemical elements are like the building blocks of the universe, bro.",
    "How does a chemical reaction work?": "Chemical reactions? It's like magic, man! Stuff changes into other stuff.",
    "Explain the periodic table.": "The periodic table is like a cheat sheet for all the elements. It's how we keep track of 'em.",
    "What are some common compounds?": "Common compounds? You mean like water and stuff? Yeah, they're everywhere."
}

# Function to get a slang response
def get_slang_response(user_input):
    response = slang_responses.get(user_input, "I ain't got a clue, bro.")
    return response

# Rest of your code remains the same

# Chatbot loop
print("Chatbot: Hey yo, shoot the questions, homie!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Peace out, amigo!")
        break

    # Check buffer memory for a relevant response
    for user_msg, responses in buffer_memory.items():
        if user_msg == user_input:
            print("Chatbot: You asked me this before, here's what I said:")
            for response in responses:
                print(f"Chatbot: {response}")
            break
    else:
        # If not found in buffer memory, generate a new response
        response = get_response(user_input)
        slang_response = random.choice(slang_responses) + response
        print(slang_response)

    # Add the conversation to buffer memory
    add_to_buffer_memory(user_input, response)

