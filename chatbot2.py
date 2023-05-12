# import string
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize

# stop_words = set(stopwords.words('english'))

# def preprocess(text):
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize into sentences
#     sentences = sent_tokenize(text)
#     # Tokenize each sentence into words
#     words = [word_tokenize(sentence) for sentence in sentences]
#     # Remove stop words
#     words = [[word for word in sentence if word not in stop_words] for sentence in words]
#     # Flatten list of words
#     words = [word for sentence in words for word in sentence]
#     # Join words back into a string
#     text = ' '.join(words)
#     return text

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# def get_most_relevant_sentence(query, sentences):
#     # Compute TF-IDF vectors for the sentences
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(sentences)
#     # Compute TF-IDF vector for the query
#     query_tfidf = vectorizer.transform([query])
#     # Compute cosine similarities between the query vector and the sentence vectors
#     similarities = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()
#     # Get the most similar sentence
#     most_similar_index = np.argmax(similarities)
#     most_similar_sentence = sentences[most_similar_index]
#     return most_similar_sentence

# import streamlit as st

# # Import the functions from the chatbot.py file
# # from chatbot import preprocess, get_most_relevant_sentence, chatbot

# # Set up the app
# def main():
#     st.title("Romeo and Juliet Chatbot")
#     st.write("Ask a question about Romeo and Juliet:")

#     # Get user input
#     user_input = st.text_input("Input your question here:")

#     if user_input:
#         # Preprocess the user input
#         preprocessed_input = preprocess(user_input)

#         # Get the most relevant sentence from the text file
#         relevant_sentence = get_most_relevant_sentence(preprocessed_input)

#         # Get the chatbot's response
#         response = relevant_sentence

#         # Display the chatbot's response
#         st.write(response)

# # Run the app
# if __name__ == "__main__":
#     main()


# Read in dataset
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

with open("human_text.txt", 'r', encoding='utf8', errors='ignore') as file:
    dataset = file.read()

import pandas as pd

# read the file with two delimiters separated by tabs
df = pd.read_csv(r'human_text.txt', sep='\t', header=None, names=['question', 'answer', 'extra'])

sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess(sentence):
    return [lemmatizer.lemmatize(sentence.lower()) for sentence in sentence if sentence.isalnum()]


corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Define chatbot function
def chatbot_response(user_input):
    # Preprocess user input
    user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

    # Vectorize user input
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(user_vector, X)

    # Get index of most similar sentence
    idx = similarities.argmax()

    # Return corresponding sentence from corpus
    return sent_tokens[idx]

import streamlit as st


st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot_response(user_input)
        st.write("Chatbot: " + response)

        # Create a history for the chat
        history.append(('User: ', user_input))
        history.append(('Bot: ', chatbot_response(user_input)))
else:
    st.write('Bye')

st.markdown('<hr><hr>', unsafe_allow_html= True)
st.subheader('Chat History')

chat_history_str = '\n'.join([f'{sender}: {message}' for sender, message in history])

st.text_area('Conversation', value=chat_history_str, height=300)



# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# import time
# import datetime
# import pandas as pd

# nltk.download('punkt')
# nltk.download('wordnet')

# # Load the text file and preprocess the data
# with open('human_text.txt', 'r', encoding='utf-8') as f:
#     dataset = f.read()

# sent_tokens = nltk.sent_tokenize(dataset)
# word_tokens = nltk.word_tokenize(dataset)
# lemmatizer = nltk.stem.WordNetLemmatizer()

# def preprocess(tokens):
#     return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

# corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)

# def chatbot_response(user_input):
#     # Preprocess user input
#     user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

#     # Vectorize user input
#     user_vector = vectorizer.transform([user_input])

#     # Calculate cosine similarity between user input and corpus
#     similarities = cosine_similarity(user_vector, X)

#     # Get index of most similar sentence
#     idx = similarities.argmax()

#     # Return corresponding sentence from corpus
#     return sent_tokens[idx]

# # Create a Streamlit app with the updated chatbot function
# def main():
#     st.title("Butt, the Bot")
#     st.write("Hello there! My name is Kurt, and I'm a simple Bot!\n\n You can ask me some simple questions, nothing more! ")
#     # Get the user's question
#     question = st.text_input("You:")
#     with open('chat_history.txt', 'a') as f:
#         message = question
#         timestamp = datetime.datetime.now()
#         f.write(f"{timestamp} User: {message}\n")
#     # Create a button to submit the question
#     if st.button("Submit"):
#         with st.spinner('Generating response...'):
#             time.sleep(2)
#         response = chatbot_response(question)
#         with open('chat_history.txt', 'a') as f:
#             message = response
#             timestamp = datetime.datetime.now()
#             f.write(f"{timestamp} Butt: {message}\n")
#         st.write("Chatbot: " + response)

#     with open('chat_history.txt') as file:
#         lines = file.readlines()

#     chat_history = []
#     for line in lines:
#         line = line.strip()
#         if ':' in line:
#             timestamp, message = line.split(':', 1)
#             chat_history.append({'timestamp': timestamp, 'message': message})
#         else:
#             # handle the case where there are not exactly 2 values in the line
#             pass

#     df = pd.DataFrame(chat_history)
#     st.write(df)

# if __name__ == "__main__":
#     main()