import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob  # For sentiment analysis

# Ensure proper SSL setup for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
try:
    file_path = os.path.abspath("./intents.json")
    print(f"Intent file path: {file_path}")
    with open(file_path, "r") as file:
        intents = json.load(file)
    print("Intents loaded successfully.")
except Exception as e:
    print(f"Error loading intents file: {e}")
    intents = []

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
if intents:
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)

# Train the model
if patterns:
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
    print("Model trained successfully.")
else:
    print("No data to train the model.")

# Fallback responses
fallback_responses = [
    "I'm sorry, I didn't understand that. Could you rephrase?",
    "I'm not sure how to respond. Could you clarify?",
]

# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative"

def chatbot(input_text):
    try:
        input_text = vectorizer.transform([input_text])
        if len(input_text.nonzero()[1]) == 0:
            return random.choice(fallback_responses)
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except Exception as e:
        print(f"Error in chatbot response generation: {e}")
    return random.choice(fallback_responses)

counter = 0
user_name = None  # Placeholder for personalization

def main():
    global counter, user_name
    st.title("Enhanced Chatbot with NLP")

    # Debug print statement
    print("Main function loaded successfully.")

    # Sidebar menu
    menu = ["Home", "Conversation History", "Add New Intent", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Debug print statement for menu selection
    print(f"Menu choice: {choice}")

    if choice == "Home":
        if user_name is None:
            user_name = st.text_input("Enter your name:", key="user_name")
            if user_name:
                st.write(f"Hello {user_name}! Welcome to the chatbot.")
        else:
            st.write(f"Welcome back, {user_name}! How can I assist you today?")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Sentiment', 'Timestamp'])

        counter += 1
        user_input = st.text_input(f"{user_name}, type your message below:", key=f"user_input_{counter}")

        if user_input:
            sentiment = analyze_sentiment(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, sentiment, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()


    elif choice == "Conversation History":

        st.header("Conversation History")

        if os.path.exists('chat_log.csv'):

            try:

                with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:

                    csv_reader = csv.reader(csvfile)

                    headers = next(csv_reader, None)  # Skip header row

                    for row in csv_reader:

                        if len(row) == 4:  # Ensure the row has exactly 4 columns

                            st.text(f"User: {row[0]} | Sentiment: {row[2]}")

                            st.text(f"Chatbot: {row[1]} | Timestamp: {row[3]}")

                            st.markdown("---")

                        else:

                            st.warning("Malformed row detected in chat log. Skipping.")

                if st.button("Download Chat History"):
                    with open("chat_log.csv", "r") as file:
                        st.download_button("Download", file, file_name="chat_log.csv", mime="text/csv")

            except Exception as e:

                st.error(f"Error reading conversation history: {e}")

        else:

            st.write("No conversation history available.")


    elif choice == "Add New Intent":
        st.header("Add New Intent")
        new_intent_tag = st.text_input("Intent Tag:")
        new_patterns = st.text_area("Patterns (one per line):")
        new_responses = st.text_area("Responses (one per line):")
        if st.button("Add Intent"):
            new_intent = {
                "tag": new_intent_tag,
                "patterns": new_patterns.split("\n"),
                "responses": new_responses.split("\n"),
            }
            intents.append(new_intent)
            with open(file_path, "w") as file:
                json.dump(intents, file, indent=4)
            st.success(f"Intent '{new_intent_tag}' added successfully!")

    elif choice == "About":
        st.write("This chatbot now includes enhanced features like sentiment analysis, personalized interactions, and the ability to add new intents dynamically. Built using NLP techniques and Streamlit.")

if __name__ == '__main__':
    main()
