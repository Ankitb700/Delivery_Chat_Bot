import streamlit as st
import random
import torch
import json

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load the pre-trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Initialize chat history
chat_history = []

# Streamlit app
st.title("Delivery Chatbot")
st.sidebar.header("User Input")

# User input textbox
user_input = st.sidebar.text_input("You:", "")

if user_input:
    # Process user input when it's not empty
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Generate bot response
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = f"{bot_name}: {random.choice(intent['responses'])}"
                chat_history.append((user_input, bot_response))
    else:
        bot_response = f"{bot_name}: I do not understand..."
        chat_history.append((user_input, bot_response))

# Display chat history
for user_msg, bot_msg in chat_history:
    st.text(f"You: {user_msg}")
    st.text(bot_msg)

# Streamlit settings
st.sidebar.markdown("---")
st.sidebar.markdown("### Chatbot Info")
st.sidebar.text("This is a simple chatbot for a delivery application.")
st.sidebar.text("Type your messages in the sidebar to chat.")

# Quit button
if st.sidebar.button("Quit"):
    st.balloons()
    st.stop()
