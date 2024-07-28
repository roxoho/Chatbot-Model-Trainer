import streamlit as st
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import zipfile

# Download NLTK data
from train import train_chatbot

def main():
    st.title("Chatbot Model Trainer")

    intents_placeholder = '''
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "How are you?", "What's up?"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
      "responses": ["Goodbye!", "See you soon!", "Take care!"]
    }
  ]
}
'''

    st.markdown("""
    Use ChatGPT or similar AI tools to generate intents for your desired topic, then paste the JSON here.
    Make sure to follow the structure shown in the placeholder.
    """)

    intents_input = st.text_area("Enter your intents JSON data:", value=intents_placeholder, height=300)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001, format="%.4f")
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=300, step=1)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=5, step=1)

    if st.button("Train Model"):
        if intents_input:
            try:
                intents = json.loads(intents_input)
                result = train_chatbot(intents, learning_rate, epochs, batch_size)
                st.success(result)
                
                # Provide download buttons for trained files
                with zipfile.ZipFile('chatbot_files.zip', 'w') as zipf:
                    zipf.write('words.pkl')
                    zipf.write('classes.pkl')
                    zipf.write('chatbot_model.h5')
                    
                    chatbot_py = '''
#importing librarbies
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from tensorflow.keras.models import load_model # type: ignore

lemmatizer = WordNetLemmatizer()

#loading the model
model = load_model("chatbot_model.h5")

#loading the intents data
intents = json.loads(open('intents.json').read())

#loading words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
ignore_letters = ["?","!",",",".","'"]


#function to tokenize and lemmatize the sentence from user
def clean_up_sentence(sentence):
    sentence_words =  nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]

    return sentence_words


#function to make sentence suitbale for model prediction using bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1

    return (np.array(bag),sentence_words)


#predicting the classes and their corresponding confidence scores for the given sentence, also applyting a threshold value
def predict_class(sentence):
    bow,sentence_words = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
        
    return (return_list,sentence_words)


#based on the class of the sentence, finding a suitable response from the intents data of same tag
def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result


#function used to provide response for the user input by utilizing the other functions
def chatbot_response(message):
    ints, sentence_words = predict_class(message)
    if len(ints) == 0:
        return random.choice([resp for intent in intents['intents'] if intent['tag'] == 'fallback' for resp in intent['responses']])
        
    return get_response(ints, intents)


#If you want to run this code separately, uncomment the below lines and run this file or you can comment the below lines and use the functions in other applications

while True:
   mes = input("")
   print(chatbot_response(mes))
'''
                    with open('chatbot.py', 'w') as f:
                        f.write(chatbot_py)
                    zipf.write('chatbot.py')

                    requirements='''numpy
tensorflow
nltk
flask
keras'''
                    with open('requirements.txt','w') as f:
                        f.write(requirements)
                    zipf.write('requirements.txt')

                    with open('intents.json', 'w') as f:
                        f.write(intents_input )
                    zipf.write('intents.json')

                # Read the zip file into a byte stream
                with open('chatbot_files.zip', 'rb') as f:
                    zip_bytes = f.read()

                st.download_button(
                    label="Download chatbot_files.zip",
                    data=zip_bytes,
                    file_name='chatbot_files.zip',
                    mime='application/zip'
                )

            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
        else:
            st.warning("Please enter intents data before training the model.")

    st.markdown("---")

    st.markdown("""
    Install the required libraries through "pip install -r requirements.txt".
    Run the Chatbot.py file or comment the last 3 lines and use the functions in a web app or other(shown in the example video).
    """)

    st.markdown("---")

    st.header("Example Video")
    st.video("https://www.youtube.com/watch?v=your_video_id_here")
    st.markdown("Replace 'your_video_id_here' with the actual YouTube video ID of your example video.")

if __name__ == "__main__":
    main()
