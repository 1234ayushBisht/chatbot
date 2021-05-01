# Import required modules
import nltk
from nltk.stem.lancaster import LancasterStemmer

from tensorflow.keras.models import load_model

import pandas 
import numpy
import pickle
import json
import random

# Defining our stemmer
stemmer = LancasterStemmer()

# Loading our model
model = load_model("my_model")

# Reading "intents.json", that is our data file
with open("intents.json") as source:
    data = json.load(source)

# Reading our saved "data.pickle"
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# This function takes a sentence and tokenize it and also stem all the words
def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]
    return sentence_words

# Return a bag of words array: 0 or 1 for each word that exists in the sentence
def bag_of_words(sentence, words, showDetails=True):
    sentence_word = cleanup_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_word:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if showDetails:
                    print("Found in bag - ", w)

    return numpy.array(bag)

# Return a sorted list of all the predicted intents.
def get_intents(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pandas.DataFrame([bag_of_words(sentence, words, False)], dtype=float, index=["input"])

    results = model.predict([input_data])[0]

    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((labels[r[0]], str(r[1])))
    
    return return_list

# Return a random response from the list of responses of the predicted intnet.
def get_reply(sentence):
    if len(sentence) != 0:
        all_intents = get_intents(sentence)
        intent = all_intents[0][0]
        p = float(all_intents[0][1])

        if p > 0.7:
            for _intent in data["intents"]:
                if _intent["tag"] == intent:
                    return random.choice(_intent["responses"]), intent
        else:
            return "I don't understand", "no_intent"
      

if __name__ == '__main__':
    # Running our chatbot in loop
    print("\n\nHello I am JARVIS, ask me something or enter q to exit")
    while True:
        sentence = input("\nYou: ")
        
        if sentence != "q":
            response = get_reply(sentence)
            print(f"Bot: {response[0]} | Intent: {response[1]}")
        else:
            print("Bye-Bye")
            exit()
    