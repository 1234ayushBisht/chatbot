# Importing all the modules
import nltk
from nltk.stem import LancasterStemmer
import json
import numpy
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD 


# Defining our stemmer
stemmer = LancasterStemmer() 

# Reading our "intents.json" file
with open("intents.json") as source:
    data = json.load(source)

# A list that stores all words
words = []

# A list that stores all intents
labels = []

# A list that store all the words 
docs_x = []

# A list that stores all the intents 
docs_y = []

# Looping through all the intents in data
for intent in data["intents"]:
    # Looping through all the patterns in a intent
    for pattern in intent["patterns"]:
        # Tokenizing each pattern 
        wrds = nltk.word_tokenize(pattern)

        # Adding tokenized pattern to words list
        words.extend(wrds)

        # Pushing tokenized pattern to doc_x list
        docs_x.append(wrds)

        # Pushing respective intent of a pattern in doc_y list
        docs_y.append(intent["tag"])

    # Adding intent to labels list
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stemming each word of word list 
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
# Sorting word list
words = sorted(list(set(words)))

# Sorting labels
labels = sorted(labels)

# Training data list
training = [] 
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Converting data to numpy array
training = numpy.array(training)
output = numpy.array(output)

# Saving all the required variable in pickle file as they are required later in other file
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# The Model
model = Sequential()
# Adding layers to model
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

# The optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(training, output, epochs=200, batch_size=5, verbose=1)

# Saving model in "my_model" folder
model.save("my_model")
