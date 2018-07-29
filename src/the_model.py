import matplotlib.pyplot as plt
import keras
from bs4 import BeautifulSoup
from urllib.request import urlopen
import datetime
import pandas as pd
import dateutil.parser
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import math
from ge_data_parser import *

#TODO: turn these constants into arguments
TARGET_ITEM = "Saradomin_brew_(4)"
FEATURE_ITEMS = ["Grimy_toadflax", "Crushed_nest", "Clean_toadflax", \
                 "Super_restore_(4)", "Toadflax_potion_(unf)", \
                 "Rocktail", "Saradomin_brew_flask_(6)", "Shark", \
                 "Mahogany_plank"]
selected_features = ["price", "trend"]
MAKE_BATCH_SIZE_LENGTH_OF_DATA = False
DEBUG_GRAPH = False
BATCH_SIZE = 32
EPOCHS = 80

GET_NEW_DATA = True #get the newest data from the Wiki

if GET_NEW_DATA:
    print("[+] Getting dataframes from internet...")
    feature_set, target_set = create_dataframes(TARGET_ITEM, FEATURE_ITEMS) 

    print("[+] Getting the latest features...")
    latest_features = feature_set.iloc[-1]
    latest_date = latest_features["date"]

    print("[+] Aligning dataframes by date...")
    feature_set, target_set = align_sets_by_date(feature_set, target_set)

    print("[!] Feature and target sets after aligning...")
    print(feature_set.head())
    print(feature_set.tail())
    print(target_set.head())
    print(target_set.tail())
    

    feature_set = feature_set.drop(columns="date")
    target_set = target_set.drop(columns="date")

    print("[+] Saving feature, target, and latest sets to pandas (../fs.panda, ts.panda, lf.panda)")
    feature_set.to_pickle("fs.panda")
    target_set.to_pickle("ts.panda")
    latest_features.to_pickle("lf.panda")  
else:
    feature_set = pd.read_pickle("fs.panda")
    target_set = pd.read_pickle("ts.panda")
    latest_features = pd.read_pickle("lf.panda")

feature_set, target_set = randomize_sets(feature_set, target_set)

selected_features_array = create_selected_features(FEATURE_ITEMS, TARGET_ITEM, selected_features)

feature_set = feature_set[selected_features_array]
latest_features = latest_features[selected_features_array]

data_length = len(feature_set)

training_feature_set = feature_set[:int(data_length/2):]
training_target_set = target_set[:int(data_length/2):]

testing_feature_set = feature_set[int(data_length/2)::]
testing_target_set = target_set[int(data_length/2)::]

num_features = len(list(training_feature_set.columns.values))

model = Sequential()

model.add(Dense(num_features, input_dim=num_features))
model.add(Activation('relu'))

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# used formula from comment 2 
alpha = 5 # usually 2-10
hidden_layer_neurons = int(data_length / (alpha * (1 + num_features)))
print("[+] The model has %i neurons in its hidden layer." % hidden_layer_neurons)
model.add(Dense(hidden_layer_neurons))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer='Adam',
            loss='mse')

if MAKE_BATCH_SIZE_LENGTH_OF_DATA:
    BATCH_SIZE = data_length

print("[+] Training the model.")
model.fit(training_feature_set, training_target_set, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("\n[+] Evaluating the model.")
testing_loss = float(model.evaluate(testing_feature_set, testing_target_set, batch_size=1))

print("[!] Our testing data rmse was: %f" %  (math.sqrt(testing_loss)) )

print("[!] The testing data std was: %f" % testing_target_set.std())

print("[!] Our model scored an rmse of %f above (or below) the std. " % (math.sqrt(testing_loss) - testing_target_set.std() ))
if DEBUG_GRAPH:
    plt.scatter(list(range(0, len(testing_feature_set))) , model.predict(testing_feature_set))
    plt.scatter(list(range(0, len(testing_feature_set))) , testing_target_set)
    plt.show()

latest_features = np.resize(latest_features[selected_features_array], (1, num_features))
print("[!] These are our latest features...")
print(latest_features)
print("[!] We predict %s will change by this much: %f." % (TARGET_ITEM, model.predict(latest_features)))
if GET_NEW_DATA:
    print("[!] The latest date collected was %s. Our model predicts the next day's so make sure this is current!\
            BTW: The way dates are calculated makes it so, if you're in PST, the date is actually +1 day." % str(latest_date))
