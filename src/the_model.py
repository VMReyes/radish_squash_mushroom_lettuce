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
FEATURE_ITEMS = ["Grimy_toadflax", "Crushed_nest", "Clean_toadflax", "Super_restore_(4)", "Toadflax_potion_(unf)"]
selected_features = ["price", "trend"]
MAKE_BATCH_SIZE_LENGTH_OF_DATA = True
BATCH_SIZE = 1
EPOCHS = 80

GET_NEW_DATA = False #get the newest data from the Wiki

if GET_NEW_DATA:
    print("Getting dataframes from internet...")
    feature_set, target_set = create_dataframes(TARGET_ITEM, FEATURE_ITEMS) 

    print("Aligning dataframes by date...")
    feature_set, target_set = align_sets_by_date(feature_set, target_set)

    print("feature and target sets after aligning")
    print(feature_set.tail())
    print(target_set.tail())

    feature_set = feature_set.drop(columns="date")
    target_set = target_set.drop(columns="date")

    latest_features = feature_set.iloc[-1]

    feature_set.to_pickle("fs.panda")
    target_set.to_pickle("ts.panda")
    latest_features.to_pickle("lf.panda")  
else:
    feature_set = pd.read_pickle("fs.panda")
    target_set = pd.read_pickle("ts.panda")
    latest_features = pd.read_pickle("lf.panda")
    feature_set, target_set = randomize_sets(feature_set, target_set)

selected_features_array = create_selected_features(FEATURE_ITEMS, selected_features)

feature_set = feature_set[selected_features_array]

data_length = len(feature_set)

training_feature_set = feature_set[:int(data_length/2):]
training_target_set = target_set[:int(data_length/2):]

testing_feature_set = feature_set[int(data_length/2)::]
testing_target_set = target_set[int(data_length/2)::]

num_features = len(list(training_feature_set.columns.values))

model = Sequential()

model.add(Dense(num_features+1, input_dim=num_features))
model.add(Activation('relu'))

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# used formula from comment 2 
alpha = 5 # usually 2-10
hidden_layer_neurons = int(data_length / (alpha * (1 + num_features+1)))
print("[+] The model has %i neurons in its hidden layer." % hidden_layer_neurons)
model.add(Dense(hidden_layer_neurons))
model.add(Activation('relu'))

model.add(Dense(1))


model.compile(optimizer='Adam',
            loss='mse')

print(training_feature_set.tail())

if MAKE_BATCH_SIZE_LENGTH_OF_DATA:
    BATCH_SIZE = data_length

model.fit(training_feature_set, training_target_set, epochs=EPOCHS, batch_size=BATCH_SIZE)
testing_loss = float(model.evaluate(testing_feature_set, testing_target_set, batch_size=1))
print("our testing data rmse was: %f" %  (math.sqrt(testing_loss)) )

print("the testing data std was: %f" % testing_target_set.std())

print("Our model scored an rmse of %f above (or below) the std. " % (math.sqrt(testing_loss) - testing_target_set.std() ))

plt.scatter(list(range(0, len(testing_feature_set))) , model.predict(testing_feature_set))
plt.scatter(list(range(0, len(testing_feature_set))) , testing_target_set)
plt.show()
latest_features = np.resize(latest_features[selected_features_array], (1, num_features))
print("These are our latest features...")
print(latest_features)
print("we predict %s will change by this much %f" % (TARGET_ITEM, model.predict(latest_features)))

