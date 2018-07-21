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
BATCH_SIZE = 64

def get_latest_features(feature_set, target_set):
    #last_feature_set = feature_set.iloc[-1]
    #latest_feature_set["%s price" % target_item] = latest_target_item_price
    pass

GET_NEW_DATA = True #get the newest data from the Wiki

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

    feature_set, target_set = randomize_sets(feature_set, target_set)
else:
    feature_set, target_set = load_saved_sets() #TODO: right now, we can't get saved data b/c we wouldn't be able to easily get
                                                #      the latest feature-set to make a prediction (we should pickle the latest
                                                #      feature set too...

###MERGING FEATURE DATAFRAMES MAY CAUSE DATE ERRORS, CHECK THIS

data_length = len(feature_set)
training_feature_set = feature_set[:int(data_length/2):]
training_target_set = target_set[:int(data_length/2):]

testing_feature_set = feature_set[int(data_length/2)::]
testing_target_set = target_set[int(data_length/2)::]

num_features = len(list(training_feature_set.columns.values))

model = Sequential()
model.add(Dense(1024, input_dim=num_features))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
#for i in range(num_features-1,1,-1):
 #   model.add(Dense(i))
  #  model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer='Adam',
              loss='mse')

print(training_feature_set.tail())
model.fit(training_feature_set, training_target_set, epochs=30, batch_size=BATCH_SIZE)
testing_loss = float(model.evaluate(testing_feature_set, testing_target_set, batch_size=1))
print("our testing data rmse was: %f" %  (math.sqrt(testing_loss)) )

print(testing_target_set.describe())

plt.scatter(list(range(0, len(testing_feature_set))) , model.predict(testing_feature_set))
plt.scatter(list(range(0, len(testing_feature_set))) , testing_target_set)
plt.show()
latest_features = np.resize(latest_features, (1, num_features))
print("These are our latest features...")
print(latest_features)
print("we predict %s will change by this much %f" % (TARGET_ITEM, model.predict(latest_features)))
