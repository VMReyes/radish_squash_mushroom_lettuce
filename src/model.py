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

class ge_predictor:

    def __init__(self):
        """
        Initializes the model with empty values.
        """
        self.model = None
        self.training_features = None
        self.training_targets = None
        self.testing_features = None
        self.testing_targets = None
        self.data_set = False

    def set_data(self, training_features, training_targets, testing_features, \
                 testing_targets):
        """
        Input: training and testing features in the form of dataframes.
        Set the data for model and store it.
        """
        self.training_features = training_features
        self.training_targets = training_targets
        self.testing_features = testing_features
        self.testing_targets = testing_targets
        self.num_features = len(list(training_features.columns.values))
        self.data_length = len(training_features)

    def setup_model(self):
        """
        Sets up, compiles and stores a predefined model.
        """
        if not self.data_set:
            print("[!!] You do not have any data set into the model. \
                   Please run set_data before setting up them model!")
            return 0
        _model = Sequential()
        print("[!] The model has %i neurons in its input layer." % \
              self.num_features)
        _model.add(Dense(self.num_features, input_dim=self.num_features))
        _model.add(Activation('relu'))
        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        # used formula from comment 2
        alpha = 5 # usually 2-10
        hidden_layer_neurons = int(self.data_length/(alpha*(self.num_features+1)))
        print("[!] The model has %i neurons in its hidden layer." % hidden_layer_neurons)
        _model.add(Dense(hidden_layer_neurons))
        _model.add(Activation('relu'))
        _model.add(Dense(1))

        _model.compile(optimizer='Adam',
                    loss='mse')

        self.model = _model
