# radish_squash_mushroom_lettuce
An attempt to predict price changes in Runescape using Machine Learning.

ge_data_parser: I've written a parser for Runesape Wiki GE item price data. The parser currently only supports RS3 data. It downloads the item's webpage, parses the data format, and saves it into a pandas array. 

the_model: This is the main executable. It takes in a target item and feature items that will provide the data used in prediction. It creates a parser for each item, downloads its feature dataframe, and merges them together. Then, it creates a model and feeds it the training data. The model is then tested on testing data and the results are given out. Finally, it outputs its prediction based on the newest information it has.

Currently, I've attempted to predict the next price change for saradomin brew (4).
My RMSE is currently around 2.48 percentage points.

Check out the rs_ml paper to better find out how the model works and what data is being fed in.
