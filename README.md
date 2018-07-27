# radish_squash_mushroom_lettuce
An attempt to predict price changes in Runescape using Machine Learning.

I've written a parser for Runesape Wiki GE item price data.
The parser converts the data to a pandas array.
Then, we train a model on the data and evaluate it.

Currently, I've attempted to predict the next price change for saradomin brew (4).
My RMSE is currently around 2.48 percentage points.

I am real world testing the model by buying and selling according to its predictions.

Check out the rs_ml paper to find out how the model works.
