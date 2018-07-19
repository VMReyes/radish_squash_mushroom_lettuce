from bs4 import BeautifulSoup
from urllib.request import urlopen
import datetime
import pandas as pd
import dateutil.parser
import numpy as np

def merge_feature_dataframes(dataframes):
    feature_set = dataframes[0]
    for feat in dataframes[1::]:
        #TODO: Merging dataframes may be problematic if they do not share identical date sequences. Look further into this.
        #      It would be smart to make sure we are merging the right dates correctly.
        feature_set = feature_set.merge(feat.drop(columns=["week_day_0", "week_day_1", "week_day_2", "week_day_3", "week_day_4", "week_day_5", "week_day_6"]), on="date")
    #feature_set = feature_set.align(feat)
    return feature_set
    
class Wiki_GE_Parser:
    
    def __init__(self, item_name):
        self.item_name = item_name
        self.data = None

    def get_data(self):
        """
        renames the price feature to include the item's name
        returns the dataframe saved in the parser
        """
        self.data["%s price" % self.item_name] = self.data["price"]
        self.data = self.data.drop("price", axis=1)
        return self.data

    def grab_data_from_wiki(self):
        """
        retrieves price data from wiki page and saves it
        as an array of string data in the form of "date:price"

        returns: 1 if successful, 0 otherwise
        """
        
        if self.item_name == None:
            print("You haven't set an item name yet!")
            return 0
        wiki_page = "http://runescape.wikia.com/wiki/Module:Exchange/%s/Data" % self.item_name
        print("accessing %s..." %wiki_page)
        page = urlopen(wiki_page)
        ge_soup = BeautifulSoup(page, 'html.parser')
        self.data = ge_soup.find_all(attrs={'class':'st0'})
        return 1
        
    def convert_data_to_dataframe(self):
        """
        converts ge data from scraper to a dataframe with the form of [weekday, price]
        also, the dataframe removes consecutive data points with the same weekday
        """
        ge_data = self.data

        skipped = 0

        date_series = []
        price_series = []
        weekday_series = []

        for i in range(len(ge_data)):
            date, weekday, price = self.parse_ge_data_string(ge_data[i].text.strip())
            if len(weekday_series) > 0:
                if weekday_series[-1] == weekday:
                    skipped += 1
                    pass
            date_series.append(datetime.datetime.fromtimestamp(date).isoformat())
            weekday_series.append(weekday)
            price_series.append(price/10000)
        print("we skipped %i entries because they shared the same date" % skipped)
        
        self.data = pd.DataFrame({"price":pd.Series(price_series),
                                  "date":pd.Series(date_series)}).join(pd.get_dummies(pd.Series(weekday_series), prefix="week_day"))
    
    def create_trend_column(self, item_type):
        """
        creates a trend column (item_type chooses if its a future trend or past trend)
        item_type: "feature item" - trend column shows the price change to the current timestamp
                   "target item" - trend column shows the upcoming price change from the current timestamp
        returns: Nothing, just modifies data within the parser to include a new feature 
        """
        trend_series = []
        previous_price = None
        for index, row in self.data.iterrows():
            if previous_price == None and item_type == "feature item":
                trend_series.append(0.0)
            elif item_type == "target item":
                try:
                    delta = self.prices_to_percentage_delta(row["price"], self.data["price"][index + 1])
                    trend_series.append(delta)
                except:
                    
                    pass
            
            if item_type == "feature item" and previous_price:
                trend_series.append(self.prices_to_percentage_delta(previous_price, row["price"]))


            previous_price = row["price"]
        
        self.data["%s trend" % self.item_name] = pd.Series(trend_series)
    
    def parse_ge_data_string(self, data_string):
        """
        inputs: string in the form " 'date:price[:volume_data]' "
        we do not use volume data currently, so we cut it off

        returns: tuple in the form of (date_integer, weekday from 0 to 6, price)
        """
        if data_string.count(":") > 1:  # deals with some data having volume data (we cut it off)
            colon2_index = self.find_second_colon_index(data_string)
            data_string = data_string[1:colon2_index:]   #removes the first quotation mark
        else:
            data_string = data_string[1:-1:]
        colon_index = data_string.find(":")
        date = int(data_string[0:colon_index])
        weekday = int(datetime.datetime.fromtimestamp(date).weekday())
        price = int(data_string[colon_index+1:len(data_string):])
        return (date, weekday, price)
    
    def find_second_colon_index(self, data_string):
        """
        returns the index of the second colon in data_string
        """
        count = 0
        for i in range(len(data_string)):
            if data_string[i] == ':':
                count += 1
                pass
            if data_string[i] == ":" and count > 1:
                return i

    def prices_to_percentage_delta(self, previous_price, cur_price):
        """
        takes a previous price and a current price and returns the percentage of change
        from previous to current
        """
        return( (cur_price-previous_price)/previous_price*100 )

def create_selected_features(feature_item_names, selected_features):
    """
    takes arrays of feature item names and selected features
    returns an array of features that match the item names
    """
    ret_features = []

    for item_name in feature_item_names:
        for feature in selected_features:
            ret_features.append("%s %s" % (item_name, feature) )
    print(ret_features)
    return ret_features
    
def create_dataframes(target_item_name, feature_item_names):
    """
    returns an array with two elements,
    element 1: dataframe containing target item dataframe (future trend)
    element 2: dataframe containing selected features of feature_item_names
    """
    feature_item_dataframes = []

    a = Wiki_GE_Parser(target_item_name)
    a.grab_data_from_wiki()
    a.convert_data_to_dataframe()
    a.create_trend_column("target item")
    target_item_name_dataframe = a.get_data()

    for item in feature_item_names:
        a = Wiki_GE_Parser(item)
        a.grab_data_from_wiki()
        a.convert_data_to_dataframe()
        a.create_trend_column("feature item")
        feature_item_dataframes.append(a.get_data())

    feature_set = merge_feature_dataframes(feature_item_dataframes)
    feature_set = feature_set.merge(target_item_name_dataframe[ ["%s price" % target_item_name ,"date"] ], on="date")[:-1:] #adds target_item price's feature

    target_set = target_item_name_dataframe[["%s trend" % target_item_name,"date"]][:-1:]

    return [feature_set, target_set]


def align_sets_by_date(feature_set, target_set):
    """
    aligns feature and target sets by date by removing mismatched dates
    returns: array with [target_set, feature_set]
    """
    switches = 1
    last_index = 0
    tot_switch = 0
    while switches:
        for index in range(last_index,len(feature_set["date"])):
            #print(index)
            switches = 0
            #print(target_set.iloc[[index]]["date"])
        #print(row["date"], target_set["date"][index])
        #print(index, target_set.size)
        #print(target_set["date"].tail())
            if dateutil.parser.parse( feature_set.iloc[index]["date"] ) > dateutil.parser.parse(target_set.iloc[index]["date"]):
                target_set = target_set.drop(target_set.index[index])
                target_set.index = range(len(target_set))
                switches = 1
                tot_switch += 1
                last_index = index
                break
            elif dateutil.parser.parse( feature_set.iloc[index]["date"] ) < dateutil.parser.parse(target_set["date"][index]):
                feature_set = feature_set.drop(feature_set.index[index])
                feature_set.index = range(len(feature_set))
                las_index=index
                switches = 1
                tot_switch += 1
                break
            #print(feature_set["date"][index-3:index+1:1], target_set["date"][index-3:index+1:1])
    print("we switched %i times" % tot_switch)
    return [feature_set, target_set]

def randomize_sets(feature_set, target_set):
    """
    randomizes feature_set and target_set by the same rules
    returns an array with feature_set and target_set
    """
    new_reindex = np.random.permutation(feature_set.index)
    feature_set = feature_set.reindex(new_reindex)
    target_set = target_set.reindex(new_reindex)
    return [feature_set, target_set]

def save_sets(feature_set, target_set):
    """
    saves feature and target sets into pickles
    """
    feature_set.to_pickle("feature_set.panda")
    target_set.to_pickle("target_set.panda")

def load_saved_sets():
    """
    returns pickled feature and target sets in a 2 element array
    """
    return [pd.read_pickle("feature_set.panda"), pd.read_pickle("target_set.panda")]




