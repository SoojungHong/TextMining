# libraries

import collections
import re
import sys

from matplotlib import pyplot

from experiments.place_category import joined_with_place_info
from lib.data_utils import *
from lib.file_utils import *
from lib.geo_utils import *
from lib.visual_utils import *


def load_google_w2v_model():
    from gensim.models import KeyedVectors
    filename = 'C:/Users/shong/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
    # binary file 3.4GB
    model = KeyedVectors.load_word2vec_format(filename, binary=True)  # 43 seconds to load
    return model


def get_all_queries_contains(query, df):
    contain_res_df = df[df['query_raw'].str.contains(query)]
    queries_contains_restaurant = contain_res_df['query_raw']
    return queries_contains_restaurant.unique()


def get_one_token_query_set(queries):
    one_token_set = list()
    for i in range(len(queries)):
        current_query = queries[i]  # narray

        if len(current_query.split()) is 1:
            # remove special character # ToDo : handle the special char but dataset contains special char at the moment
            # cleanStr = re.sub('\W+','', current_query)
            # print(cleanStr)
            one_token_set.append(current_query)

    return one_token_set


def get_word_mover_distance(w2v_model, doc1, doc2):
    dist = w2v_model.wmdistance(doc1, doc2)
    return dist


def get_category_names(pid):
    match = merged_data[merged_data['ppid'] == pid]
    if match.empty:
        print("empty")
        return None
    else:
        print(type(match['category_names'].iloc[0]))
        return match['category_names'].iloc[0]  # list


def get_word_vector(model, word):
    print(word)
    word = re.sub('\W+', ' ', word)
    tokens = word.split()
    print(tokens)
    word_vec = ''
    if len(tokens) > 1:
        for w in range(len(tokens)):
            cleaned_word = re.sub('\W+', '', tokens[w])
            curr_vec = model.get_vector(cleaned_word)
            print(curr_vec)
            print(len(curr_vec))
    else:
        word = re.sub('\W+', '', word)
        word = word.lower()
        if word in model.vocab:
            word_vec = model.get_vector(word)
            print(word_vec)
            print(len(word_vec))
    return word_vec


def get_food_type(pid):
    match = merged_data[merged_data['ppid'] == pid]
    if match.empty:
        print("empty")
        return None
    else:
        print(type(match['food_type'].iloc[0]))
        return match['food_type'].iloc[0]  # list


def get_closest_category(category_list, model, pname):
    min_dist = sys.maxsize
    closest_cat_name = ""
    if type(category_list) is type(None):
        print('NoneType')
    else:
        if len(category_list) > 0:
            for c in range(len(category_list)):
                curr_category = re.sub('\W+', ' ',  category_list[c])
                place = re.sub('\W+', '', pname)
                distance = model.wmdistance(pname, curr_category)
                if distance < min_dist:
                    min_dist = distance
                    closest_cat_name = category_list[c]
            #print(min_dist)
            #print(closest_cat_name)
            #get_word_vector(model, closest_cat_name)
    return (re.sub('\W+', ' ', closest_cat_name)).lower()


def get_closest_foodtype(food_type_list, model, pname):
    # ToDo : do check if the food_type set is empty - if it is empty, return empty, empty food type will be added as a adjective
    min_dist = sys.maxsize
    closest_food_name = ""
    if type(food_type_list) is type(None):
        print('NoneType')
    else:
        if len(food_type_list) > 0:
            for c in range(len(food_type_list)):
                curr_food_type = re.sub('\W+', '', food_type_list[c])
                place = re.sub('\W+', '', pname)
                distance = model.wmdistance(place, curr_food_type)
                #print(curr_food_type, distance)
                if distance < min_dist:
                    min_dist = distance
                    closest_food_name = food_type_list[c]
            #print(min_dist)
            #print(closest_food_name
            #get_word_vector(model, closest_food_name)
    return (re.sub('\W+', ' ', closest_food_name)).lower()


def compose_place_description_by_word2vec(category_name, food_type):
    place_desc = food_type + "" + category_name
    print(place_desc)
    return place_desc


def represent_to_vector(model, desc):
    split_desc = desc.split()
    sum_all_vec = np.ndarray((300,), float)  # we know the word vector size is 300
    for des_i in range(len(split_desc)):
        print(split_desc[des_i])
        desc_vec = model.get_vector(split_desc[des_i])
        print(type(desc_vec))  # ndarray type
        sum_all_vec = sum_all_vec + desc_vec

    return sum_all_vec



def test_visualize():
    import numpy as np
    from sklearn.decomposition import PCA
    from gensim.models import Word2Vec
    from sklearn.decomposition import PCA
    from matplotlib import pyplot
    # define training data
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
    			['this', 'is', 'the', 'second', 'sentence'],
    			['yet', 'another', 'sentence'],
    			['one', 'more', 'sentence'],
    			['and', 'the', 'final', 'sentence']]

    # train model
    model = Word2Vec(sentences, min_count=1)
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    print(X)
    print(type(X)) # array   X[0] is array , X is array
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
	    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


# test
#test_visualize()

# data
RECO_HEADER = ['A', 'B', 'C']
RECO_DATA = "C:data.tsv"

# read data
df = read_file(RECO_DATA, RECO_HEADER)  # (209921, 18)

# get records contain query_raw == 'restaurant'
ret = get_all_queries_contains("restaurant", df)
one_set = get_one_token_query_set(ret)
# print(one_set)

# restaurant_query = one_set[0] # query - 'restaurants'
restaurant_query = one_set[1]  # query - 'restaurant'

# get all recommended place records on query and same query location
all_records = df[df['query_raw'] == restaurant_query]
all_timestamp = all_records['timestamp'].unique()

one_query_records = all_records[
    all_records['timestamp'] == all_timestamp[0]]  # we check only one timestamp which execute one query
# print(one_query_records)  # shape : 22 * 18

# add 'place_desc' by using Google word2vec
# do first prepare the merged data frame and load the google word2vec model

# get joined place data frame
merged_data = joined_with_place_info()

# add columns 'category_names' and 'food_type'
one_query_records['category_names'] = one_query_records.apply(lambda row: get_category_names(row.result_name), axis=1)
# print(one_query_records)  # 22 * 19

one_query_records['food_type'] = one_query_records.apply(lambda row: get_food_type(row.result_name), axis=1)
# print(one_query_records)  # 22 * 20

# google word2vec
model = load_google_w2v_model()

one_query_records['closest_category'] = one_query_records.apply(lambda row: get_closest_category(row.category_names, model, row.name_chosen), axis=1)
print(one_query_records['closest_category'])

one_query_records['closest_food_type'] = one_query_records.apply(lambda row: get_closest_foodtype(row.food_type, model, row.name_chosen), axis=1)
print(one_query_records['closest_food_type'])

one_query_records['place_desc'] = one_query_records.apply(lambda row: compose_place_description_by_word2vec(row.closest_category, row.closest_food_type), axis=1)
# remove special character
# one_query_records['place_desc'] = one_query_records.apply(lambda row: re.sub('\W+', '', row.place_desc), axis=1)
print(one_query_records['place_desc'])

labels = ['name_chosen', 'place_desc']  # ['query_raw', 'name_chosen', 'place_desc']
print(one_query_records[labels])


# Prepare data frame which does not contain empty category or empty food type
final_df = one_query_records[one_query_records['place_desc'] != '']

#test = represent_to_vector(model, final_df['place_desc'].iloc[0])
#print(test)

final_df['desc_vec'] = final_df.apply(lambda row: represent_to_vector(model, row.place_desc), axis=1)
print(type(final_df['desc_vec'].iloc[0]))  # ndarray


# visualize the word vector (description to one vector)
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
pca = PCA(n_components=2)  # number of components to keep
result = pca.fit_transform((final_df['desc_vec'].values).tolist())
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
places = list(final_df['place_desc'].values)
for i, word in enumerate(places):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)
pyplot.show()

# ToDo : check the two vectors of same place description are same or not, for example, 'American Casual Dining' 

