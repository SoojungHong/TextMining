# libraries

import collections
import re
import sys

from experiments.place_category import joined_with_place_info
from lib.data_utils import *
from lib.file_utils import *
from lib.geo_utils import *
from lib.visual_utils import *

from sklearn.decomposition import PCA
from matplotlib import pyplot

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt




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


def get_long_queries_contains(query, df):
    all_queries = df['query_raw'].unique()
    for i in range(len(all_queries)):
        if all_queries[i] == query:
            return query


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
        return None
    else:
        return match['category_names'].iloc[0]  # list


def get_word_vector(model, word):
    print(word)
    word = re.sub('\W+', ' ', word)
    tokens = word.split()
    word_vec = ''
    if len(tokens) > 1:
        for w in range(len(tokens)):
            cleaned_word = re.sub('\W+', '', tokens[w])
            curr_vec = model.get_vector(cleaned_word)
            print(curr_vec)
    else:
        word = re.sub('\W+', '', word)
        word = word.lower()
        if word in model.vocab:
            word_vec = model.get_vector(word)
            print(word_vec)
    return word_vec


def get_food_type(pid):
    match = merged_data[merged_data['ppid'] == pid]
    if match.empty:
        print("empty")
        return None
    else:
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
    return place_desc


def represent_to_vector(model, desc):
    split_desc = desc.split()
    # remove special charactor
    #sum_all_vec = np.ndarray((300,), float)  # error - the vector is initialized with random values
    sum_all_vec = np.zeros((300,), dtype=float)

    for des_i in range(len(split_desc)):
        token = re.sub('\W+', '', split_desc[des_i])
        if token in model.vocab:
            desc_vec = model.get_vector(token)
            sum_all_vec = sum_all_vec + desc_vec

    return sum_all_vec


def pca_place_description(df):
    # visualize the word vector (description to one vector)
    import numpy as np
    from sklearn.decomposition import PCA
    from matplotlib import pyplot
    pca = PCA(n_components=2)  # number of components to keep
    result = pca.fit_transform((df['desc_vec'].values).tolist())
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    places = list(final_df['place_desc'].values)
    for i, word in enumerate(places):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)
    pyplot.show()


def get_vec_by_pre_trained_doc2vec(input_str):
    import gensim.models as g
    pre_trained_model = 'C:/Users/shong/Downloads/enwiki_dbow/enwiki_dbow/doc2vec.bin'
    # load model
    m = g.Doc2Vec.load(pre_trained_model)
    tokens = input_str.split()
    doc_vec = m.infer_vector(tokens)
    return doc_vec


def normalize(list, range): # range should be (lower_bound, upper_bound)
  l = np.array(list)
  a = np.max(l)
  c = np.min(l)
  b = range[1]
  d = range[0]

  m = (b - d) / (a - c)
  pslope = (m * (l - c)) + d
  return pslope


def get_z_distance_normalized(x_data_list, y_data_list, z_data_list):
    min_x = min(x_data_list)
    max_x = max(x_data_list)
    min_y = min(y_data_list)
    max_y = max(y_data_list)
    xy_ran = list()
    xy_ran.append(min(min_x, min_y))
    xy_ran.append(max(max_x, max_y))
    z_normalized_list = normalize(z_data_list, xy_ran)
    return z_normalized_list


# data
RECO_HEADER = [#removed intentionally]
RECO_DATA = "data.tsv"

# read data
df = read_file(RECO_DATA, RECO_HEADER)  # (209921, 18)

# get records contain given the input query
ret_query = get_long_queries_contains("'nearest gas station'", df)
print('given query :', ret_query)

# get all recommended place records on query and same query location
all_records = df[df['query_raw'] == ret_query]
all_timestamp = all_records['timestamp'].unique()

one_query_records = all_records[all_records['timestamp'] == all_timestamp[0]]  # we check only one timestamp which execute one query
print('query result shape :', one_query_records.shape)

one_query_records['distance'] = one_query_records.apply(lambda row: measure_distance(row.query_lat, row.query_lon, row.result_lat, row.result_lon), axis=1)

# get joined place data frame
merged_data = joined_with_place_info()

# add columns 'category_names' and 'food_type'
one_query_records['category_names'] = one_query_records.apply(lambda row: get_category_names(row.result_name), axis=1)
one_query_records['food_type'] = one_query_records.apply(lambda row: get_food_type(row.result_name), axis=1)

# google word2vec
model = load_google_w2v_model()
one_query_records['closest_category'] = one_query_records.apply(lambda row: get_closest_category(row.category_names, model, row.name_chosen), axis=1)
one_query_records['closest_food_type'] = one_query_records.apply(lambda row: get_closest_foodtype(row.food_type, model, row.name_chosen), axis=1)
one_query_records['place_desc'] = one_query_records.apply(lambda row: compose_place_description_by_word2vec(row.closest_category, row.closest_food_type), axis=1)

# Prepare data frame which does not contain empty category or empty food type
final_df = one_query_records[one_query_records['place_desc'] != '']

# option1 using word2vec
# final_df['desc_vec'] = final_df.apply(lambda row: represent_to_vector(model, row.place_desc), axis=1)

# option2 using doc2vec
final_df['desc_vec'] = final_df.apply(lambda row: get_vec_by_pre_trained_doc2vec(row.place_desc), axis=1)

# recommendation data frame
labels = ['query_raw', 'name_chosen', 'place_desc', 'desc_vec', 'distance']

# 'desc_vec' (description vector) dimension reduction using PCA
pca = PCA(n_components=2)  # number of components to keep
desc_list = (final_df['desc_vec'].values).tolist()
origin_vec = get_vec_by_pre_trained_doc2vec(final_df['query_raw'].iloc[0])
origin_name = final_df['query_raw'].iloc[0]
desc_list.append(origin_vec)  # add the vector of the query itself
result = pca.fit_transform(desc_list)

z_distance = (final_df['distance'].values).tolist()
z_distance.append(0)
x_data = result[:, 0]
y_data = result[:, 1]
point_labels = []

# find range of x and y value, normalize the distance (z values)
z_normalized = get_z_distance_normalized(x_data, y_data, z_distance)

# pca results in x- y- and distance in z-
visualize_3dimensional_pca(x_data, y_data, z_normalized, final_df, result)


# k-means algorithm (k = 2)
# 3-dimensional data points
df = pd.DataFrame({
    'x': x_data,
    'y': y_data,
    'z': z_normalized
})

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
colmap = {1: 'r', 2: 'g'}   # , 3:' b'}

fig = pyplot.figure()
ax = Axes3D(fig)
colors = map(lambda x: colmap[x+1], labels)

ax.scatter(df['x'], df['y'], df['z'], c=labels, cmap='viridis', edgecolor='k', s=40, alpha=0.5)

#give the labels to each point
places = list(final_df['place_desc'].values)
place_names = list(final_df['name_chosen'].values)
origin_name = final_df['query_raw'].iloc[0]
for i in range(len(result)):
    if i == (len(result) - 1):
        point_labels.append(origin_name)
    else:
        point_labels.append(place_names[i] + ':' + places[i])

for x_label, y_label, z_label, label in zip(df['x'], df['y'], df['z'], point_labels):
    ax.text(x_label, y_label, z_label, label, size=6)

for idx, centroid in enumerate(centroids):
    ax.scatter(*centroid, cmap=colmap[idx+1])
    ax.text(centroid[0], centroid[1], centroid[2], 'cluster_{}'.format(idx), None, size=6, color='Red')
pyplot.show()

