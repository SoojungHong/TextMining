from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from stanfordcorenlp import StanfordCoreNLP
from sklearn.utils import shuffle
from nltk import word_tokenize

import pandas as pd
import numpy as np
import random
import spacy
import torch
import json
import nltk


class StandardNLP:

    # -------------------------------------------------------------------------------------------------------------------
    # Instruction to start StanfordCoreNLP
    # -------------------------------------------------------------------------------------------------------------------
    # Execute the command in terminal
    # First, go to C:\Users\shong\Documents\Next_Generation_Location_Service\NLP_tools\stanford-corenlp-full-2018-10-05\
    # > cd stanford-corenlp-full-2018-10-05
    # > java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000
    # -------------------------------------------------------------------------------------------------------------------

    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = nltk.defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

    def show_parse_tree(self, query):
        # show it with tree
        from nltk.parse.corenlp import CoreNLPParser
        cnlp = CoreNLPParser('http://localhost:9000')
        next(cnlp.raw_parse(query)).draw()


# -----------------------
# Utility functions
# -----------------------

def get_category_frame_with_OneHotEncoder(df):
    ohc = OneHotEncoder()
    ohe = ohc.fit_transform(df.Category.values.reshape(-1, 1)).toarray()    # all hot one encoding for all rows
    dfOneHot = pd.DataFrame(ohe, columns=["HotEncoding_"+str(ohc.categories_[0][i]) for i in range(len(ohc.categories_[0]))])
    dfh = pd.concat([df, dfOneHot], axis=1)
    return dfh


def get_category_frame_with_LabelEncoder(df):
    # Encoding with Label
    df['Category_Label_Encoded'] = LabelEncoder().fit_transform(df.Category)  # Label : ADDRESS 0, AREA 1, EVENT 2, GEOLOCATION 3, NONE 4, PLACE 5, ROUTE 6
    df.to_csv(queries_feature_encoded_label_file)
    return df


def show_count_per_category(df):
    data = get_category_frame_with_LabelEncoder(df)
    print(data.groupby('Category_Label_Encoded').count())


def read_query_and_make_list(df):
    queries = []
    for index, row in df.iterrows():
        tokenized_query = sNLP.word_tokenize(row['Query'])
        queries = queries.append(tokenized_query)


def sentence_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw


def words_embedding(query, embedding_model):
    query_in_tokens = sNLP.word_tokenize(query)
    ret = sentence_vectorizer(query_in_tokens, embedding_model)
    return ret


def query_embedding_usingFastText(df, embedding_model):
    df['Query_vector'] = df.apply(lambda row: words_embedding(row['Query'], embedding_model), axis=1)
    return df


# -------------------------------------------------------
#  raw data : queries category (label) and/or features
# -------------------------------------------------------

queries_feature_file = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/query_feature_dataframe_with_label.csv"
queries_feature_encoded_label_file = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/query_feature_dataframe_with_encoded_label.csv"
queries_feature_query_embedding = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/query_feature_query_embedding.csv"
queries_label_file_ver2 = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/query_with_label_ver2.csv"
queries_label_file_ver2_shuffle = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/query_with_label_ver2_shuffle.csv"

query_columns = ['Query', 'Strong_Target', 'Strong_Compound_Target', 'Weak_Target', 'Weak_Compound_Target', 'Default_Target', 'Category']
query_columns_ver2 = ['Query', 'Category']

data = pd.read_csv(queries_label_file_ver2, encoding='unicode_escape')
df = data[query_columns_ver2]
df = shuffle(df)    # shuffle rows
df.to_csv(queries_label_file_ver2_shuffle, index=False)


# ==========================================================
#  experiment : query intent type classifier using PyText
# ==========================================================

sNLP = StandardNLP()
spacy_en = spacy.load("en_core_web_sm")

raw_data = pd.read_csv(queries_label_file_ver2_shuffle)
print('reading data.....\n', raw_data)
labels = ['Query', 'Category']
dataframe_query_label = raw_data[labels]

BATCH_SIZE = 1
SEED = 1234
MAX_VOCAB_SIZE = 25_0000
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare torch data from tabular data
queries_label_tabular = "C:/Users/shong/PycharmProjects/QuerySpellCorrection/data/tabular_data.csv"
dataframe_query_label.to_csv(queries_label_tabular, index=False)

from torchtext import data  # import data should be here before define TEXT and LABEL
TEXT = data.Field(sequential=True, tokenize=word_tokenize, lower=False, fix_length=None)
LABEL = data.Field(sequential=False, use_vocab=True, unk_token=None)
tData = data.TabularDataset(path=queries_label_tabular, format='csv', skip_header=True, fields=[('Query', TEXT), ('Category', LABEL)])

train_data, test_data = tData.split(split_ratio=0.8, random_state=random.seed(SEED))
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

# vocabulary
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# set up iteration
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), sort=False, #sort=False fixed the bug
    batch_size=BATCH_SIZE, device=device)


import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,    # input channel is one - it is one-dimensional embedding
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))  # the kernel size can be array e.g. [2,3,4]
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [2]  # if batch size > 2, then consider filter size array [2, 3, 4]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)


def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)   # get the index with the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.LongTensor([y.shape[0]])


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.Query)  # Fix (Yah! :)
        loss = criterion(predictions, batch.Category)
        acc = categorical_accuracy(predictions, batch.Category)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.Query)
            loss = criterion(predictions, batch.Category)
            acc = categorical_accuracy(predictions, batch.Category)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 50

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut5-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('tut5-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

import spacy
nlp = spacy.load("en_core_web_sm")

def predict_class(model, sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()


# ------------------------
# prediction test
# ------------------------
query_1 = "30.222, 34.1234"
query_2 = "best restaurant nearby"

pred_class_1 = predict_class(model, query_1)
print("query : ", query_1)
print(f'Predicted class is: {pred_class_1} = {LABEL.vocab.itos[pred_class_1]}')

print("query : ", query_2)
pred_class_2 = predict_class(model, query_2)
print(f'Predicted class is: {pred_class_2} = {LABEL.vocab.itos[pred_class_2]}')



