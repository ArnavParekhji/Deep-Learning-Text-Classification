import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump, load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Embedding, Flatten, Input, Dropout, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_dataset(filename):
    return load(open(filename, "rb"))

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

def process_docs(directory, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding="post")
    return padded

def load_clean_dataset(is_train):
    neg = process_docs('txt_sentoken/neg', is_train)
    pos = process_docs('txt_sentoken/pos', is_train)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)

def define_model(length, vocab_size):
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation="relu")(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation="relu")(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation="relu")(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])

    dense1 = Dense(10, activation="relu")(merged)
    outputs = Dense(1, activation="sigmoid")(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

# train_docs, ytrain = load_clean_dataset(True)
# test_docs, ytest = load_clean_dataset(False)
# save_dataset([train_docs, ytrain], 'train.pkl')
# save_dataset([test_docs, ytest], 'test.pkl')

train_lines, train_labels = load_dataset("train.pkl")
test_lines, test_labels = load_dataset("test.pkl")

length = max_length(train_lines)
tokenizer = create_tokenizer(train_lines)
vocab_size = len(tokenizer.word_index) + 1

train_X = encode_text(tokenizer, train_lines, length)
test_X = encode_text(tokenizer, test_lines, length)

# model = define_model(length, vocab_size)
# model.fit([train_X, train_X, train_X], train_labels, epochs=7, batch_size=16, verbose=1)
# model.save("model.h5")

model = load_model("model.h5")

_, acc = model.evaluate([train_X, train_X, train_X], train_labels, verbose=1)
print("Train accuracy: " + str(acc*100) + "%")
_, acc = model.evaluate([test_X, test_X, test_X], test_labels, verbose=1)
print("Test accuracy: " + str(acc*100) + "%")