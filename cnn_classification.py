import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc, vocab):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

def process_docs(directory, vocab, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc, vocab)
        documents.append(tokens)
    return documents

def load_clean_dataset(vocab, is_train):
    neg = process_docs('txt_sentoken/neg', vocab, is_train)
    pos = process_docs('txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def predict_sentiment(review, vocab, tokenizer, max_length, model):
    line = clean_doc(review, vocab)
    padded = encode_docs(tokenizer, max_length, [line])
    yhat = model.predict(padded, verbose=1)
    percent_pos = yhat[0,0]
    if round(percent_pos)==0:
        return (1-percent_pos), "NEGATIVE"
    return percent_pos, "POSITIVE"

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab,  False)

tokenizer = create_tokenizer(train_docs)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)

Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)

# model = define_model(vocab_size, max_length)
# model.fit(Xtrain, ytrain, epochs=10, verbose=1)

# model.save('model.h5')

model = load_model("model.h5")

# _, acc = model.evaluate(Xtrain, ytrain, verbose=1)
# print("Train accuracy: " + str(acc*100) + "%")

# _, acc = model.evaluate(Xtest, ytest, verbose=1)
# print("Test accuracy: " + str(acc*100) + "%")

text = "Brilliant movie. A must watch"
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))

text = "This is a bad movie. Do not watch it. It sucks."
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))