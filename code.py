#Importing the libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
from textblob import TextBlob
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten,BatchNormalization
from keras.models import Sequential, load_model, model_from_config
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import cohen_kappa_score
import random
from PIL import Image
import language_check

#Reading the dataset
data = pd.read_csv("training_set_rel3.tsv", sep='\t', encoding = "ISO-8859-1")
col_to_keep =['essay','domain1_score']
data = data[col_to_keep]

#Adding the image widget
image = Image.open('nlp.png')
st.image(image, use_column_width=True)

#Adding the title and subsequent text
st.title("Welcome to Automated Essay Grading System")
st.header("""
This is a system which lets you input a given essay for a given prompt and the system will return a score for the same
""")
st.warning("The following is the essay prompt. It belongs to the persuasive form of essay")
st.success("""
More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

Write an essay to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.

""")

#Basic input
name = st.text_input("Before we proceed, enter your name in the box below: ")
organ = ['Academic', 'Non Academic','Corporate','Other']
organisation = st.multiselect("Enter the purpose of using this software:", organ)
st.write("All righty then, Click on this button to view the dataset")
click = st.button("CLICK")
if click == True:
    st.dataframe(data)


filter_data = data

#Preprocessing the dataset
def preprocessing():
    #Calculates word count
    def word_counting(x):
        return (len(TextBlob(x).words))

    filter_data['word_length'] = filter_data['essay'].apply(word_counting)

    #Calculates sentence count
    def sentence_counting(x):
        sentence_len = len([len(sentence.split(' ')) for sentence in TextBlob(x).sentences])
        return sentence_len

    filter_data['no_of_sentence'] = filter_data['essay'].apply(sentence_counting)

    #Calculates sentiment of sentence
    def avg_sentence_sentiment(x):
        sentiment_essay = TextBlob(x).sentiment.polarity
        return sentiment_essay

    filter_data['sentiment_essay'] = filter_data['essay'].apply(avg_sentence_sentiment)

    #Calculates average length of words
    def avg_length_of_words(x):
        word_len = [len(word) for word in TextBlob(x).words]
        return (sum(word_len) / len(word_len))

    filter_data['avg_word_len'] = filter_data['essay'].apply(avg_length_of_words)

    #Checks the grammatical error
    def grammar_check(x):
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(x)
        return len(matches)

    filter_data['Grammar_check'] = filter_data['essay'].apply(grammar_check)

#The next 5 lines can be commented as they take some time to load. If time is not an issue, feel free to go ahead
st.write("If you wish to, Click on this button to view the features of the essay set")
click_1 = st.checkbox("Check this box")
if click_1 == True:
    preprocessing()
    st.dataframe(filter_data)

#Enter theessay to be graded here
essay_to_be_graded = st.text_area("Enter here the essay to be graded")
data = data.append({'essay': essay_to_be_graded, 'domain1_score': random.randint(2,12)}, ignore_index=True)

#Processing of essay to be graded
def processing():
    y = data['domain1_score']

    #splitting into train and test set
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

    train_e = X_train['essay'].tolist()
    test_e = X_test['essay'].tolist()

    train_sents = []
    test_sents = []

    stop_words = set(stopwords.words('english'))


    def sent2word(x):
        x = re.sub("[^A-Za-z]", " ", x)
        x.lower()
        filtered_sentence = []
        words = x.split()
        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
        return filtered_sentence

    def essay2word(essay):
        essay = essay.strip()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw = tokenizer.tokenize(essay)
        final_words = []
        for i in raw:
            if (len(i) > 0):
                final_words.append(sent2word(i))
        return final_words

    for i in train_e:
        train_sents += essay2word(i)

    for i in test_e:
        test_sents += essay2word(i)

    #Layout of the LSTM Model
    def get_model():
        model = Sequential()
        model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
        BatchNormalization()
        model.add(LSTM(64, recurrent_dropout=0.4))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
        model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['mae'])
        model.summary()
        return model

    # Training Word2Vec model
    num_features = 300
    min_word_count = 20
    num_workers = -1
    context = 10
    downsampling = 1e-3

    model = Word2Vec(train_sents,
                     workers=num_workers,
                     size=num_features,
                     min_count=min_word_count,
                     window=context,
                     sample=downsampling)

    model.init_sims(replace=True)
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)

    def makeVec(words, model, num_features):
        vec = np.zeros((num_features,), dtype="float32")
        noOfWords = 0.
        index2word_set = set(model.wv.index2word)
        for i in words:
            if i in index2word_set:
                noOfWords += 1
                vec = np.add(vec, model[i])
        vec = np.divide(vec, noOfWords)
        return vec

    def getVecs(essays, model, num_features):
        c = 0
        essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
        for i in essays:
            essay_vecs[c] = makeVec(i, model, num_features)
            c += 1
        return essay_vecs

    clean_train = []
    for i in train_e:
        clean_train.append(sent2word(i))
    training_vectors = getVecs(clean_train, model, num_features)

    clean_test = []

    for i in test_e:
        clean_test.append(sent2word(i))
    testing_vectors = getVecs(clean_test, model, num_features)

    training_vectors = np.array(training_vectors)
    testing_vectors = np.array(testing_vectors)

    # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
    training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
    testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))
    lstm_model = get_model()

    #fitting the model
    lstm_model.fit(training_vectors, y_train, batch_size=64, epochs=150)
    y_pred = lstm_model.predict(testing_vectors)
    y_pred = np.around(y_pred)
    st.write("Your score is", y_pred[8])

#button widget to calcuate score
button_two=st.button("Calculate Score")
#while click==True:
if button_two==True:
    preprocessing()
    processing()



