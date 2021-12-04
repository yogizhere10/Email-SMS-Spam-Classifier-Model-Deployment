import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter Your Message")

if st.button('Predict'):
    # 1 Preprocess
    trans_sms = transform_text(input_sms)

    # 2 Vectorizer
    vector = tfidf.transform([trans_sms])

    # 3 Predict
    result = model.predict(vector)[0]

    # 4 Display
    if result == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")




