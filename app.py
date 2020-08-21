import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
import pickle
iris = datasets.load_iris()


st.write('''
# Simple Iris Flower Prediction App
''')

st.sidebar.header('User Input')

def input_params():
	sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
	sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
	petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
	petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

	data = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}

	inputs = pd.DataFrame(data, index=[0])

	return inputs

df = input_params()

st.subheader('User Inputs')
st.write(df)

pickle_in = open("kmeans.pkl", "rb")
kmeans = pickle.load(pickle_in)

preds = kmeans.predict(df)

st.subheader('Prediction')

# df2 = pd.DataFrame(index=[0])
# df2.columns = df.columns

l = ['setosa', 'versicolor', 'virginica']

st.write(iris.target_names[preds])

#df2 = pd.DataFrame(columns={iris.target_names},index[0])