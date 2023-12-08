import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import StringIO
import time


st.sidebar.title("Models for Data")

options = ['Home', 'Linear Regression']
side_menu = st.sidebar.radio("Side menu widgets", options, label_visibility="hidden")

#if side_menu

if side_menu == 'Home':
	st.title("Data Organizer and Analyzer")
	st.divider()
	st.header("Input data for analysis")

	uploaded_file = st.file_uploader("Choose a file")

	if uploaded_file is not None:

		dataframe = pd.read_csv(uploaded_file)
		st.session_state['data'] = dataframe	
		dataframe 

if side_menu == 'Linear Regression':
	st.title('Linear Regression')

	if 'data' not in st.session_state:
		"Couldn't find data, please try again."
	else:
		data = st.session_state['data']
		features = list(data.columns)

		x = data[st.selectbox('Set x label', features)]
		y = data[st.selectbox('Set y label', features)]

		ones = np.ones(data.shape[0])
		X = np.vstack([[x], [ones]]).T
		
		beta_hat = np.linalg.pinv(X.T @ X + 0.001 * np.eye(X.shape[1])) @ X.T @ y.T

		y_hat = np.dot(X, beta_hat)

		plt.plot(x, y, 'ok')
		plt.plot(x, y_hat)
		st.pyplot(plt)