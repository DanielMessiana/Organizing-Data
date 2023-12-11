import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing 
import time

st.sidebar.title("Models for Data")

options = ['Home', 'Linear Regression', 'Logistic Regression', 'Kmeans']
side_menu = st.sidebar.radio("Side menu widgets", options, label_visibility="hidden")

# Home Page

if side_menu == 'Home':
	st.title("Data Organizer and Analyzer")
	st.divider()
	st.header("Input data for analysis")

	uploaded_file = st.file_uploader("Choose a file")

	if uploaded_file is not None:

		dataframe = pd.read_csv(uploaded_file)


		label_encoder = preprocessing.LabelEncoder()
		st.header("Unique String Values")
		for column in dataframe.columns:
			
			if type(pd.unique(dataframe[column])[0]) == str:
				f"{column.capitalize()} has {pd.unique(dataframe[column]).shape[0]}."
				f"Classes (strings): {pd.unique(dataframe[column])}"
			dataframe[column]= label_encoder.fit_transform(dataframe[column]) 
		
		st.session_state['data'] = dataframe	
	if 'data' not in st.session_state:
		"Couldn't find data, please try again."
	else:
		st.session_state['data']

	st.divider()
	st.write("This program is useful for analyzing and organizing data in one place.")

# Linear Regression Page

if side_menu == 'Linear Regression':
	st.title('Linear Regression')

	if 'data' not in st.session_state:
		"Couldn't find data, please upload a dataset."
	else:
		data = st.session_state['data']
		features = list(data.columns)

		x_label = st.selectbox('Set x label', features)
		y_label = st.selectbox('Set y label', features)
		x = data[x_label]
		y = data[y_label]

		ones = np.ones(data.shape[0])
		X = np.vstack([[x], [ones]]).T
		
		beta_hat = np.linalg.pinv(X.T @ X + 0.001 * np.eye(X.shape[1])) @ X.T @ y.T
		y_hat = np.dot(X, beta_hat)

		plt.plot(x, y, 'ok', alpha=0.6)
		plt.plot(x, y_hat)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		st.pyplot(plt)

if side_menu == 'Logistic Regression':
	st.title('Logistic Regression')

	if 'data' not in st.session_state:
		"Couldn't find data, please upload a dataset."
	else:
		data = st.session_state['data']
		features = list(data.columns)

		x_label = st.selectbox('Set x label', features)
		y_label = st.selectbox('Set y label', features)
		x = data[x_label]
		y = data[y_label].astype(float)

		ones = np.ones(data.shape[0])
		X = np.vstack([x, ones]).T

		beta = np.zeros(X.shape[1])

		def sig(x):
			return 1 / (1 + np.exp(-x))

		def cost_function(X, y, beta):
			z = np.dot(X, beta)
			p = sig(z)
			return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

		def gradient(X, y, beta):
			z = np.dot(X, beta)
			p = sig(z)
			return np.dot(X.T, (p - y)) / len(y)

		learning_rate = 0.01
		num_iterations = 1000

		for i in range(num_iterations):
			beta -= learning_rate * gradient(X, y, beta)

		probabilities = sig(np.dot(X, beta))


		sorted_indices = np.argsort(x)
		x_sorted = x.iloc[sorted_indices]
		probabilities_sorted = probabilities[sorted_indices]

		plt.scatter(x, y, color='black', marker='o', label='Actual')
		plt.plot(x_sorted, probabilities_sorted, color='blue', label='Predicted Probabilities')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.legend()
		st.pyplot(plt)

# Kmeans Class

class kmeans:
    """ The k-Means algorithm"""
    
    def __init__(self, k, data):
        """Initially assign each sample to a random cluster.
           Assign each centroid to be a zero vector.
        """
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.k = k
        self.data = np.hstack((np.random.randint(0, self.k, (self.nData, 1)), data))
        self.centroids = np.zeros((self.k, self.nDim))
        self.make_centroids()
        self.wcv = np.zeros(self.k)

    def make_centroids(self):
        """Calculate centroids of current clusters"""
        
        for i in range(self.k):
            cluster = self.data[self.data[:, 0]==i]
            cluster = cluster[:, 1:]
            self.centroids[i] = np.mean(cluster, axis=0)

    def make_clusters(self):
        distances = np.zeros((self.nData, self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((self.data[:, 1:] - self.centroids[i,:])**2,axis=1)
        self.data[:,0] = np.argmin(distances, axis=1)
        
    def within_cluster_variation(self):
        """Calculate within-cluster variation of current clusters"""
        
        for i in range(self.k):
            cluster = self.data[self.data[:, 0]==i]
            cluster = cluster[:, 1:]
            distances = euclidean_distances(cluster, cluster)
            self.wcv[i] = np.sum(distances)/(2*cluster.shape[0])
        
    def cluster(self, iterations=1):
        """Perform k-Means"""
        for i in range(iterations):
            self.make_clusters()
            self.make_centroids()
        self.within_cluster_variation()
        labels = self.data[:,0]
        return labels, self.centroids

# Kmeans Page

if side_menu == 'Kmeans':
	st.title('Kmeans Clustering')

	if 'data' not in st.session_state:
		"Couldn't find data, please upload a dataset."
	else:
		data = st.session_state['data']
		features = list(data.columns)

		x_label = st.selectbox('Set x label', features)
		y_label = st.selectbox('Set y label', features)
		x = data[x_label]
		y = data[y_label]
		
		x_u = pd.unique(x)
		y_u = pd.unique(y)
		x_u.shape[0]
		
		features = np.vstack([[x], [y]]).T

		c = st.number_input("Number of Classes", value=3)

		km = kmeans(c, features)
		labels, centroids = km.cluster(8)
		c0 = data[labels==0]
		c1 = data[labels==1]
		c2 = data[labels==2]

		plt.plot(c0[x_label], c0[y_label], 'ob', alpha=0.2)
		plt.plot(c1[x_label], c1[y_label], 'or', alpha=0.2)
		plt.plot(c2[x_label], c2[y_label], 'og', alpha=0.2)
		plt.plot(centroids[0,0], centroids[0,1], 'ob')
		plt.plot(centroids[1,0], centroids[1,1], 'or')
		plt.plot(centroids[2,0], centroids[2,1], 'og')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		st.pyplot(plt)

if side_menu == 'Neural Network':
	st.title('Neural Network')

	if 'data' not in st.session_state:
		"Couldn't find data, please upload a dataset."
	else:
		data = st.session_state['data']
		features = list(data.columns)













