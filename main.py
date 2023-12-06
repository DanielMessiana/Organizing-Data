import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import StringIO
import time
st.title("Data Organizer")


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

	dataframe = pd.read_csv(uploaded_file)
