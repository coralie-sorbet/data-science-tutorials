# import streamlit as st
import pandas as pd

# st.title('Introduction to pandas')
# st.markdown('Visualizing Titanic,Master 2 TSE, Raphaël Sourty')
# st.markdown("You can download the data here: https://www.kaggle.com/c/titanic/data/train")

df = pd.read_csv('data/train.csv') 

df.head()

