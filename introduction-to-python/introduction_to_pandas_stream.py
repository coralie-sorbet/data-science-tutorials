import streamlit as st
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Introduction to pandas')
st.markdown('Visualizing Titanic, Master 2 TSE')
st.markdown('Raphaël Sourty')
st.markdown("You can download the data here: https://www.kaggle.com/c/titanic/data/train")

df = pd.read_csv('data/train.csv') 
st.write(df.head())

st.markdown('Display number of null values')
st.write(df.isnull().sum())

st.markdown("Who are the survivors of the Titanic?")
st.markdown("Survival rate")

survival_count = df.groupby('Survived')['PassengerId'].agg(['count'])
st.write(survival_count)

fig, ax = plt.subplots(figsize=(10, 10))
survival_count.reset_index().plot(x='Survived', y='count', kind='bar', ax=ax)
st.pyplot(fig)

st.markdown('Survival rate depending on genre')

survival_by_sex = df.groupby(['Survived', 'Sex'])['PassengerId'].agg(['count']).unstack()
st.write(survival_by_sex)

fig, ax = plt.subplots(figsize=(10, 10))
survival_by_sex.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.markdown('Survival rate depending on ticket class')

survival_by_class = df.groupby(['Survived', 'Pclass'])['PassengerId'].agg(['count']).unstack()
st.write(survival_by_class)

fig, ax = plt.subplots(figsize=(10, 10))
survival_by_class.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Survival rate depending on age
st.markdown("Survival rate depending on age")
df['generation'] = pd.cut(df['Age'], 8)
survival_age = df.groupby(['Survived', 'generation'])['PassengerId'].count().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
survival_age.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Survival rate depending on fare
st.markdown("Survival rate depending on fare")
df['fare_category'] = pd.cut(df['Fare'], 12)
survival_fare = df.groupby(['Survived', 'fare_category'])['PassengerId'].count().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
survival_fare.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Correlation matrix
st.markdown("What about correlations?")
corr_matrix = df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].corr()
st.write(corr_matrix)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Best side on the boat to survive
st.markdown('Best side on the boat to survive')

# Extract cabin number and manage missing values
df['Cabin_number'] = df['Cabin'].str.extract('(\d+)')
df['Cabin_number'] = df['Cabin_number'].fillna(-1).astype('int')
df['parity'] = df['Cabin_number'].apply(lambda x: x % 2 if x != -1 else -1)
df = df[df['parity'] != -1]

parity_survival = df.groupby(['Survived', 'parity'])['PassengerId'].agg(['mean']).unstack()

fig, ax = plt.subplots(figsize=(10, 10))
parity_survival.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Best deck to be on
st.markdown('Best deck to be on')
df['Cabin_letter'] = df['Cabin'].str.extract('([A-Za-z])')
df = df[df['Cabin_letter'].notnull()]

deck_survival = df.groupby(['Survived', 'Cabin_letter'])['PassengerId'].agg('mean').unstack()

fig, ax = plt.subplots(figsize=(10, 10))
deck_survival.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Link between number of family members on the boat and survival
st.markdown('Link between the number of parents/family on the boat and chances of survival')

sibsp_survival = df.groupby(['Survived', 'SibSp'])['PassengerId'].count().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
sibsp_survival.plot(kind='bar', ax=ax)
st.pyplot(fig)

parch_survival = df.groupby(['Survived', 'Parch'])['PassengerId'].count().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
parch_survival.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Typical profile of the person who will survive
st.markdown('Typical profile of the person who will survive the shipwreck')
profile_survival = df.groupby(['SibSp', 'Parch'])['Survived'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
profile_survival.plot(kind='bar', ax=ax)
st.pyplot(fig)