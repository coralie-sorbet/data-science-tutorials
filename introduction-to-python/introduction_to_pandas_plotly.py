import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Survival count
survival_count = df.groupby('Survived')['PassengerId'].agg(['count'])
st.write(survival_count)

# Plot survival rate
fig = px.bar(survival_count.reset_index(), x='Survived', y='count', title="Survival Rate")
st.plotly_chart(fig)

st.markdown('Survival rate depending on genre')


# Survival rate by sex
survival_by_sex = df.groupby(['Survived', 'Sex'])['PassengerId'].agg(['count']).unstack()
st.write(survival_by_sex)

# Reformater les données pour les rendre 1D
survival_by_sex_reset = survival_by_sex.reset_index()
survival_by_sex_reset.columns = ['Survived', 'Female', 'Male']

# Plot survival rate by sex avec données reformées
fig = px.bar(survival_by_sex_reset.melt(id_vars='Survived', value_vars=['Female', 'Male']),
             x='Survived', y='value', color='variable', barmode='group', 
             title="Survival Rate by Gender")
st.plotly_chart(fig)

st.markdown('Survival rate depending on ticket class')


# Survival rate by class
survival_by_class = df.groupby(['Survived', 'Pclass'])['PassengerId'].agg(['count']).unstack()
st.write(survival_by_class)

# Reformater les données pour les rendre 1D
survival_by_class_reset = survival_by_class.reset_index()
survival_by_class_reset.columns = ['Survived', 'Class_1', 'Class_2', 'Class_3']

# Plot survival rate by class avec données reformées
fig = px.bar(survival_by_class_reset.melt(id_vars='Survived', value_vars=['Class_1', 'Class_2', 'Class_3']),
             x='Survived', y='value', color='variable', barmode='group', 
             title="Survival Rate by Class")
st.plotly_chart(fig)

# Survival rate depending on age
st.markdown("Survival rate depending on age")
df['generation'] = pd.cut(df['Age'], 8)
survival_age = df.groupby(['Survived', 'generation'])['PassengerId'].count().unstack()

# Survival rate by age
st.markdown("Survival rate depending on age")
df['generation'] = pd.cut(df['Age'], 8)
survival_age = df.groupby(['Survived', 'generation'])['PassengerId'].count().unstack()

# Reformater les données pour les rendre utilisables par Plotly
survival_age_reset = survival_age.reset_index()
survival_age_reset = survival_age_reset.melt(id_vars='Survived', var_name='Age Group', value_name='Count')

# Plot survival rate by age group
fig = px.bar(survival_age_reset, x='Survived', y='Count', color='Age Group', barmode='group', 
             title="Survival Rate by Age Group")
st.plotly_chart(fig)


# Survival rate by fare category
st.markdown("Survival rate depending on fare category")
df['fare_category'] = pd.qcut(df['Fare'], 6)
survival_fare = df.groupby(['Survived', 'fare_category'])['PassengerId'].count().unstack()

# Reformater les données pour les rendre utilisables par Plotly
survival_fare_reset = survival_fare.reset_index()
survival_fare_reset = survival_fare_reset.melt(id_vars='Survived', var_name='Fare Category', value_name='Count')

# Plot survival rate by fare category
fig = px.bar(survival_fare_reset, x='Survived', y='Count', color='Fare Category', barmode='group', 
             title="Survival Rate by Fare Category")
st.plotly_chart(fig)

# Correlation matrix
st.markdown("What about correlations?")
corr_matrix = df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].corr()
st.write(corr_matrix)

# Plot correlation matrix
fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, 
                                x=corr_matrix.columns, 
                                y=corr_matrix.columns,
                                colorscale='RdBu'))  # Remplace 'coolwarm' par une palette reconnue comme 'RdBu'
fig.update_layout(title="Correlation Matrix")
st.plotly_chart(fig)

# Best side on the boat to survive
st.markdown('Best side on the boat to survive')

# Extract cabin number and manage missing values
df['Cabin_number'] = df['Cabin'].str.extract('(\d+)')
df['Cabin_number'] = df['Cabin_number'].fillna(-1).astype('int')
df['parity'] = df['Cabin_number'].apply(lambda x: x % 2 if x != -1 else -1)
df_parity = df[df['parity'] != -1]

# Group by Survived and parity, and calculate the count
parity_survival = df_parity.groupby(['Survived', 'parity'])['PassengerId'].count().unstack()

# Plot survival rate by parity
fig = px.bar(parity_survival.reset_index(), x='Survived', y=[0, 1], barmode='group', title="Survival Rate by Cabin Parity",
             labels={0: 'Even Parity', 1: 'Odd Parity'})
st.plotly_chart(fig)

# Best deck to be on
st.markdown('Best deck to be on')
df['Cabin_letter'] = df['Cabin'].str.extract('([A-Za-z])')
df_deck = df[df['Cabin_letter'].notnull()]

# Group by Survived and Cabin_letter, and calculate the mean
deck_survival = df_deck.groupby(['Survived', 'Cabin_letter'])['PassengerId'].count().unstack()

# Plot survival rate by deck
fig = px.bar(deck_survival.reset_index(), x='Survived', y=deck_survival.columns, barmode='group', title="Survival Rate by Deck")
st.plotly_chart(fig)

# Link between number of family members on the boat and survival
st.markdown('Link between the number of parents/family on the boat and chances of survival')

# Group by Survived and SibSp, and calculate the count
sibsp_survival = df.groupby(['Survived', 'SibSp'])['PassengerId'].count().unstack()

# Plot survival rate by siblings/spouses
fig = px.bar(sibsp_survival.reset_index(), x='Survived', y=sibsp_survival.columns, barmode='group', title="Survival Rate by Siblings/Spouses")
st.plotly_chart(fig)

# Group by Survived and Parch, and calculate the count
parch_survival = df.groupby(['Survived', 'Parch'])['PassengerId'].count().unstack()

# Plot survival rate by parents/children
fig = px.bar(parch_survival.reset_index(), x='Survived', y=parch_survival.columns, barmode='group', title="Survival Rate by Parents/Children")
st.plotly_chart(fig)

# Typical profile of the person who will survive
st.markdown('Typical profile of the person who will survive the shipwreck')

# Group by SibSp and Parch, and calculate the mean survival rate
profile_survival = df.groupby(['SibSp', 'Parch'])['Survived'].mean().unstack()

# Plot survival profile
fig = px.bar(profile_survival.reset_index(), x='SibSp', y=profile_survival.columns, title="Typical Survival Profile")
st.plotly_chart(fig)
