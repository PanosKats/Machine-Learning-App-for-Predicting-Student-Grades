import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



st.title('Machine Learning App for Predicting Student Grades')

st.info('This app builds a machine learning model designed to predict a student\'s final grade based on various input features.')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/PanosKats/Datasets/refs/heads/main/grades_for_streamlit.csv')
  df

  st.write('**X**')
  X_raw = df.drop('GradeClass', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.GradeClass
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='StudyTimeWeekly', y='Absences', color='GradeClass')

# The sidebar for the input features
with st.sidebar:
  st.header('Input features')
  gender = st.selectbox('Gender', ('male', 'female'))
  ParentalEducation = st.selectbox('Parental Education', ('No Education', 'High School', 'Some College', 'Bachelor', 'Higher'))
  ParentalSupport = st.selectbox('Parental Support', ('None', 'Low', 'Moderate', 'High', 'Very High'))
  Tutoring = st.selectbox('Tutoring', ('Yes', 'No'))
  Extracurricular = st.selectbox('Extracurricular', ('Yes', 'No'))
  StudyTimeWeekly = st.slider('Study Time Weekly', 0.001057, 19.978094, 9.771992)
  Absences = st.slider('Absences', 0.0, 29.0, 14.54)
  GPA = st.slider('GPA', 0.1, 4.0, 1.90)
  Age = st.slider('Age', 15.0, 18.0, 16.46)
  
  
  # The DataFrame for the input features
  data = {'Gender': gender,
          'ParentalEducation': ParentalEducation,
          'ParentalSupport': ParentalSupport,
          'Tutoring': Tutoring,
          'Extracurricular': Extracurricular,
          'StudyTimeWeekly': StudyTimeWeekly,
          'Absences': Absences,
          'GPA': GPA,
          'Age': Age}
  input_df = pd.DataFrame(data, index=[0])
  inputs = pd.concat([input_df, X_raw], axis=0)



# Data preparation
# Encode X
encode = ['gender', 'ParentalEducation','ParentalSupport','Tutoring','Extracurricular']
df_grades = pd.get_dummies(inputs, prefix=encode)

X = df_grades[1:]
input_row = df_grades[:1]

# Encode y
target_mapper = {'A': 0,
                 'B': 1,
                 'C': 2,
                 'D': 3,
                 'E': 4}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input parameters)**')
  input_row
  st.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['A', 'B', 'C',  'D',  'E']
df_prediction_proba.rename(columns={0: 'A',
                                 1: 'B',
                                 2: 'C',
                                 3: 'D',
                                 4: 'E'})

# Display predicted Grade
st.subheader('Predicted Grade')
st.dataframe(df_prediction_proba,
             column_config={
               'A': st.column_config.ProgressColumn(
                 'A',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'B': st.column_config.ProgressColumn(
                 'B',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'C': st.column_config.ProgressColumn(
                 'C',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'D': st.column_config.ProgressColumn(
                 'D',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'E': st.column_config.ProgressColumn(
                 'E',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


Grade = np.array(['A', 'B','C','D', 'E'])
st.success(str(Grade[prediction][0]))
