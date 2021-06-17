#Loading the required libraries
#------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#Remove Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Shareena_ML_B7_project_SpineDataset")

#import dataset
df = pd.read_csv("Dataset_spine.csv")
df.rename(columns = {"Col1" : "pelvic_incidence", "Col2" : "pelvic_tilt","Col3" : "lumbar_lordosis_angle",
                     "Col4" : "sacral_slope", "Col5" : "pelvic_radius","Col6" : "degree_spondylolisthesis", 
                     "Col7" : "pelvic_slope","Col8" : "direct_tilt","Col9" : "thoracic_slope",
                     "Col10" :"cervical_tilt", "Col11" : "sacrum_angle","Col12" : "scoliosis_slope", 
                     "Class_att" : "Spine_Condition"}, inplace=True)

#First 5 rows
df_5= df.head()
#Display the table
st.table(df_5)

#Standard scaler
X=df[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis',
 'pelvic_slope','direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)
scaled_df = pd.DataFrame(data = scaled_data, columns = X.columns)
y=df['Spine_Condition']
df=scaled_df.join(y)
df.head(2)

#Visualisation Using Seaborn
#------------------------------
st.header("Visualisation Using Seaborn")
#bar plot
st.subheader(" Bar Plot")
st.subheader("Frequency Distribution of target Spine condition")
sns.countplot(x='Spine_Condition', data=df,hue='Spine_Condition', palette= 'Set1')
st.pyplot()

st.subheader("Hisplot")
st.subheader("Frequency Distribution of numeric coulumns ")
df.plot(kind='hist', subplots=True, layout=(4,3), sharex=False ,figsize=(15,12),title = "Features Distribution")
st.pyplot()

#Correation
st.subheader("Heatmap")
sns.heatmap(df.corr(),cmap="viridis", annot=True , linewidths=.5, fmt= '.1f')
st.pyplot()


#Calling the model we saved above:
#------------------------------
pickle_in = open('Shareena_ML_B7_project_SpineDataset.pkl', 'rb')
classifier = pickle.load(pickle_in)

#Creating the UI for the application:
#------------------------------------
st.sidebar.header('Spine Condition Prediction')

pelvic_incidence = st.text_input("Enter the pelvic_incidence ")
pelvic_tilt = st.text_input("Enter the pelvic_tilt ")
lumbar_lordosis_angle =  st.text_input("Enter the lumbar_lordosis_angle ")
sacral_slope = st.text_input("Enter the sacral_slope ")
degree_spondylolisthesis = st.text_input("Enter the degree_spondylolisthesis ")
submit = st.button("Classify")

if submit:
        result = classifier.predict([[pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,
                                      sacral_slope,degree_spondylolisthesis]])
        if result ==0:
            st.write("Spine condition is Abnormal")
        else:
            st.write("Spine condition is Normal")                
#st.write(result)

