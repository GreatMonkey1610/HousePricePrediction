from PIL import Image
import pickle
import pandas as pd
import numpy as np
import sklearn
import math
import streamlit as st
model = pickle.load(open("housepricepredictionmodel.pkl",'rb'))

df = pd.read_csv("GreatMonkey1610/HousePricePrediction/kc_house_data.csv")
original_title = '<p style="font-family:w3-cursive;background-color:hsl(3, 100%, 64%); color:LightGray; font-size: 40px;">House Price Prediction by ML</p>'
# Title
#st.title("")
st.markdown(original_title, unsafe_allow_html=True)
# Text
st.write("""House Price Prediction, ya it sounds excellent, let's know what's is it about. It's 
a simple machine learning project that uses linear regression algorithm to predict
the house price. It is based on cost and sale price comparison. This model 
will be a great help for real estate marketers and common people like us 
searching for houses.""")
#video_file = open("C:\\Users\\Irfana\\Downloads\\Up_Pixar_Flying_House_Scene.mp4", 'rb')
#video_bytes = video_file.read()
#st.video(video_bytes)
uphouse = "C:\\Users\\Irfana\\Downloads\\Up_Pixar_Flying_House_Scene.gif.mp4"
st.markdown('<img src="./C:\\Users\\Irfana\\Downloads\\Up_Pixar_Flying_House_Scene.gif.mp4"/>', unsafe_allow_html=True)
img = Image.open("C:\\Users\\Irfana\\PycharmProjects\\HousePricePredition\\salvatore.png")
st.image(img, use_column_width ='auto')
st.text("This is the database we used to create the model")
if st.checkbox('Just click the box to view the data'):
    st.text("Oh you clicked the check box.. i see you are interested 🦊")
    st.dataframe(df)
#if (st.button('Try this yourself by clicking me')):
st.selectbox("Select your model: ",
             ['Linear Regression'])

bedrooms = st.text_input("BedRooms")
bathrooms = st.text_input("BathRooms")
sqft_living = st.text_input("sqft of living")
grade = st.text_input("Rating")
floors = st.text_input("floors")
features_dict = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'grade': grade,
    'floors': floors
}
features = features_dict
features_df = pd.DataFrame([features])

# st.table(features_df)
st.markdown("Predicted house price")
if (st.button('Submit')):
    PredictedPrice = model.predict(features_df)
    predicted_price = int(PredictedPrice)
    st.success(f" Predicted cost of the house is {predicted_price}")


st.write("Done by:  \n"
   "Mahendra.R  \n"
   "Syed Barakkath Irfana. Z  \n")
    # Selection box


