import streamlit as st 
import numpy as np
import joblib
import pickle

model = joblib.load('model.joblib')

# Load the LabelEncoder from the file
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

st.header("Dissease Classification Project")

question = "Enter your symptoms:"

# Converting text file to array
# Read lines from a text file and convert each line into an array
symptoms_dictionary = {}
with open('array.txt', 'r') as file:
    lines = file.readlines()
    # symptoms.append(line.strip().split() for line in lines)
    for i, line in enumerate(lines): 
        lines[i] = line[:-1]
    
    lines = lines[:-1]

    # print(lines)

for i, line in enumerate(lines):
    symptoms_dictionary[line] = i

symptoms = lines

options = st.multiselect(question,symptoms)

def create_row(row, symptoms_dictionary, options):
    for option in options: 
        index = symptoms_dictionary[option]
        row[index] = 1

    print(row)


# Creating an array with 132 zeros
row = np.zeros(132)

# st.button("Check", on_click=create_row(row, symptoms_dictionary, options))

# predicts answer for our row
prediction = model.predict(row.reshape(1, -1))

# decodes from numerical to categorical
prediction = le.inverse_transform(prediction)


st.header("You have the following symptoms: ")
for option in options: 
    st.markdown("- " + option)

st.header("Diagnosis says you have: ")

# You need to input minimum 4 symptoms to get diagnosed

if len(options) <=4:
    st.write("")
else:
    st.write(prediction[0])

