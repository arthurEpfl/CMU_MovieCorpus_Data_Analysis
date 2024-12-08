import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Welcome to Cinematic Moral Dilemmas !")
st.write("Are searching for the most profitable plot structrue ? Haha, great, let's go!")

# Line chart
st.header("Line Chart")
st.line_chart(df)

# Bar chart
st.header("Bar Chart")
st.bar_chart(df)

# Matplotlib figure
st.header("Matplotlib Figure")
fig, ax = plt.subplots()
sns.heatmap(df, annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Interactive widgets
st.header("Interactive Widgets")
if st.button("Click me!"):
    st.write("Button clicked!")

slider_value = st.slider("Select a value", 0, 100, 50)
st.write(f"Slider value: {slider_value}")

# File uploader
st.header("File Uploader")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

# Text input
st.header("Text Input")
text_input = st.text_input("Enter some text")
st.write(f"You entered: {text_input}")

# Selectbox
st.header("Selectbox")
option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {option}")

# Checkbox
st.header("Checkbox")
if st.checkbox("Check me"):
    st.write("Checkbox checked!")

# Radio buttons
st.header("Radio Buttons")
radio_option = st.radio("Choose an option", ["Option A", "Option B", "Option C"])
st.write(f"You selected: {radio_option}")

# Date input
st.header("Date Input")
date = st.date_input("Select a date")
st.write(f"Selected date: {date}")

# Time input
st.header("Time Input")
time = st.time_input("Select a time")
st.write(f"Selected time: {time}")

# Color picker
st.header("Color Picker")
color = st.color_picker("Pick a color")
st.write(f"Selected color: {color}")

# To run : streamlit run streamlit_website.py in terminal