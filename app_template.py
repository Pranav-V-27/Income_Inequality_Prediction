# Import necessary libraries 
import streamlit as st
import pandas as pd
import numpy as np

# Add title to the app
st.title('App Template')

# Add a brief introduction for the app
st.write("""
This is a template Streamlit app that can be used as a starting point for building your own apps.
""")

# Input widgets 
# Take user inputs using widgets like text boxes, sliders, radio buttons etc.

name = st.text_input('Name')
age = st.slider('Age', 1, 100, 25)
gender = st.radio('Gender', ['Male', 'Female'])

# Processing
# Use the input variables for any processing or calculations required

if gender=='Male':
    st.write('Hello Mr.', name)
else:
    st.write('Hello Ms.', name)
    
# Display outputs
# Display the outputs in text, tables, charts, plots etc. 

st.write('Your age is:', age)

# Visualizations
# Add any charts or plots using pandas, matplotlib etc.

data = {
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}
df = pd.DataFrame(data)
st.line_chart(df)

# Run the app   
if __name__ == '__main__':
    main()
