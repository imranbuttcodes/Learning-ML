

import streamlit as st
import pandas as pd

# # # # Title of your app
# # st.title("🚀 My ML Journey: SVM & Beyond")

# # # # Adding a simple header
# # # st.header("SVM Model Insights")

# # # # Let's add some text
# # st.write("This app will eventually display my SVM Revision results!")

# # # # Creating a simple slider (Great for tuning ML parameters!)
# # # gamma = st.slider("Select Gamma for SVM", 0.01, 1.0, 0.1)
# # # st.write(f"The selected Gamma is: {gamma}")

# # st.text("This is the string", text_alignment='center')


# import streamlit as st

# st.title("My First Streamlit App")  # Big title

# st.write("Hello bro! Welcome to Streamlit")  # Text / output

# name = st.text_input("Enter your name")  # Input box
# if st.button("Greet me"):  # Button
#     st.write(f"Hello {name} 😎")    

# age = st.slider("Select age",0,100,25)

# st.write(f'Your Age is {age}')

# color = st.selectbox("Pick color", ["Red","Blue"])

# agree = st.checkbox("I agree")

# option = st.radio("Pick one", ["A","B"])

# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# # Example DataFrame
# df = pd.DataFrame({
#     "X": [1,2,3,4,5],
#     "Y": [10,20,25,30,40],
#     'Z': [1,2,3,34,45]
# })

# st.dataframe(df)  # Display dataframe
# st.line_chart(df)  # Simple line chart
# st.bar_chart(df)   # Simple bar chart

# Custom plot
# plt.plot(df["X"], df["Y"])
# st.pyplot(plt)

st.title("Imran Butt",text_alignment='center')
st.header('Section 1')
st.subheader('Details')
st.write('This is the details of the Imran Butt')
df = pd.DataFrame({'A':[1,2], 'B':[3,4]})
st.write("Here is my dataframe:")
st.write(df) 
#OR 
st.dataframe(df)

select = st.selectbox(
    label='Choose an option',       # Display text above dropdown
    options=['Red','Green','Blue','White'],          # List of choices
    index=0,         # Default selection (0 = first item)
)
st.write(f"You Picked this {select}")

df = pd.DataFrame({
    'dept': ['Sales', 'Engineering', 'Marketing'] * 3,
    'salary': [50000, 70000, 60000, 55000, 75000, 65000, 52000, 72000, 62000]
})

st.dataframe(df)
dept = st.selectbox('Filter By department', df['dept'].unique(), index=0)

st.dataframe(df[df['dept'] == dept])

age = st.slider('Select Age',0,100,25)


#Parameters:

#label → text

#min_value, max_value → range

#value → default selected


st.write(f'Your Age is',age)

fs = st.number_input('Family Size', 0,10,1)
if st.button('Next'):
    st.write('Alright so its',fs)
# label → text

# min_value, max_value → range

# value → default

selected = st.multiselect('Select One', options=['A', 'B', 'C', 'D', 'E'])
if st.button('Select'):
    st.write(f'You Selected {selected}')
print(type(selected))

