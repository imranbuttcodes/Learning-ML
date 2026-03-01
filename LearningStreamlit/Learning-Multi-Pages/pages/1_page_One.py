import streamlit as st

st.title('This is the Page1')

st.set_page_config(page_title='Gender Predictor', page_icon='🧞‍♂️')



with st.sidebar:
    st.header("About")
    st.write("ML Model: SVM")
    st.write("Accuracy: 83%")