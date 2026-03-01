import streamlit as st

st.title('This is the Page2 ')

st.write(f'User Entered {st.session_state.name}')

st.set_page_config(page_title='Page 2Gender Predictor', page_icon='🧞👋')



with st.sidebar:
    st.header("About")
    st.write("ML Model: Linear Regression")
    st.write("Accuracy: 90%")