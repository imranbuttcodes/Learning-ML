import streamlit as st

st.title('This is the main Page')
if 'name' not in st.session_state:
    st.session_state.name = ""
name = st.text_input('Enter Your Name Buddy')
if st.button('Enter'):
    st.session_state.name = name
    st.write(f'You Entered {name}')

st.markdown('- checkout www.chatgpt.com')

import streamlit as st

st.title('This is the Page2 ')

st.write(f'User Entered {st.session_state.name}')

st.set_page_config(page_title='Page Gender Predictor', page_icon='😎')

with st.sidebar:
    st.header("About")
    st.write("ML Model: Random Forest")
    st.write("Accuracy: 85%")