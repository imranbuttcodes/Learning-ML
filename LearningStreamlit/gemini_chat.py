import streamlit as st
import google.generativeai as genai
import os

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("gemini-2.5-flash")


# if "messages" not in st.session_state:
#     st.session_state.messages = []


# st.title("Gemini AI Chat 💬")

# user_input = st.text_input("Type your message here:")

# if st.button("Send"):
#     if user_input:
#         st.session_state.messages.append({"role": "user", "content": user_input})
 
#         response = model.generate_content(user_input)
   
#         st.session_state.messages.append({"role": "assistant", "content": response.text})

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

if 'messages' not in st.session_state:
    st.session_state.messages = []

model = genai.GenerativeModel("gemini-2.5-flash")

st.title('Gemini What can i help you?')
prompt = st.text_input('Ask Anything')
if st.button('Ask'):
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })
    response = model.generate_content(prompt)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response.text
    })

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
# with st.chat_message('user'):
#     st.markdown('> Imran butt is the beast')

