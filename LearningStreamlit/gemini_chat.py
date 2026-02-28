# import streamlit as st
# import google.generativeai as genai
# import os

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

import google.generativeai as genai
import os

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if not os.getenv("GEMINI_API_KEY"):
    print("Please set GEMINI_API_KEY environment variable!")
    exit(1)

# Create a chat model (supports conversation history)
model = genai.GenerativeModel('gemini-2.5-flash')  # or 'gemini-1.5-pro'

# Start a new chat session
chat = model.start_chat(history=[])

print("🤖 Gemini Chat - Type 'quit' to exit")
print("-" * 40)

while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("👋 Goodbye!")
        break
    
    if not user_input:
        continue
    
    print("🤖 Gemini: ", end="", flush=True)
    
    try:
        # Send message and get response
        response = chat.send_message(user_input)
        
        # Print response
        print(response.text)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Try again!")
