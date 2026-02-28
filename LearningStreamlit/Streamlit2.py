import streamlit as st
import pandas as pd
import datetime

st.title("Learning Streamlit Basics 😎",text_alignment='center')

st.write("Wha's Up")
st.header("THis is imean butt")
st.subheader("THis is subheader")
shoice = st.selectbox("Enter Your Choice", ['Imran ', 'Butt', 'Butt2'], index= 1)
drop = st.slider('Entre Butt',1,100,10)


name = st.text_input('Enter Your Name')
if st.button('Next'):
    st.success(f"Operation Success {name}")
agreed = st.checkbox('I agree to the terms and conditions')
if agreed:    
    st.success('Brother Agreed')
rd = st.radio('Pick Your Option', options = ['Yes', 'maybe','No'], captions=[
         "Oh yeah.",
         "Uhmm hmm",
         "What the heck Bro",
], index=None)
if rd:
    st.write(f'So you selected {rd}')
num = st.number_input('Enter Your Favoroute Number Between 0-10',min_value=0,max_value=10,step=1)
st.write('So Your Selected',num)
#date = st.date_input('Enter Your Date',datetime.date(2019,7,6))
date = st.date_input('Enter Your Date',)
st.write('Your DOB is',date)

col1, col2 = st.columns(2)
with col1:
    st.header('Column 1')
    st.image('data:image/jpeg;base64,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', width=200, caption='Testing',use_container_width=False,)
    
    st.subheader('Column1 Subheader')
    age = st.slider('Enter Your Age', 1,99,25)
    st.write('Your Age',age)
with col2:
    st.header('Column 2', divider='gray') # or st.header('Column 2', divider=True) 
    st.subheader('Column 2 Subheader')
    bro_age = st.slider("Enter Your Bro's Age", 1,99,25)
    st.write("Your Bro's Age",bro_age)
name = st.sidebar.text_input('Enter Your Name')
st.sidebar.write('This is side bar')
if st.sidebar.button('Next'):
    st.success('Success')
else:
    st.error('Error')
st.sidebar.selectbox('Choose Your Mentor', ['Imran', 'Butt', 'Whome'])
with st.expander('See Explanations'):
    st.write('''
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    ''')
st.markdown("# Hello Imran's World",text_alignment='center')
st.markdown('> What should i write here?')
st.markdown('- What should i write here as well?')
st.markdown(' This is **bold** and *italic* string')
st.markdown("- Item 1\n- Item 2\n- Item3")
st.markdown("**Important Note:** Always Site Your Recources", help='This Displays Markdown text with hover tooltip')
st.markdown("**Need Help 😢?**", help='https://www.chatgpt.com')


# # Sample data
df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
    help='Bro Download bro00000000'
) 

file = open('StreamLitBasicFuncitons.txt','r', encoding='utf-8')
st.download_button('Download StreamlitInfo',data=file,
                   file_name='StreamlitInfo.txt',mime='text',help='hehe Bro 😎')

file = st.file_uploader('Upload Your File', type=['csv'])
if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)
        if st.button('See Summery'):
            st.subheader('Summery Stats')
            st.dataframe(df.describe())
        
        if st.button('Save File'):
            with open(file.name,'wb') as f:
                f.write(file.getbuffer())
            st.success('File Uploaded Successfully')
          
print(type(file))

st.download_button(
    'Download that file',
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='bro.txt',
    mime='txt/csv',
    help='www.github.com/imranbuttcodes'
)