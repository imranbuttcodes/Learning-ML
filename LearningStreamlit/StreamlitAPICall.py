import requests
import json
import streamlit as st



st.title("Find Your Artist Songs",text_alignment='center')

artist = st.text_input("Enter Artist Name")
response = None
t_songs = st.number_input('Enter How Many Songs You wanna Search')
if st.button('Search Songs'):
    if artist:
        with st.spinner(f'Searching for songs 🎵 ...'):
            url = f"https://itunes.apple.com/search?entity=song&limit={t_songs}&term={artist}"
            response = requests.get(url)
            data = response.json()
            st.caption(f'API Status: {response.status_code}')
            if data['resultCount'] > 0:
                st.success("Songs Found")
                all_songs = [result['trackName'] for result in data['results']]
                print(all_songs)
                with st.expander("All Songs"):
                    for song in all_songs:
                        st.markdown(f"- {song}")
            else:
                st.warning('No Results Found for this Artist')
    else:
        st.warning("Please Enter an artist name")
