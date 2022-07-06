import streamlit as st

import about
import actors


# Page settings
st.set_page_config(page_title="GDELT Actors Explorer")#, page_icon="❤️")

# Sidebar settings

PAGES = {
    'About this App': about,
    'Actor Types Explorer': actors,
}


image = 'https://blog.gdeltproject.org/wp-content/uploads/2015-gdelt-2.png'
st.sidebar.image(image, use_column_width=True)


st.sidebar.title("GDELT News Explorer")

page = st.sidebar.radio("Navigation", list(PAGES.keys()))

# Display the selected page in the main viewport
PAGES[page].show_page()

# Made by section - footer in the sidebar
st.sidebar.markdown('''
### Made with ❤️ by:
 - [Jovan Veljanoski](https://www.linkedin.com/in/jovanvel/)
 - [Maarten Breddels](https://www.linkedin.com/in/maartenbreddels/)
''')

st.sidebar.image('./logos/logo-white.png', use_column_width=True)

# END OF SCRIP
