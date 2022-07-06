import streamlit as st

from actor_codes import actor_codes


def show_page():

    # About the GDELT Project
    with st.expander("About this App"):
        st.markdown("""
        The GDELT Project monitors the world's broadcast, print and web news around
        the globe in over 100 languages. It identifies people, locations, organizations,
        themes, emotions, events and other entities that continuously drive our society.

        Given its global reach, the GDELT Project is a unique resource which allows us
        to gain a global perspective on what is happening and how the world feels about it.
        The GDELT archives are constantly updated every 15 minutes. The data is
        fully public and is available via Google BigQuery, and via GDELT's own FTP servers.

        Learn more about the GDELT Project [here](https://gdeltproject.org/about/).
        """)

    # About the data we are using
    with st.expander("The Events dataset"):
        st.markdown("""
        This app uses the GDELT 2.0 Events dataset. It is a collection of worldwide
        activities, i.e. events, in over 300 categories such as diplomatic exchanges,
        natural disasters, or any other significant event of relevance. Each event
        record comprises over 50 fields capturing many different aspects of the event.
        The dataset is continuously updated every 15 minutes. The snapshot used in this app
        ranges from February 2014 until April 2022, and comprises a little over 625 million events.

        Learn more about this dataset [here](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/).
        """)

    # What are Actors?
    with st.expander("What are Actors?"):
        st.markdown("""
        Actors are the entities that are involved in the events captured by the GDELT dataset.
        Actors can be people, organizations, countries, or any other entities or objects that
        were the main topic of the news repots. The GDELT 2.0 Events dataset contains various
        attributes related to the actors.

        This app focuses on the actor type attributes, which
        are 3-character codes that describe the "type" or "role" of the actor. Example of such
        types or roles are "Police Force", "Government", "Non-governmental Organizations".

        Learn more about the actor attributes [here](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf).
        """)

    # Actor Code Types definition
    with st.expander("Actor code types definitions"):
        code_definitions_md = ''
        for code in actor_codes.keys():
            code_definitions_md += f" - {code}: {actor_codes[code]}\n"
        st.markdown(code_definitions_md)

    # How to use the app
    with st.expander("How to use the app"):
        st.markdown("""
        With this app you can explore different actor types, and see how the asscoiated news impact the world.
        Simply choose one or more types in the side panel on the left, and you will be presented with:
         - General statistics about the news articles related to these actor types
         - Trends of the [GoldStein scale](http://web.pdx.edu/~kinsella/jgscale.html) and the average tone (sentiment) of the selected articles
         - Map of the world showing showing where the actors are coming from and how the news impact each country
         - A word-cloud of the 300 most common actor names.

        Optionally you can also constrain the time period for which you are interested in.
        """)


    # About Vaex
    with st.expander("About Vaex"):
        st.markdown("""
        Vaex is high performance DataFrame library in Python that allows for fast processing on very large
        datasets on a single node. With its efficient algorithms, Vaex can go over *a billion* samples per second.
        Using memory mapping techniques, Vaex can work with datasets that are much larger then RAM.

        This makes Vaex a perfect backend choice for a variety of dashboards and data applications.

        Vaex is open source and available on [Github](https://github.com/vaexio/vaex/).
        A variety of relevant resources can be found [here](https://vaex.io/).
        """)
