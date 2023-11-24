import langchain_helper as lch
import streamlit as st
import textwrap


st.title("YouTube Assistant(Question Answering app based on youtube video)")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="Paste the youtube video URL",
            max_chars=50
        )

        query = st.sidebar.text_area(
            label="Ask me anything based on the video",
            max_chars=50,
            key="query"
        )

        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        with st.sidebar:
            submit_btn = st.form_submit_button(label="Ask")


if query and youtube_url and submit_btn:
    with st.spinner("Wait while I query the video..."):
        db = lch.create_vector_db_from_ytUrl(youtube_url)
        response, docs = lch.get_response_from_query(db, query)

        st.subheader(f"You asked: {query}. Here is what I got:")
        st.text(textwrap.fill(response, width=80))
