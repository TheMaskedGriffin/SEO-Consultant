import sqlite_fix
from crewai import Crew, Process
from agents import website_analyst, competitor_analyst, competitor_researcher, report_writer
from task import analysis_task, competitor_finding_task, competitor_researching_task, writing_task
import streamlit as st
import validators
import os
from flows import SEOAnalysisFlow, SEOState

st.title("Appear before your Competitors in Search Rankings 📈")

st.sidebar.title("Settings")
llm_model = st.sidebar.selectbox("Select the model to use:", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"], index=0)
openai_api_key=st.sidebar.text_input("Enter your OpenAI API Key:",type="password")
serper_api_key=st.sidebar.text_input("Enter your Serper API Key:",type="password")

website_url = st.text_input(
        "Enter the URL of the website you want to analyze:",
        placeholder="https://www.example.com",
    )

if not (openai_api_key and serper_api_key):
    st.sidebar.warning("Please enter your OpenAI and Serper API keys to proceed.")
elif not(validators.url(website_url)):
    st.warning("Please enter a valid URL.")
else:
    st.success("Valid URL! Proceeding with analysis...")

    progress = st.progress(20, text="🔐 Validating API keys and URL...")

    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["OPENAI_MODEL_NAME"]= llm_model
    os.environ["SERPER_API_KEY"] = serper_api_key

    progress.progress(40, text="🤖 Initializing Flow...")

    # Run the flow instead of raw crew
    flow = SEOAnalysisFlow(state=SEOState(website=website_url))

    progress.progress(70, text="🚀 Running analysis... (This will take a few minutes)")
    result = flow.kickoff()

    progress.progress(100, text="✅ Analysis complete!")

    st.subheader("Analysis Result")
    st.markdown(result, unsafe_allow_html=True)
