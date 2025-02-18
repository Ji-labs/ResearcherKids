import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from crewai import Agent, Task, Crew
from langchain.tools import SerperDevTool
from langchain.agents import Tool
from PIL import Image
import streamlit as st
import graphviz
import io

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment variables")
    st.stop()
if not os.getenv("SERPER_API_KEY"):
    st.error("SERPER_API_KEY not found in environment variables")
    st.stop()

# Initialize OpenAI and tools
llm = OpenAI()
search_tool = SerperDevTool()

# Create agents
researcher = Agent(
    role='Research Expert',
    goal='Find accurate and age-appropriate information on given topics',
    backstory='Expert at finding reliable information for children and teens',
    tools=[search_tool],
    llm=llm  # Add OpenAI LLM
)

content_writer = Agent(
    role='Content Writer',
    goal='Write engaging and educational content for young audiences',
    backstory='Experienced in writing for children aged 8-16',
    tools=[search_tool],
    llm=llm  # Add OpenAI LLM
)

storyteller = Agent(
    role='Visual Storyteller',
    goal='Create engaging visual explanations of research topics',
    backstory='Expert at breaking down complex topics into visual stories',
    tools=[search_tool],
    llm=llm  # Add OpenAI LLM
)

# Streamlit UI
st.set_page_config(page_title="Kids Research Helper", layout="wide")

st.title("🔍 Kids Research Helper")

# Input fields
topic = st.text_input("What would you like to research?")
age = st.slider("How old are you?", 8, 16, 12)

if st.button("Start Research"):
    if topic:
        # Create research task
        research_task = Task(
            description=f"Research {topic} and provide information suitable for age {age}",
            agent=researcher
        )
        
        # Create content writing task
        writing_task = Task(
            description=f"Write an engaging explanation about {topic} for a {age}-year-old",
            agent=content_writer
        )
        
        # Create visualization task
        visualization_task = Task(
            description=f"Create a visual explanation of {topic} for a {age}-year-old",
            agent=storyteller
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[researcher, content_writer, storyteller],
            tasks=[research_task, writing_task, visualization_task]
        )
        
        result = crew.kickoff()
        
        # Display results
        st.write("### Research Results")
        st.write(result)