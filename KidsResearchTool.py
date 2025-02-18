import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from crewai import Agent, Task, Crew
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool
from PIL import Image
import streamlit as st
import graphviz
import io

# Load environment variables
load_dotenv()

# Set up Streamlit secrets
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if 'SERPER_API_KEY' not in st.session_state:
    st.session_state.SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "")

# Set environment variables
os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = st.session_state.SERPER_API_KEY

# Initialize OpenAI and search tools
llm = OpenAI()
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Search the internet for information"
)

# Create agents
researcher = Agent(
    role='Research Expert',
    goal='Find accurate and age-appropriate information on given topics',
    backstory='Expert at finding reliable information for children and teens',
    tools=[search_tool],
    llm=llm
)

content_writer = Agent(
    role='Content Writer',
    goal='Write engaging and educational content for young audiences',
    backstory='Experienced in writing for children aged 8-16',
    tools=[search_tool],
    llm=llm
)

storyteller = Agent(
    role='Visual Storyteller',
    goal='Create engaging visual explanations of research topics',
    backstory='Expert at breaking down complex topics into visual stories',
    tools=[search_tool],
    llm=llm
)

def create_research_crew(topic):
    # Create tasks with all required fields
    research_task = Task(
        description=f"Research {topic} and find key information suitable for children and teens",
        agent=researcher,
        expected_output="A detailed research document about " + topic,
        context=f"Find age-appropriate information about {topic} that would interest and educate children and teens",
        tools=[search_tool]
    )
    
    writing_task = Task(
        description=f"Write an engaging and educational article about {topic}",
        agent=content_writer,
        expected_output="An engaging article written for young audiences about " + topic,
        context=f"Use the research findings to create an engaging article about {topic} suitable for children and teens",
        tools=[search_tool]
    )
    
    visualization_task = Task(
        description=f"Create a visual explanation of {topic}",
        agent=storyteller,
        expected_output="A visual representation explaining " + topic,
        context=f"Create a visual explanation of {topic} that helps children and teens understand the concept better",
        tools=[search_tool]
    )
    
    # Create the crew
    crew = Crew(
        agents=[researcher, content_writer, storyteller],
        tasks=[research_task, writing_task, visualization_task],
        verbose=True
    )
    
    return crew

# Streamlit interface
st.title("Kids Research Tool 🔍📚")

# Topic input
topic = st.text_input("Enter a topic to research:", "")

if topic:
    if st.button("Start Research"):
        with st.spinner(f"Researching {topic}..."):
            try:
                crew = create_research_crew(topic)
                result = crew.kickoff()
                st.success("Research completed!")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

