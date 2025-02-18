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

# Load environment variables and set up API keys
load_dotenv()

# Set up Streamlit secrets
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if 'SERPER_API_KEY' not in st.session_state:
    st.session_state.SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "")

# Set environment variables
os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = st.session_state.SERPER_API_KEY

# Initialize tools
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

def create_tasks(topic):
    tasks = [
        Task(
            description=f"Research {topic} and find key information suitable for children and teens",
            expected_output=f"Detailed research findings about {topic}",
            agent=researcher,
            async_execution=False,
            output_file=None,
            context=f"Research {topic} in a way that's suitable for children and teens"
        ),
        Task(
            description=f"Write an engaging and educational article about {topic}",
            expected_output=f"An engaging article about {topic} for young audiences",
            agent=content_writer,
            async_execution=False,
            output_file=None,
            context=f"Write about {topic} in a way that children and teens can understand"
        ),
        Task(
            description=f"Create a visual explanation of {topic}",
            expected_output=f"A visual representation explaining {topic}",
            agent=storyteller,
            async_execution=False,
            output_file=None,
            context=f"Create visuals that help explain {topic} to children and teens"
        )
    ]
    return tasks

# Streamlit interface
st.title("Kids Research Tool 🔍📚")

topic = st.text_input("Enter a topic to research:", "")

if topic:
    if st.button("Start Research"):
        with st.spinner(f"Researching {topic}..."):
            try:
                tasks = create_tasks(topic)
                crew = Crew(
                    agents=[researcher, content_writer, storyteller],
                    tasks=tasks,
                    verbose=True
                )
                result = crew.kickoff()
                st.success("Research completed!")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
