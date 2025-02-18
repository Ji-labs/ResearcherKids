from crewai import Agent, Task, Crew
from langchain.tools import SerperDevTool
from langchain.agents import Tool
from PIL import Image
import streamlit as st
import graphviz
import io

# Initialize tools
search_tool = SerperDevTool()

# Create agents
researcher = Agent(
    role='Research Expert',
    goal='Find accurate and age-appropriate information on given topics',
    backstory='Expert at finding reliable information for children and teens',
    tools=[search_tool]
)

content_writer = Agent(
    role='Content Writer',
    goal='Write engaging and educational content for young audiences',
    backstory='Experienced in writing for children aged 8-16',
    tools=[search_tool]
)

storyteller = Agent(
    role='Visual Storyteller',
    goal='Create engaging visual explanations of research topics',
    backstory='Expert at breaking down complex topics into visual stories',
    tools=[search_tool]
)

# Streamlit UI
st.set_page_config(page_title="Kids Research Helper", layout="wide")

# Custom CSS for blue and yellow theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .stButton button {
        background-color: #ffd700;
        color: #000080;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Kids Research Helper")

# Input fields
topic = st.text_input("What would you like to research?")
age = st.slider("How old are you?", 8, 16, 12)

if st.button("Start Research!"):
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
        
        # Ask if flowchart is needed
        if st.checkbox("Would you like a flowchart to explain this topic?"):
            visualization_task = Task(
                description=f"Create a flowchart explaining {topic} for a {age}-year-old",
                agent=storyteller
            )
            crew = Crew(
                agents=[researcher, content_writer, storyteller],
                tasks=[research_task, writing_task, visualization_task]
            )
        else:
            crew = Crew(
                agents=[researcher, content_writer],
                tasks=[research_task, writing_task]
            )
        
        # Execute tasks
        result = crew.kickoff()
        
        # Display results
        st.write("### Research Results")
        st.write(result)
        
        # Store in chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({
            'topic': topic,
            'age': age,
            'result': result
        })

# Display chat history
if st.sidebar.checkbox("Show Chat History"):
    st.sidebar.write("### Previous Searches")
    for chat in st.session_state.get('chat_history', []):
        with st.sidebar.expander(f"Research: {chat['topic']}"):
            st.write(f"Age: {chat['age']}")
            st.write(chat['result'])