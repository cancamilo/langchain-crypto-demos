import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
import streamlit as st


def main():
    st.title("Smart search app")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "agent" not in st.session_state:
        st.session_state["agent"] = init_agent()

    # Create a form to capture user input
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input("User input", key="user_input")
        submit_button = st.form_submit_button("Send")

        if submit_button:
            st.session_state["messages"].append(("User", user_input))

            # Generate bot response
            bot_response = generate_bot_response(user_input)

            # clear_text()

            # Append bot response to the conversation
            st.session_state["messages"].append(("Bot", bot_response))

    # Display the conversation history
    conversation_html = format_conversation(st.session_state["messages"])
    st.markdown(conversation_html, unsafe_allow_html=True)


def clear_text():
    st.session_state["user_input"] = ""

def init_agent():
    # initialize agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    search = BingSearchAPIWrapper()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world",
        ),
    ]
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    return agent_chain


def format_conversation(messages):
    # Format the conversation as HTML
    conversation_html = ""
    for sender, message in messages:
        conversation_html += f"<p><strong>{sender}: </strong>{message}</p>"
    return conversation_html


def generate_bot_response(user_input):
    # reuse same agent
    agent = st.session_state["agent"]
    extracted_messages = agent.memory
    return agent.run(input=user_input)


if __name__ == "__main__":
    main()
