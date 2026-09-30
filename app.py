"""Streamlit chat interface.  Run:  streamlit run app.py"""
import streamlit as st
from rag_agent import CollegeFAQAgent

st.set_page_config(page_title="College FAQ Agent", page_icon="🎓")
st.title("🎓 College FAQ AI Agent")
st.caption("Answers only from the college-provided information.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_agent():
    return CollegeFAQAgent()


agent = load_agent()

with st.sidebar:
    st.header("Try these")
    for q in ["What time does the college start?",
              "What is the library late fee?",
              "Where is the CSE department?",
              "Who won the 2030 World Cup?"]:
        if st.button(q):
            st.session_state["pending"] = q

if "history" not in st.session_state:
    st.session_state["history"] = []

for role, text, extra in st.session_state["history"]:
    with st.chat_message(role):
        st.write(text)

question = st.chat_input("Ask a question about the college...") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Thinking..."):
        answer, sources, trace = agent.ask(question)
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Agent trace & sources"):
            st.write("**Steps**")
            for t in trace:
                st.write("- " + t)
            if sources:
                st.write("**Sources used**")
                for s in sources:
                    st.info(s)
    st.session_state["history"] += [("user", question, None), ("assistant", answer, None)]
