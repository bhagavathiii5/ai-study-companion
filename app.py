import streamlit as st
from summarizer import summarize_transcript
from rag import chunk_text, store_chunks, answer_question

st.set_page_config(page_title="AI Study Companion", layout="centered")

st.title("AI Study Companion")
st.caption(
    "Paste a lecture or meeting transcript to get a clean summary and action items, "
    "then ask follow-up questions grounded in that exact content."
)

# session_state keeps track of app state across Streamlit's reruns
if "transcript_processed" not in st.session_state:
    st.session_state.transcript_processed = False

# --- Sidebar: quick info, keeps main area focused on the app itself ---
with st.sidebar:
    st.header("About")
    st.write(
        "This tool turns long lecture or meeting transcripts into a quick, structured "
        "summary with action items, and lets you ask specific questions about the content "
        "instead of re-reading everything."
    )
    st.write(
        "It uses Google Gemini for summarization and answering questions, combined with a "
        "local retrieval (RAG) pipeline that grounds every answer in your actual transcript "
        "rather than letting the model guess."
    )
    

# --- Input section ---
st.subheader("1. Paste your transcript")
transcript = st.text_area("Transcript text", height=250, label_visibility="collapsed")

col1, col2 = st.columns([1, 1])
with col1:
    process_clicked = st.button("Process transcript", type="primary")
with col2:
    if st.button("Clear"):
        st.session_state.clear()
        st.rerun()

if process_clicked:
    if transcript.strip() == "":
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_transcript(transcript)
            st.session_state.summary = summary

        with st.spinner("Indexing transcript for Q&A..."):
            chunks = chunk_text(transcript, chunk_size=300, overlap=60)
            store_chunks(chunks)

        st.session_state.transcript_processed = True
        st.success("Done! See the summary below.")

# --- Summary section ---
if "summary" in st.session_state:
    st.subheader("2. Summary and action items")
    st.write(st.session_state.summary)

# --- Q&A section ---
if st.session_state.transcript_processed:
    st.subheader("3. Ask a question about this transcript")
    question = st.text_input("Your question", label_visibility="collapsed", placeholder="e.g. What is due on Friday?")

    if st.button("Get answer"):
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                answer = answer_question(question, top_k=3)
            st.write(answer)