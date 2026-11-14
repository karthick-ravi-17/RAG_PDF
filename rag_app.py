
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5


# --------------------------------------------------
# 2. Initialize Session State
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 3. Load and Extract PDF Documents
# --------------------------------------------------

def load_pdf_documents(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        temp_path = None

        try:
            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Load PDF page by page
            loader = PyPDFLoader(temp_path)

            documents = loader.load()

            # Preserve original file name and page metadata
            for document in documents:

                document.metadata["file_name"] = uploaded_file.name

                document.metadata["page_number"] = (
                    document.metadata.get("page", 0) + 1
                )

                # PyPDFLoader does not automatically
                # identify PDF section headings.
                document.metadata["section"] = (
                    document.metadata.get("section", "Not detected")
                )

                document.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return all_documents


# --------------------------------------------------
# 4. Split Documents into Chunks
# --------------------------------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# 5. Create a Single FAISS Vector Database
# --------------------------------------------------

def create_vector_store(chunks):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# --------------------------------------------------
# 6. Build the RAG Index
# --------------------------------------------------

def build_rag_index(uploaded_files):

    with st.spinner("Reading and processing PDF documents..."):

        documents = load_pdf_documents(uploaded_files)

        if not documents:
            raise ValueError(
                "No readable PDF pages were found."
            )

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text chunks were generated from the PDFs."
            )

        vector_store = create_vector_store(chunks)

        # Store the combined vector database
        # in Streamlit session state
        st.session_state.vector_store = vector_store

        st.session_state.chunk_count = len(chunks)

        st.session_state.indexed_files = [
            file.name for file in uploaded_files
        ]

        # Clear previous chat after rebuilding index
        st.session_state.chat_history = []

        return len(chunks)


# --------------------------------------------------
# 7. Retrieve Relevant Documents
# --------------------------------------------------

def retrieve_documents(question):

    vector_store = st.session_state.vector_store

    relevant_documents = vector_store.similarity_search(
        question,
        k=RETRIEVAL_K
    )

    return relevant_documents


# --------------------------------------------------
# 8. Generate Answer using OpenAI LLM
# --------------------------------------------------

def generate_answer(question):

    relevant_documents = retrieve_documents(question)

    context_parts = []

    for index, document in enumerate(relevant_documents, start=1):

        file_name = document.metadata.get(
            "file_name", "Unknown file"
        )

        page_number = document.metadata.get(
            "page_number", "Unknown"
        )

        section = document.metadata.get(
            "section", "Not detected"
        )

        context_parts.append(
            f"""
            [SOURCE {index}]

            File: {file_name}
            Page: {page_number}
            Section: {section}

            Content:
            {document.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=API_KEY
    )

    messages = [
        (
            "system",
            """
            You are a PDF document question-answering assistant.

            Answer the user's question using ONLY the
            retrieved document context provided below.

            Instructions:

            1. Do not invent facts or information.
            2. If the answer is not available in the
               documents, clearly say so.
            3. Every factual claim must include its
               supporting source reference.
            4. Use source references such as [SOURCE 1]
               and [SOURCE 2].
            5. Never fabricate page numbers or file names.
            6. Provide a clear and concise answer.

            Retrieved document context:

            {context}
            """.format(context=context)
        ),
        (
            "human",
            question
        )
    ]

    response = llm.invoke(messages)

    return response.content, relevant_documents


# --------------------------------------------------
# 9. Streamlit User Interface
# --------------------------------------------------

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload multiple PDF documents and ask questions "
    "across all uploaded documents."
)


# --------------------------------------------------
# 10. Sidebar - PDF Upload
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not API_KEY:

            st.error(
                "OPENAI_API_KEY not found. "
                "Please configure your .env file."
            )

        elif not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            try:

                chunk_count = build_rag_index(
                    uploaded_files
                )

                st.success(
                    f"Successfully indexed {chunk_count} chunks!"
                )

                st.info(
                    f"Processed {len(uploaded_files)} PDF documents."
                )

            except Exception as error:

                st.error(
                    f"Indexing failed: {str(error)}"
                )


    # Show indexing information

    if st.session_state.vector_store is not None:

        st.divider()

        st.subheader("📊 Index Information")

        st.metric(
            "Total Chunks Indexed",
            st.session_state.chunk_count
        )

        st.write("**Indexed Documents:**")

        for file_name in st.session_state.indexed_files:
            st.write(f"📄 {file_name}")


# --------------------------------------------------
# 11. Chat Interface
# --------------------------------------------------

st.subheader("💬 Ask Questions")

if st.session_state.vector_store is None:

    st.info(
        "Upload and process your PDF documents "
        "before asking questions."
    )

else:

    # Display previous conversation
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":

                for source in message.get("sources", []):

                    with st.expander(
                        f"📄 {source['file']} - Page {source['page']}"
                    ):

                        st.write(
                            f"Section: {source['section']}"
                        )

                        st.write(source["content"])


    user_question = st.chat_input(
        "Ask something about your uploaded PDFs..."
    )

    if user_question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents and generating answer..."):

                try:

                    answer, source_documents = generate_answer(
                        user_question
                    )

                    st.markdown(answer)

                    # Prepare source information
                    sources = []

                    for document in source_documents:

                        sources.append({
                            "file": document.metadata.get(
                                "file_name", "Unknown"
                            ),
                            "page": document.metadata.get(
                                "page_number", "Unknown"
                            ),
                            "section": document.metadata.get(
                                "section", "Not detected"
                            ),
                            "content": document.page_content
                        })

                    # Display sources
                    st.markdown("### 📚 Answer Sources")

                    for index, source in enumerate(sources, start=1):

                        with st.expander(
                            f"Source {index}: "
                            f"{source['file']} | "
                            f"Page {source['page']}"
                        ):

                            st.write(
                                f"Section: {source['section']}"
                            )

                            st.write(source["content"])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as error:

                    error_message = (
                        f"Error generating answer: {str(error)}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })
