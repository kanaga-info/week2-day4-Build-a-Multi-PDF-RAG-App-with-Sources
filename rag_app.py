
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Palo Alto Cloud NGFW RAG Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Palo Alto Cloud NGFW RAG Assistant")

st.markdown(
    """
    **AWS + Azure Technical Documentation Assistant**

    Ask technical questions about Palo Alto Networks
    Cloud NGFW deployment, architecture, routing,
    security policies, NAT and traffic inspection.
    """
)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Check your .env file.")
    st.stop()


# ==================================================
# 2. SESSION STATE
# ==================================================

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ==================================================
# 3. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🛡️ Cloud NGFW Assistant")

    st.write("Supported documentation:")

    st.write("☁️ AWS Cloud NGFW")
    st.write("☁️ Azure Cloud NGFW")

    st.divider()

    st.subheader("Knowledge Base")

    if st.session_state.store is not None:

        st.success("Knowledge Base Ready")

        st.metric(
            "Indexed Documents",
            len(st.session_state.indexed_files)
        )

        st.metric(
            "Total Chunks",
            st.session_state.total_chunks
        )

        for filename in st.session_state.indexed_files:
            st.caption(f"📄 {filename}")

    else:
        st.info("Upload and index your PDF documents.")

    st.divider()

    st.caption("Built using LangChain, Streamlit, OpenAI and FAISS.")


# ==================================================
# 4. MULTIPLE PDF UPLOAD
# ==================================================

st.subheader("Step 1: Upload Cloud NGFW Documents")

uploaded_files = st.file_uploader(
    "Upload AWS and Azure Cloud NGFW PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write("Selected documents:")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")


# ==================================================
# 5. DOCUMENT INDEXING
# ==================================================

if st.button("🚀 Build Cloud NGFW Knowledge Base"):

    if not uploaded_files:

        st.warning("Please upload your PDF documents.")

    else:

        all_chunks = []
        processed_files = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        progress = st.progress(0)

        try:

            for index, uploaded_file in enumerate(uploaded_files):

                st.write(f"Processing: {uploaded_file.name}")

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:

                    # Load PDF
                    loader = PyPDFLoader(temp_path)

                    pages = loader.load()

                    if not pages:
                        raise ValueError(
                            f"No pages found in {uploaded_file.name}"
                        )

                    # Preserve original PDF filename
                    for page in pages:

                        page.metadata["source"] = uploaded_file.name

                    # Split PDF into chunks
                    chunks = splitter.split_documents(pages)

                    all_chunks.extend(chunks)

                    processed_files.append(uploaded_file.name)

                    st.write(
                        f"✅ {uploaded_file.name}: "
                        f"{len(chunks)} chunks"
                    )

                finally:

                    Path(temp_path).unlink(missing_ok=True)

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            if not all_chunks:

                st.error("No readable PDF content found.")
                st.stop()

            # Initialize OpenAI embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            # Create ONE FAISS database for all PDFs
            vector_store = FAISS.from_documents(
                all_chunks,
                embeddings
            )

            # Save in Streamlit session state
            st.session_state.store = vector_store

            st.session_state.indexed_files = processed_files

            st.session_state.total_chunks = len(all_chunks)

            st.success(
                "Cloud NGFW Knowledge Base created successfully!"
            )

            st.metric(
                "Total Indexed Chunks",
                len(all_chunks)
            )

            st.balloons()

        except Exception as e:

            st.error(f"Indexing error: {str(e)}")


# ==================================================
# 6. QUESTION AND ANSWER
# ==================================================

st.divider()

st.subheader("Step 2: Ask Technical Questions")

question = st.text_input(
    "Enter your Cloud NGFW question",
    placeholder=(
        "Example: Explain AWS centralized deployment "
        "and its traffic flow."
    )
)

# Suggested questions
st.markdown("**Try these questions:**")

example_questions = [
    "Explain AWS centralized Cloud NGFW deployment.",
    "How does Azure Local Rulestack work?",
    "Explain Azure Cloud NGFW inbound traffic flow.",
    "Compare AWS and Azure East-West traffic inspection.",
    "How is logging configured in Azure Cloud NGFW?"
]

for example in example_questions:

    if st.button(example, key=example):

        st.session_state.selected_question = example


if "selected_question" in st.session_state:

    question = st.session_state.selected_question


# ==================================================
# 7. RETRIEVE DOCUMENTS FROM BOTH SOURCES
# ==================================================

if st.button("🔍 Generate Technical Answer"):

    if st.session_state.store is None:

        st.warning("Build the Knowledge Base first.")

    elif not question.strip():

        st.warning("Please enter a technical question.")

    else:

        try:

            retrieved_docs = []

            # Search each uploaded PDF separately.
            # All PDFs still use the same FAISS database.

            for filename in st.session_state.indexed_files:

                source_docs = st.session_state.store.similarity_search(
                    question,
                    k=5,
                    filter={"source": filename}
                )

                retrieved_docs.extend(source_docs)

            if not retrieved_docs:

                st.warning("No relevant information was found.")
                st.stop()

            # ==================================================
            # 8. PREPARE SOURCE CONTEXT
            # ==================================================

            context_parts = []

            for doc in retrieved_docs:

                filename = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                # PyPDFLoader page metadata starts from zero
                page_number = int(
                    doc.metadata.get("page", 0)
                ) + 1

                context_parts.append(
                    f"""
                    SOURCE FILE: {filename}
                    PDF PAGE: {page_number}

                    DOCUMENT CONTENT:
                    {doc.page_content}
                    """
                )

            context = "\n\n".join(context_parts)


            # ==================================================
            # 9. INITIALIZE CHAT MODEL
            # ==================================================

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )


            # ==================================================
            # 10. PALO ALTO CLOUD NGFW RAG PROMPT
            # ==================================================

            prompt = f"""
            You are a Senior Palo Alto Networks Cloud NGFW
            Technical Assistant.

            You answer technical questions using ONLY the
            provided AWS and Azure Cloud NGFW documentation.

            DOCUMENTATION:
            {context}

            USER QUESTION:
            {question}

            INSTRUCTIONS:

            1. Answer using the provided document content only.

            2. Explain technical concepts clearly with
               appropriate technical terminology.

            3. For AWS questions, focus on AWS architecture,
               Transit Gateway, Cloud WAN, deployment models,
               routing and traffic inspection where relevant.

            4. For Azure questions, focus on VNet, Virtual WAN,
               Local Rulestack, NAT, NSG, logging and routing
               where relevant.

            5. For comparison questions, compare AWS and Azure
               separately. Do not assume their configurations
               or capabilities are identical.

            6. If AWS and Azure documents contain conflicting
               information, clearly identify the difference
               and cite both sources. Do not silently choose
               one answer.

            7. Distinguish differences in deployment models,
               cloud architecture and documentation scope
               from actual contradictions.

            8. Never invent CLI commands, configuration values,
               IP addresses, steps or unsupported capabilities.

            9. Cite supporting information using this format:
               [Filename, PDF Page X]

            10. If information is not available in the supplied
                documents, clearly state:
                "This information is not available in the
                uploaded Cloud NGFW documentation."

            ANSWER FORMAT:

            ### Technical Explanation

            Provide a clear answer.

            ### Configuration / Traffic Flow

            Explain the relevant steps when applicable.

            ### Important Notes

            Mention relevant limitations or differences.

            ### Sources

            List the supporting PDF filenames and page numbers.
            """

            # Generate answer
            response = llm.invoke(prompt)


            # ==================================================
            # 11. DISPLAY ANSWER
            # ==================================================

            st.divider()

            st.subheader("💡 Technical Answer")

            st.markdown(response.content)


            # ==================================================
            # 12. DISPLAY SOURCE DOCUMENTS
            # ==================================================

            st.subheader("📚 Source References")

            unique_sources = {}

            for doc in retrieved_docs:

                filename = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page_number = int(
                    doc.metadata.get("page", 0)
                ) + 1

                key = (filename, page_number)

                if key not in unique_sources:

                    unique_sources[key] = doc.page_content

            for (filename, page_number), content in unique_sources.items():

                with st.expander(
                    f"📄 {filename} | PDF Page {page_number}"
                ):

                    st.write(content)

            st.caption(
                f"Retrieved {len(retrieved_docs)} document chunks "
                f"from {len(st.session_state.indexed_files)} PDFs."
            )

        except Exception as e:

            st.error(f"Question answering failed: {str(e)}")


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "Palo Alto Cloud NGFW RAG Assistant | "
    "AWS + Azure | Powered by LangChain and FAISS"
)
