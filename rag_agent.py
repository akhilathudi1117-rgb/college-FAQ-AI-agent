import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = "data"
TOP_K = 3
MAX_DISTANCE = 1.1

NOT_FOUND = (
    "I don't have that information in my knowledge base. "
    "Please contact the college office."
)

GREETING = (
    "Hello! I'm the college FAQ assistant. "
    "Ask me about timings, library, fees, exams, hostel, and more."
)

SYSTEM_PROMPT = """You are a college FAQ assistant.
Answer the student's question using ONLY the context provided.
If the context does not contain the answer, reply exactly:
"I don't have that information in my knowledge base."
Do not guess or use outside knowledge. Keep the answer short and clear."""


# -----------------------------------
# 1. LLM
# -----------------------------------

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=0
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0
        )

    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0
        )


# -----------------------------------
# 2. RAG VECTOR STORE
# -----------------------------------

def build_vectorstore():

    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n## ", "\n\n", "\n", " "]
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    return FAISS.from_documents(
        chunks,
        embeddings
    )


# -----------------------------------
# 3. LCEL CHAIN
# -----------------------------------

def build_chain(llm):

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}"
        ),
    ])

    return prompt | llm | StrOutputParser()


# -----------------------------------
# 4. COLLEGE FAQ AGENT
# -----------------------------------

class CollegeFAQAgent:

    def __init__(self):

        self.vs = build_vectorstore()

        self.chain = build_chain(
            get_llm()
        )

    def ask(self, question: str):

        trace = []

        q = question.strip()

        if not q:
            return (
                "Please type a question.",
                [],
                ["Empty question"]
            )

        if q.lower().strip("!.? ") in {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon"
        }:

            return (
                GREETING,
                [],
                ["Detected greeting - no retrieval needed"]
            )

        trace.append(
            f"Understand: question = '{q}'"
        )

        results = self.vs.similarity_search_with_score(
            q,
            k=TOP_K
        )

        for doc, score in results:

            trace.append(
                f"Retrieved chunk "
                f"(distance {score:.2f}): "
                f"{doc.page_content[:60]}..."
            )

        good = [
            d for d, s in results
            if s <= MAX_DISTANCE
        ]

        if not good:

            trace.append(
                "Decide: nothing relevant -> refuse "
                "(no hallucination)"
            )

            return (
                NOT_FOUND,
                [],
                trace
            )

        trace.append(
            f"Decide: {len(good)} relevant chunk(s) "
            "-> ask LLM"
        )

        context = "\n\n".join(
            d.page_content for d in good
        )

        answer = self.chain.invoke({
            "context": context,
            "question": q
        })

        return (
            answer,
            [d.page_content for d in good],
            trace
        )