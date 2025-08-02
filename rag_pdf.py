from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(texts, embeddings)
    return vectordb

def ask_question(vectorstore, question):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(question)

# ✅ 加上這段才會執行
if __name__ == "__main__":
    vectorstore = create_vector_store("課程介紹.pdf")
    response = ask_question(vectorstore, "這門課的目標是什麼？")
    print(response)
