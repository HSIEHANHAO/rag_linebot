from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


def load_documents_from_pdf():
    loader = PyPDFLoader("課程介紹.pdf")  # PDF 路徑請放對
    documents = loader.load()
    return documents

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def build_vector_store():
    print("📄 載入 PDF 文件...")
    documents = load_documents_from_pdf()

    print("✂️ 拆分文字...")
    split_docs = split_documents(documents)

    print("🧠 建立向量資料庫...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")

    print("✅ 向量資料庫建立完成")

if __name__ == "__main__":
    build_vector_store()
