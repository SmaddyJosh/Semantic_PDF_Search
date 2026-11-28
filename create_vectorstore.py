
# 1. Import required libraries

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma




loader = PyPDFLoader("data/Reg.pdf")

documents = loader.load()



splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)




embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"
)


# Display results

print("Documents successfully stored!")

print("Number of PDF pages:", len(documents))

print("Number of chunks:", len(chunks))

print("Number of stored vectors:", vectorstore._collection.count())
