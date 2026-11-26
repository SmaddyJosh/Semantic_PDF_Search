from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_huggingface import HuggingFaceEmbeddings 

loader=PyPDFLoader("data/Reg.pdf")
documents=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

vector= embeddings.embed_query(
    chunks[0].page_content
)

print ("embedding vector for the first chunk:", vector)

print("Vector dimension:", len(vector))