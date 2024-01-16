from langchain.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

my_activeloop_id="your_deeplake_id" #replace with your deeplake ID
my_activeloop_dataset_name = "your_dataset_name" #replace with your dataset name
dataset_path = f"hub://{my_activeloop_id}/{my_activeloop_dataset_name}"

def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    dbs = DeepLake(
        dataset_path=dataset_path, 
        read_only=True, 
        embedding=embeddings
        )
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor = CohereRerank(
        model = 'rerank-english-v2.0',
        top_n=5,
        user_agent="langchain"
        )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

dbs, compression_retriever, retriever = data_lake()

def memory():
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

memory=memory()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, verbose=True, temperature=0, max_tokens=1500)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    verbose=False,
    chain_type="stuff",
    return_source_documents=False
)

# Load the memory variables, which include the chat history
memory_variables = memory.load_memory_variables({})

# JSON to store the result
result = {}

#1st question
query = "Can the outsource agreement with Document Systems be terminatied before the expiratoin of its initial term? "
result = chain({"question": query, "chat_history": memory_variables})
print(result)

#2nd question
query = "What should be the notie period in case of early termination?"
result = chain({"question": query, "chat_history": memory_variables})
print(result)
