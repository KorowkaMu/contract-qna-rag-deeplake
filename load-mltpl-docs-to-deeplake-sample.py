from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
import os


# Set up the folder path
folder_path = 'path/to/your/contracts/folder'

# Define embeddings model
embeddings = CohereEmbeddings(model = "embed-english-v2.0")

# create Deep Lake dataset
#use your organization id here. (by default, org id is your username)
my_activeloop_id="your_deeplake_id"
my_activeloop_dataset_name = "your_dataset_name"
dataset_path = f"hub://{my_activeloop_id}/{my_activeloop_dataset_name}"

#uncomment the following line if you need to create a new dataset
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

#reload the existing dataset
#db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=False)


# Iterate through each file in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        file_path = os.path.join(folder_path, filename)
        
        # Open the PDF file in read-binary mode
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # we split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100, length_function=len
        )
        docs_split = text_splitter.split_documents(docs)

        #add documents to our Deep Lake "sample contracts" dataset. Uncomment the following code to add a new document
        db.add_documents(docs_split)




