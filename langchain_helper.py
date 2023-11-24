from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS  # for similarity search
from langchain.embeddings.openai import OpenAIEmbeddings


from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# video_url = "https://www.youtube.com/watch?v=OyFJWRnt_AY"


def create_vector_db_from_ytUrl(video_url: str) -> FAISS:
    '''First load the youtube video. calling loader.load() will automatically transcribe the youtube video. Next we split the text with RecursiveCharacterTextSplitter. Create and return a vector store object '''

    loader = YoutubeLoader.from_youtube_url(video_url)
    video_transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.split_documents(video_transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    ''' text_davinci max tokens = 4097 
    Semantic search '''

    docs = db.similarity_search(query, k=k)

    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""You are a helpful YouTube assistant that can answer questions about videos from video transcripts.
        Answer the following question: {question}
        By searching the following video transcript: {docs}

        If you feel like you do not have enough information to give the answer, simply say "I have not much information to answer the question!"

        Your answers should be detailed.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
