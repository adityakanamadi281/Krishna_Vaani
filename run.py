import os
import sys
import typer
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.ingestion import BhagavadGitaParser, SemanticChunker
from src.indexing import FAISSIndexManager
from src.retrieval import HybridRetrieverManager, LocalReranker
from src.generation import KrishnaGenerator

app = typer.Typer(help="Krishna Vaani - Bhagavad Gita Production RAG CLI")

@app.command()
def ingest(
    pdf_path: str = typer.Option(None, "--pdf", "-p", help="Path to Bhagavad Gita PDF"),
    chunk_size: int = typer.Option(1200, "--chunk-size", "-c", help="Size of semantic chunks"),
    overlap: int = typer.Option(150, "--overlap", "-o", help="Overlap between chunks")
):
    """
    Parses, chunks, and indexes the Bhagavad Gita PDF into the FAISS vector database.
    """
    logger.info("Starting Ingestion Pipeline...")
    
    path_to_pdf = pdf_path or str(settings.PDF_PATH)
    if not os.path.exists(path_to_pdf):
        logger.error(f"PDF not found at {path_to_pdf}. Place the PDF file there or specify another path.")
        sys.exit(1)
        
    # 1. Parse PDF
    parser = BhagavadGitaParser(path_to_pdf)
    parsed_docs = parser.parse()
    
    # 2. Chunk semantically
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    final_chunks = chunker.chunk_documents(parsed_docs)
    
    # 3. Save documents & Build FAISS Vector Index
    faiss_manager = FAISSIndexManager()
    faiss_manager.build_index(final_chunks)
    
    # 4. Save documents list for BM25
    retriever_manager = HybridRetrieverManager(faiss_manager)
    retriever_manager.save_documents(final_chunks)
    
    logger.info("Ingestion Pipeline completed successfully!")


@app.command()
def query(
    user_query: str = typer.Argument(..., help="The search query or question for Krishna"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of retrieved candidates before reranking"),
    rerank_n: int = typer.Option(4, "--rerank-n", "-r", help="Number of final context blocks to feed LLM")
):
    """
    Query the RAG system and print the response from Lord Krishna.
    """
    logger.info(f"Querying Krishna Vaani for: '{user_query}'")
    
    # Initialize managers
    faiss_manager = FAISSIndexManager()
    
    # Load index to verify existence
    try:
        faiss_manager.load_index()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    retriever_manager = HybridRetrieverManager(faiss_manager)
    retriever_manager.initialize_retrievers()
    
    reranker = LocalReranker()
    generator = KrishnaGenerator()
    
    # 1. Retrieve candidates
    candidates = retriever_manager.retrieve(user_query)
    
    # 2. Rerank candidates
    top_docs = reranker.rerank(user_query, candidates, top_n=rerank_n)
    
    # 3. Generate response
    response = generator.generate(user_query, top_docs)
    
    print("\n" + "="*50)
    print("RESPONSE FROM LORD KRISHNA:")
    print("="*50)
    print(response)
    print("="*50 + "\n")


@app.command()
def chat(
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of retrieved candidates before reranking"),
    rerank_n: int = typer.Option(4, "--rerank-n", "-r", help="Number of final context blocks to feed LLM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show ingestion and retrieval logs in console")
):
    """
    Start an interactive chat session with Lord Krishna in the terminal.
    """
    if not verbose:
        # Suppress logging in console to keep the chat UI clean
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        
    print("\n" + "="*80)
    print("Welcome to Krishna Vaani (Bhagavad Gita RAG Chat)")
    print("Speak with Lord Krishna. Type 'exit', 'quit', or 'q' to end the session.")
    print("="*80 + "\n")
    
    faiss_manager = FAISSIndexManager()
    
    # Load index to verify existence
    try:
        faiss_manager.load_index()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the ingestion pipeline first: uv run python run.py ingest")
        sys.exit(1)
        
    retriever_manager = HybridRetrieverManager(faiss_manager)
    retriever_manager.initialize_retrievers()
    
    reranker = LocalReranker()
    generator = KrishnaGenerator()
    
    history = []
    
    while True:
        try:
            user_input = input("\nSeeker: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nKrishna: Go forth and perform your duty with dedication. Farewell, seeker.\n")
                break
                
            # 1. Rewrite query based on history if available
            rewritten_query = generator.rewrite_query(user_input, history)
            
            # 2. Retrieve candidates
            candidates = retriever_manager.retrieve(rewritten_query)
            
            # 3. Rerank candidates
            top_docs = reranker.rerank(rewritten_query, candidates, top_n=rerank_n)
            
            # 4. Generate streaming response
            print("\nKrishna: ", end="", flush=True)
            response_chunks = []
            for chunk in generator.generate_stream(user_input, top_docs):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print()
            
            full_response = "".join(response_chunks)
            
            # 5. Append to history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\nKrishna: Farewell, seeker. May peace be with you.\n")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    app()

