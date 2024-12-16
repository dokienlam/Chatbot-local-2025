from RAG_Model import *
from sentence_transformers import SentenceTransformer
import spacy 
from Neo4j import *


def main2():
    file = r'D:\AI1709\Ky7\DAT301m\Chatbot-local-2025\Chatbot\New.docx'  # TV 
    
    document_text = read_word_file(file)
    
    dpr_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, dpr_model)
    add_to_neo4j(driver, sentences)




def main(query):    
    
    best_sentence, score = dpr_search_main(query, driver)
    print(best_sentence)
    answer = generate_answer(query, best_sentence)
    
    print(f"\nAnswer: {answer}")

    driver.close()
    return answer 
