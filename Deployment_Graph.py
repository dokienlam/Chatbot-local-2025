from Graph_model import *
from sentence_transformers import SentenceTransformer
import spacy 
from Neo4j import *


def main2():
    file = r'D:\AI1709\Ky7\DAT301m\Chatbot-local-2025\Chatbot\New.docx'  
    document_text = read_word_file(file)
    
    nlp = spacy.load("en_core_web_sm")  
    
    dpr_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, dpr_model ,nlp)

    add_to_neo4j(graph, driver)




def main(query):    

    dpr_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    best_entity_name, best_entity_type, best_score = find_most_similar_entity(query, driver, dpr_model)
    related_entities = query_related_entities(best_entity_name, driver)
    answer = generate_answer(query, best_entity_name)
    
    print(f"Best entity: {best_entity_name}, Similarity: {best_score}")
    print(f"\nAnswer: {answer}")

    driver.close()
    return answer 

