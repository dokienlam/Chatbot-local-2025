from Graph_model import *
from sentence_transformers import SentenceTransformer
import spacy 
from Neo4j import *

if __name__ == "__main__":
    
    file = r'C:\Users\ADMIN\Documents\GitHub\Chatbot-local-2025\Chatbot\Data_TV.docx'  # TV 
#     file = '/kaggle/input/10k-dataset/10000_word.docx'
    
    document_text = read_word_file(file)
    
    nlp = spacy.load("en_core_web_sm")  
    
#     dpr_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    dpr_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, dpr_model ,nlp)
    
    add_to_neo4j(graph, driver)
    
    query = "Trí tuệ nhân tạo (AI) là gì?"
#     query = "AI có thể chia thành mấy loại chính?"
#     query = "Học không có giám sát là gì?"

    best_entity_name, best_entity_type, best_score = find_most_similar_entity(query, driver, dpr_model)
    
    related_entities = query_related_entities(best_entity_name, driver)

    answer, key_sentences = generate_answer(query, best_entity_name)
    keywords = extract_keywords(key_sentences)
    
    print(f"Best entity: {best_entity_name}, Similarity: {best_score}")
    print(f"\nAnswer: {answer}")
#     print(f"Keywords: {keywords}")

    driver.close()
