from RAG_Model import *
from sentence_transformers import SentenceTransformer
import spacy 
from Neo4j import *


def main2():
    file = r'D:\AI1709\Ky7\DAT301m\Chatbot-local-2025\Chatbot\New.docx'  # TV 
#     file = '/kaggle/input/10k-dataset/10000_word.docx'Deployment_Graph.py
    
    document_text = read_word_file(file)
    
    dpr_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    # dpr_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, dpr_model)
    add_to_neo4j(driver, sentences)




def main(query):    
    
    best_sentence, score = dpr_search_main(query, driver)
    print(best_sentence)
    answer = generate_answer(query, best_sentence)
    
    # print(f"Best sentence: {best_sentence}, Similarity: {score}")
    print(f"\nAnswer: {answer}")

    driver.close()
    return answer 


# def main(query):    
#     # query = "Trí tuệ nhân tạo (AI) là gì?"
# #     query = "AI có thể chia thành mấy loại chính?"
# #     query = "Học không có giám sát là gì?"

#     answer = generate_answer(query)
    
#     # print(f"Best entity: {best_entity_name}, Similarity: {best_score}")
#     print(f"\nAnswer: {answer}")
# #     print(f"Keywords: {keywords}")

#     driver.close()
#     return answer 