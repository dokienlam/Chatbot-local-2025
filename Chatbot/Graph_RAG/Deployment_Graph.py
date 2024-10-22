from Graph_model import *
from Data_Re_processing import *
from sentence_transformers import SentenceTransformer
from Neo4j import *

if __name__ == "__main__":
    
    file = 'C:/Users/ADMIN/Documents/GitHub/Chatbot-local-2025/Chatbot/Data_TV.docx'   # English
    # file = '/media/hanu/Learn/Self_learn/Chatbot/Data_TV.docx'  # TV 

    document_text = read_word_file(file)
    
    query = "what is the name of the fairy in this story?"
    true = 'The name of the fairy in this story is Lila'
    
    # query = "What is the name of the cat in the story?"
    # true = 'The name of the cat in the story is Whiskers'
    
    # query = "Trí tuệ nhân tạo (AI) là gì?"
    
    dpr_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    best_sentence, related_sentences, graph, best_score = graph_search(query, document_text, dpr_model)
    
    add_to_neo4j(split_sentences(document_text), graph)
    
    # draw_graph(graph)

    # answer, key_sentences = generate_answer(query, best_sentence)
    
    # keywords = extract_keywords(key_sentences)
    
    answer = generate_answer(query, best_sentence)
    keywords = extract_keywords(best_sentence)

    print(f"\nquery: {query}")
    print(f"\nBest_score: {best_score}")
    print(f"\n{answer}")
    print(f'True answer: {true}')
    print(f"Keywords: {keywords}")

    driver.close()

