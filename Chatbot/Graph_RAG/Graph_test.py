import re
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from keybert import KeyBERT
from neo4j import GraphDatabase
import spacy 

uri = 'bolt://34.201.205.231:7687'
username = "neo4j"
password = "shotlines-retailers-breach"
driver = GraphDatabase.driver(uri, auth=(username, password))

nlp = spacy.load("en_core_web_sm")  

def split_sentences(document_text):
    sentences = re.split(r'\.', document_text)
    return [s.strip() for s in sentences if s.strip()]

def extract_entities(sentence):
    doc = nlp(sentence)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_  
    return entities

def build_graph(sentences, model):
    graph = nx.Graph()
    
    for sentence in sentences:
        # Xác định thực thể trong câu
        entities = extract_entities(sentence)
        for entity, label in entities.items():
            if not graph.has_node(entity):
                graph.add_node(entity, entity_type=label)
        
        # Tính toán độ tương đồng giữa các thực thể thay vì câu
        entity_names = list(entities.keys())
        entity_embeddings = model.encode(entity_names, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(entity_embeddings, entity_embeddings)

        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                similarity = similarities[i][j].item()
                if similarity > 0.5:  # Ngưỡng để tạo liên kết
                    graph.add_edge(entity_names[i], entity_names[j], weight=similarity)

    return graph

def add_to_neo4j(graph, driver):
    with driver.session() as session:
        # Thêm các thực thể vào Neo4j
        for node in graph.nodes(data=True):
            entity = node[0]
            entity_type = node[1].get("entity_type", "Unknown")
            session.run("MERGE (e:Entity {name: $name, type: $type})", name=entity, type=entity_type)

        # Thêm các cạnh giữa các thực thể dựa trên độ tương đồng
        for u, v, weight in graph.edges(data=True):
            session.run("""
                MATCH (e1:Entity {name: $name1}), (e2:Entity {name: $name2})
                CREATE (e1)-[:SIMILARITY {weight: $weight}]->(e2)
            """, name1=u, name2=v, weight=weight['weight'])

def query_entities_from_neo4j(driver):
    def query_entities(tx):
        result = tx.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type")
        return [(record["name"], record["type"]) for record in result]
    
    with driver.session() as session:
        entities = session.read_transaction(query_entities)
    return entities

def find_most_similar_entity(query, driver, model):
    entities = query_entities_from_neo4j(driver)
    query_embedding = model.encode(query, convert_to_tensor=True)
    entity_names = [name for name, _ in entities]
    entity_embeddings = model.encode(entity_names, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, entity_embeddings)[0]
    best_idx = similarities.argmax().item()
    
    best_entity_name = entity_names[best_idx]
    best_entity_type = entities[best_idx][1]
    best_similarity_score = similarities[best_idx].item()

    return best_entity_name, best_entity_type, best_similarity_score

def query_related_entities(entity_name, driver):
    def query_neighbors(tx, entity_name):
        result = tx.run("""
            MATCH (e:Entity {name: $name})-[:SIMILARITY]->(related)
            RETURN related.name AS related_entity
        """, name=entity_name)
        return [record["related_entity"] for record in result]
    
    with driver.session() as session:
        related_entities = session.read_transaction(query_neighbors, entity_name)
    
    return related_entities

def generate_answer(query, context):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_text = f"Query: {query} \nDocument: {context} \nAnswer:"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='longest')

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=1024,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.1,
        top_k=1,
        top_p=0.1,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    key_sentences = answer.split("Answer:")[-1].strip()
    return answer.strip(), key_sentences

def extract_keywords(answer):
    kw_model = KeyBERT()
    answer_keywords = kw_model.extract_keywords(answer, top_n=100, stop_words='english')
    answer_keywords = [kw[0] for kw in answer_keywords]
    combined_keywords = list(set(answer_keywords))
    return str(combined_keywords)

if __name__ == "__main__":
    document_text = """
    tôi tên là Lâm, chuyên ngành của tôi là trí truệ nhân tạo, sở thích của tôi là đi chơi.
    """
    query = "tôi tên là gì"
    
    dpr_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, dpr_model)
    
    add_to_neo4j(graph, driver)
    
    best_entity_name, best_entity_type, best_score = find_most_similar_entity(query, driver, dpr_model)
    print(f"Best entity: {best_entity_name}, Type: {best_entity_type}, Similarity: {best_score}")
    
    related_entities = query_related_entities(best_entity_name, driver)

    answer, key_sentences = generate_answer(query, best_entity_name)
    keywords = extract_keywords(key_sentences)
    
    print(f"\nAnswer: {answer}")
    print(f"Keywords: {keywords}")

    driver.close()
