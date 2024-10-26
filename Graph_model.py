from Data_Re_processing import *
from sentence_transformers import util
import networkx as nx
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# from ollama_client import generate  
from huggingface_hub import login


def build_graph(sentences, model, nlp ):
    graph = nx.Graph()
    
    for sentence in sentences:
        entities = extract_entities(sentence, nlp)
        doc = nlp(sentence)
        verbs = [token.text for token in doc if token.pos_ == "VERB"]

        for entity, label in entities.items():
            if not graph.has_node(entity):
                graph.add_node(entity, entity_type=label)

        for verb in verbs:
            if not graph.has_node(verb):
                graph.add_node(verb, entity_type="VERB")

        entity_names = list(entities.keys())
        entity_embeddings = model.encode(entity_names, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(entity_embeddings, entity_embeddings)

        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                cosine_similarity = similarities[i][j].item()
                if cosine_similarity > 0.5:  
                    graph.add_edge(entity_names[i], entity_names[j], weight=cosine_similarity * 0.5)

        for verb in verbs:
            for entity in entities.keys():
                graph.add_edge(verb, entity, weight=0.5)  

    return graph

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


# def generate_answer(query):

#     model_path = "vinai/PhoGPT-4B-Chat"  
    
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
    
#     config.init_device = "cuda"
    
#     model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
# #     model.eval() 
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    
# #     instruction = f"Dựa vào văn bản sau đây:\n{context}\nHãy trả lời câu hỏi: {query}"
#     instruction = f"\n{query}"

#     # PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:" 
#     PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:" 


#     input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  
    
#     input_ids = tokenizer(input_prompt, return_tensors="pt")
    
#     outputs = model.generate(  
#         inputs=input_ids["input_ids"].to("cuda"),  
#         attention_mask=input_ids["attention_mask"].to("cuda"),  
#         do_sample=True,  
#         temperature=1,  
#         num_return_sequences=1,
#         top_k=50,  
#         top_p=0.9, 
#         max_new_tokens=2048,  
#         eos_token_id=tokenizer.eos_token_id,  
#         pad_token_id=tokenizer.pad_token_id  
#     )  

#     answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
#     # key_sentences = answer.split("### Trả lời:")[1]

#     # return answer.strip(), key_sentences.strip() 
#     return answer.strip()

####

# from ollama import Ollama

# def generate_answer(query, context):

#     model_name = "ollama"  # Replace PhoGPT with Ollama

#     model = Ollama(model_name)

#     # Modify the prompt as needed
#     instruction = f"\n{query}\nBest relative sentences: {context}"

#     PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:" 

#     input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

#     # Send the prompt to the Ollama model for generation
#     response = model.generate(prompt=input_prompt)

#     answer = response['text']  # Access the generated text

#     key_sentences = answer.split("### Trả lời:")[1]

#     return answer.strip(), key_sentences.strip()


# def generate_answer(query, context):
#     instruction = f"\n{query}\nBest relative sentences: {context}"
#     PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:"
#     input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

#     # Gọi hàm generate để sinh ra phản hồi với model Llama
#     response = generate(prompt=input_prompt, model="llama3.2") 

#     # Debugging: Print the entire response to check its structure
#     print("Response from generate:", response)

#     # Check if 'text' key is present in the response
#     if 'text' in response:
#         answer = response['text']
#         key_sentences = answer.split("### Trả lời:")[1] if "### Trả lời:" in answer else ""
#     else:
#         # Handle the case where 'text' key is missing
#         answer = "No answer generated."
#         key_sentences = ""

#     return answer.strip(), key_sentences.strip()



# def generate_answer(query):
#     model_path = "vinai/PhoGPT-4B-Chat"  
    
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
#     config.init_device = "cuda"
    
#     model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    
#     instruction = f"\n{query}"
#     PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:"
    
#     input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  
#     input_ids = tokenizer(input_prompt, return_tensors="pt")
    
#     outputs = model.generate(
#         inputs=input_ids["input_ids"].to("cuda"),
#         attention_mask=input_ids["attention_mask"].to("cuda"),
#         do_sample=True,
#         temperature=1,
#         num_return_sequences=1,
#         top_k=50,
#         top_p=0.9,
#         max_new_tokens=2048,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )
    
#     answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
#     # Trích xuất phần sau "### Trả lời:" và loại bỏ khoảng trắng dư thừa
#     key_sentences = answer.split("### Trả lời:")[1].strip()
    
#     return key_sentences


# Đăng nhập vào Hugging Face với mã thông báo của bạn
login(token="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")

def generate_answer(query):
    model_path = "openbmb/MiniCPM-Llama3-V-2_5"  # Thay đổi đường dẫn mô hình thành Llama 2.5
    
    # Tạo cấu hình cho mô hình
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
    config.init_device = "cuda"  # Chỉ định sử dụng GPU

    # Tải mô hình Llama 2.5
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
    # Tải tokenizer cho mô hình
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  

    # Tạo prompt cho mô hình
    instruction = f"\n{query}"
    PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:"
    
    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  
    input_ids = tokenizer(input_prompt, return_tensors="pt")

    # Tạo phản hồi từ mô hình
    outputs = model.generate(
        inputs=input_ids["input_ids"].to("cuda"),
        attention_mask=input_ids["attention_mask"].to("cuda"),
        do_sample=True,
        temperature=1,
        num_return_sequences=1,
        top_k=50,
        top_p=0.9,
        max_new_tokens=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # Giải mã phản hồi từ mô hình
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Trích xuất phần sau "### Trả lời:" và loại bỏ khoảng trắng dư thừa
    key_sentences = answer.split("### Trả lời:")[1].strip()
    
    return key_sentences
