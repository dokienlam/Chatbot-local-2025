from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from Data_Re_processing import *
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from RAG_Model import *


def build_graph(sentences, model):
    graph = nx.Graph()
    
    for sentence in sentences:
        if not graph.has_node(sentence):
            graph.add_node(sentence)

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            cosine_similarity = similarities[i][j].item()
            if cosine_similarity > 0.5:  
                graph.add_edge(sentences[i], sentences[j], weight=cosine_similarity)

    return graph

def query_sentences_from_neo4j(driver):
    def query_sentences(tx):
        result = tx.run("MATCH (s:Sentence) RETURN s.content AS content")
        return [record["content"] for record in result]

    with driver.session() as session:
        sentences = session.read_transaction(query_sentences)
        print(f"Found {len(sentences)} sentences.")
    return sentences

def dpr_search(query, sentences, model):
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    dpr_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    return dpr_scores


def dpr_search_main(query, driver, model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2')):
    document_text = query_sentences_from_neo4j(driver)
    sentences = split_sentences(str(document_text))
    dpr_scores = dpr_search(query, sentences, model)
    best_idx = dpr_scores.argmax()
    return sentences[best_idx], dpr_scores[best_idx].item()
# def generate_answer(query, context):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model = GPT2LMHeadModel.from_pretrained('gpt2')

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     input_text = f"Query: {query} \nDocument: {context} \nAnswer:"
    
#     inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='longest')
    
#     outputs = model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=100,   
#         num_return_sequences=1,
#         pad_token_id=tokenizer.pad_token_id,
#         do_sample = True, 
#         temperature=0.1,   
#         top_k=1,          
#         top_p=0.1,   
#     )
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     key_sentences = answer.split("Answer:")[-1].strip()
#     return answer.strip(), key_sentences
    
#------------------------------------ RAG 
def generate_answer(query, best_entity_name):
    model_path = "vinai/PhoGPT-4B-Chat"  
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
    config.init_device = "cuda"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    
    instruction = f"\n###Câu hỏi: {query} \n ###Tài liệu: {best_entity_name}" 
    # instruction = f"{query}" 

    PROMPT_TEMPLATE = f"### Câu hỏi: {instruction} \n### Trả lời:"
    
    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  
    input_ids = tokenizer(input_prompt, return_tensors="pt")
    
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
    
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Trích xuất phần sau "### Trả lời:" và loại bỏ khoảng trắng dư thừa
    key_sentences = answer.split("### Trả lời:")[1].strip()
    
    return key_sentences


#----------------------------------------Chatbot


# def generate_answer(query, best_entity_name):
#     model_path = "vinai/PhoGPT-4B-Chat"  
    
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
#     config.init_device = "cuda"
    
#     model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    
#     # instruction = f"\n###Câu hỏi: {query} \n ###Tài liệu: {best_entity_name}" 
#     instruction = f"{query}" 

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
