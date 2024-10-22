from sentence_transformers import util
import networkx as nx
from matplotlib import pyplot as plt
from Data_Re_processing import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import os
from dotenv import load_dotenv


def build_graph(sentences, model):
    graph = nx.Graph()
    
    for i, sentence in enumerate(sentences):
        graph.add_node(i, sentence=sentence)

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = similarities[i][j].item()
            if similarity > 0.5:  
                graph.add_edge(i, j, weight=similarity)

    return graph

def draw_graph(graph):
    pos = nx.spring_layout(graph) 
    weights = nx.get_edge_attributes(graph, 'weight')
    
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    
    plt.title("Sentence Similarity Graph")
    plt.show()  

def graph_search(query, document_text, model):
    sentences = split_sentences(document_text)
    
    graph = build_graph(sentences, model)

    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    
    best_idx = similarities.argmax().item()

    neighbors = list(graph.neighbors(best_idx))
    
    best_sentence = sentences[best_idx]
    related_sentences = [sentences[n] for n in neighbors]

    return best_sentence, related_sentences, graph, similarities[best_idx]
    
def generate_answer(query, context):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    

    # tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("vinai/PhoGPT-4B", trust_remote_code=True)

    # model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)  # load checkpoint
     
    # tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")    #GPU to run
    # model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")  #GPU to run

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_text = f"Query: {query} \nDocument: {context} \nAnswer:"
    # input_text = f"{query}"

    # input_text = f"Please answer the user's question using information extracted from the document below. Ensure your answer is accurate and relevant to the query. \n\nUser Question: {query} \n\nDocument: {context} \n\nAnswer:"

    
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='longest')

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=1025,
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
