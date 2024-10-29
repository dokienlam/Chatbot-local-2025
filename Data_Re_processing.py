from langchain.text_splitter import CharacterTextSplitter
import docx
from keybert import KeyBERT

def split_sentences(document_text):
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
#     )
    text_splitter = CharacterTextSplitter(separator='.', chunk_size=557, chunk_overlap=0)
    sentences = text_splitter.split_text(document_text)
    return [s.strip() for s in sentences if s.strip()]
    
def read_word_file(file_path):
    doc = docx.Document(file_path)
    content = []
    
    for para in doc.paragraphs:
        content.append(para.text)
    return str("\n".join(content))

def extract_keywords( answer):
    kw_model = KeyBERT()
    answer_keywords = kw_model.extract_keywords(answer, top_n=10, stop_words='english')
    answer_keywords = [kw[0] for kw in answer_keywords]
    combined_keywords = list(set(answer_keywords))
    return combined_keywords

def extract_entities(sentence, nlp):
    doc = nlp(sentence)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_  
    return entities