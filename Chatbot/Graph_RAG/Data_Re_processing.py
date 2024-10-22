import re
import docx
from langchain.text_splitter import CharacterTextSplitter
from keybert import KeyBERT


def split_sentences(document_text):
    text_splitter = CharacterTextSplitter(separator=r'\.', chunk_size=300, chunk_overlap=10)
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
    
    answer_keywords = kw_model.extract_keywords(answer, top_n=100, stop_words='english')

    answer_keywords = [kw[0] for kw in answer_keywords]

    combined_keywords = list(set(answer_keywords))

    return combined_keywords