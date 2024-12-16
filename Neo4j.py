from neo4j import GraphDatabase

uri = 'bolt://44.200.216.167:7687'
username = "neo4j"  
password = "snap-anthems-incentives"

driver = GraphDatabase.driver(uri, auth=(username, password), )

def add_to_neo4j(driver, sentences):
    def create_sentences(tx, sentences):
        for sentence in sentences:
            tx.run("CREATE (s:Sentence {content: $content})", content=sentence)

    with driver.session() as session:
        session.write_transaction(create_sentences, sentences)

