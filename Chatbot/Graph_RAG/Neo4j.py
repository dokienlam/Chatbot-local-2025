from neo4j import GraphDatabase

uri = 'bolt://35.171.28.102:7687'
username = "neo4j"  
password = "drawers-sex-widths"

driver = GraphDatabase.driver(uri, auth=(username, password))

def add_to_neo4j(sentences, graph):
    with driver.session() as session:
        for i, sentence in enumerate(sentences):
            session.run("CREATE (s:Sentence {id: $id, text: $text})", id=i, text=sentence)

        for (u, v, weight) in graph.edges(data=True):
            session.run("""
                MATCH (s1:Sentence {id: $id1}), (s2:Sentence {id: $id2})
                CREATE (s1)-[:SIMILARITY {weight: $weight}]->(s2)
            """, id1=u, id2=v, weight=weight['weight'])
