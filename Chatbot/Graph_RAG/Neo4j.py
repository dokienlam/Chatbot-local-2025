from neo4j import GraphDatabase

uri = 'bolt://35.171.28.102:7687'
username = "neo4j"  
password = "drawers-sex-widths"

driver = GraphDatabase.driver(uri, auth=(username, password))

def add_to_neo4j(graph, driver):
    with driver.session() as session:
        for node in graph.nodes(data=True):
            entity = node[0]
            entity_type = node[1].get("entity_type", "Unknown")
            session.run("MERGE (e:Entity {name: $name, type: $type})", name=entity, type=entity_type)

        for u, v, weight in graph.edges(data=True):
            session.run("""
                MATCH (e1:Entity {name: $name1}), (e2:Entity {name: $name2})
                CREATE (e1)-[:SIMILARITY {weight: $weight}]->(e2)
            """, name1=u, name2=v, weight=weight['weight'])