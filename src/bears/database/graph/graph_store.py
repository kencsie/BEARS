"""
Graph store manager.

Handles all Neo4j graph database operations.
"""

import logging
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from bears.core.config import get_settings

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Entity model."""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (e.g. PERSON/ORGANIZATION/LOCATION)")


class Relationship(BaseModel):
    """Relationship model."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type")
    description: str = Field(description="Brief description of the relationship")


class GraphStoreManager:
    """Neo4j graph store manager."""

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None
    ):
        settings = get_settings()
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD

        self.graph = Neo4jGraph(
            url=self.uri,
            username=self.username,
            password=self.password
        )

        logger.info("GraphStoreManager initialized")

    def add_entity(self, entity: Entity, doc_id: str = None):
        """Add entity to graph."""
        try:
            cypher_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type, e.doc_id = $doc_id
            ON MATCH SET e.doc_id = $doc_id
            """
            self.graph.query(
                cypher_query,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "doc_id": doc_id
                }
            )
            logger.debug(f"Added entity: {entity.name}")
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            raise

    def add_relationship(self, relationship: Relationship, doc_id: str = None):
        """Add relationship to graph."""
        try:
            rel_type = relationship.type.upper().replace(" ", "_")

            cypher_query = f"""
            MERGE (source:Entity {{name: $source}})
            MERGE (target:Entity {{name: $target}})
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET r.description = $description, r.doc_id = $doc_id
            """

            self.graph.query(
                cypher_query,
                {
                    "source": relationship.source,
                    "target": relationship.target,
                    "description": relationship.description,
                    "doc_id": doc_id
                }
            )
            logger.debug(f"Added relationship: {relationship.source} -[{rel_type}]-> {relationship.target}")
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            raise

    def query_entity(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query all relationships of an entity."""
        try:
            cypher_query = """
            MATCH (e:Entity {name: $name})-[r]-(neighbor)
            RETURN e.name as entity, type(r) as relationship, neighbor.name as neighbor
            LIMIT $limit
            """

            results = self.graph.query(
                cypher_query,
                {
                    "name": entity_name,
                    "limit": limit
                }
            )

            return results
        except Exception as e:
            logger.error(f"Failed to query entity: {e}")
            return []

    def query_cypher(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query."""
        try:
            results = self.graph.query(cypher_query, params or {})
            return results
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    def clear_all(self):
        """Clear all nodes and relationships (dangerous)."""
        try:
            logger.warning("Clearing Neo4j graph database")
            cypher_query = "MATCH (n) DETACH DELETE n"
            self.graph.query(cypher_query)
            logger.info("Neo4j graph database cleared")
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            node_count_query = "MATCH (n) RETURN count(n) as count"
            node_result = self.graph.query(node_count_query)
            node_count = node_result[0]["count"] if node_result else 0

            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_result = self.graph.query(rel_count_query)
            rel_count = rel_result[0]["count"] if rel_result else 0

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "database": "Neo4j"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
