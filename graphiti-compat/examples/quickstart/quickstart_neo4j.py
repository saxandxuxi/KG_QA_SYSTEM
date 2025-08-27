"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import openai
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO
from dotenv import load_dotenv
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
# # from graphiti_core.llm_client.config import LLMConfig
# from graphiti_core.llm_client.openai_client import OpenAIClient
# from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
# from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
# from graphiti_core.llm_client.openai_compat_client import LLMClient,OpenAICompatClient
from ennity_type import *
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_compat_client import LLMClient,OpenAICompatClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', "bolt://localhost:7687")
neo4j_user = os.environ.get('NEO4J_USER', "neo4j")
neo4j_password = os.environ.get('NEO4J_PASSWORD', "12345678")
DEEPSEEK_API_KEY = os.environ.get('deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_Model: str = 'deepseek-chat'

QWEN_API_KEY = 'sk-e6c95d2c8d7c441c86e0c5c7f5f4d81b'  # 从环境变量获取API密钥
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 智谱AI的OpenAI兼容接口地址
QWEN_MODEL = "qwen-plus"
if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

from graphiti_core.driver.neo4j_driver import Neo4jDriver

async def init_llm():
    llmconfig = LLMConfig(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, model=QWEN_MODEL)
    llm_client = OpenAICompatClient(llmconfig)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="abc",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            base_url="http://localhost:11434/v1",
        )
    )

    # 如果不是默认的数据库，一定要指定！！！！
    graph_driver = Neo4jDriver(
        uri="bolt://localhost:7687",  # 数据库URI
        user="neo4j",  # 正确的用户名（默认是'neo4j'）
        password="12345678",  # 你的Neo4j密码（确保与数据库密码一致）
        database="product"  # 目标数据库名
    )
    # Initialize Graphiti with Ollama clients
    client = Graphiti(
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llmconfig),
        graph_driver=graph_driver
    )

    # await clear_data(client.driver)!!!!!!!!!!!!!慎用
    await client.build_indices_and_constraints()
    return client



async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Ollama clients
    graphiti = await init_llm()
    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        # await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',# 来源描述（播客文字稿）
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]
        # 实体类型映射
        entity_types = {
            "Person": Person,
            "GovernmentPosition": GovernmentPosition,
            "PoliticalEntity": PoliticalEntity
        }

        # 关系类型映射
        edge_types = {
            "HoldsPosition": HoldsPosition,
            "PreviouslyHeld": PreviouslyHeld,
            "Governs": Governs
        }

        # 实体间关系映射
        edge_type_map = {
            ("Person", "GovernmentPosition"): ["HoldsPosition", "PreviouslyHeld"],
            ("Person", "PoliticalEntity"): ["Governs"],
            ("GovernmentPosition", "PoliticalEntity"): ["HoldsPosition"]
        }

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
                entity_types=entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who has served as the District Attorney of San Francisco?'")
        results = await graphiti.search('"Who has served as the District Attorney of San Francisco?"')

        # Print search results
        print('\n-------------------------Search Results----------------------------------------------------:')
        for result in results:#回复有三个值
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\n---------------------------------------Reranking search results based on graph distance:-----------------------------')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                "Who has served as the District Attorney of San Francisco?", center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\n------------------------------Reranked Search Results:---------------------------------------------')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\n------------Performing node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:------------------'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\n------------------Node Search Results:------------')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
