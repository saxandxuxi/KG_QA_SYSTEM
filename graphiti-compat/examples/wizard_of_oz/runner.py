"""
Copyright 2024, Zep Software, Inc.

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

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from examples.wizard_of_oz.parser import get_wizard_of_oz_messages
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_compat_client import LLMClient,OpenAICompatClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'
DEEPSEEK_API_KEY = os.environ.get('deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_Model: str = 'deepseek-chat'

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


async def main():
    setup_logging()
    # llm_client = AnthropicClient(LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY')))
    # client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client)
    llmconfig = LLMConfig(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_Model)
    llm_client = OpenAICompatClient(llmconfig)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="abc",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            base_url="http://localhost:11434/v1",
        )
    )

    # Initialize Graphiti
    client = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "12345678",
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llmconfig),
    )

    messages = get_wizard_of_oz_messages()
    print(messages[0])
    print(len(messages))
    now = datetime.now(timezone.utc)
    # episodes: list[BulkEpisode] = [
    #     BulkEpisode(
    #         name=f'Chapter {i + 1}',
    #         content=chapter['content'],
    #         source_description='Wizard of Oz Transcript',
    #         episode_type='string',
    #         reference_time=now + timedelta(seconds=i * 10),
    #     )
    #     for i, chapter in enumerate(messages[0:50])
    # ]

    # await clear_data(client.driver)
    # await client.build_indices_and_constraints()
    # await client.add_episode_bulk(episodes)

    # await clear_data(client.driver)
    await client.build_indices_and_constraints()
    for i, chapter in enumerate(messages):
        if i < 2 :
            continue
        await client.add_episode(
            name=f'Chapter {i + 1}',
            episode_body=chapter['content'],
            source_description='Wizard of Oz Transcript',
            reference_time=now + timedelta(seconds=i * 10),
        )


asyncio.run(main())
