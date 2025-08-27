import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_openai import ChatOpenAI

# Configure Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_compat_client import LLMClient, OpenAICompatClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from openai import AsyncOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
import ipywidgets as widgets
from dotenv import load_dotenv
from IPython.display import Image, display
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from typing_extensions import TypedDict


os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_PROJECT'] = 'Graphiti LangGraph Tutorial'
load_dotenv()
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()



neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

DEEPSEEK_API_KEY = os.environ.get('deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_Model: str = 'deepseek-chat'

# 配置智谱AI的API密钥和基础URL
ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY')  # 从环境变量获取API密钥
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # 智谱AI的OpenAI兼容接口地址
ZHIPU_MODEL = "glm-3-turbo"  # 可替换为其他模型，如glm-4-flash、glm-3-turbo等

QWEN_API_KEY = 'sk-e6c95d2c8d7c441c86e0c5c7f5f4d81b'  # 从环境变量获取API密钥
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def edges_to_facts_string(entities: list[EntityEdge]):
    return '-' + '\n- '.join([edge.fact for edge in entities])


async def ingest_products_data(client: Graphiti):
    script_dir = Path.cwd().parent
    json_file_path = script_dir / 'data' / 'manybirds_products.json'

    with open(json_file_path) as file:
        products = json.load(file)['products']

    for i, product in enumerate(products):
        await client.add_episode(
            name=product.get('title', f'Product {i}'),
            episode_body=str({k: v for k, v in product.items() if k != 'images'}),
            source_description='ManyBirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )

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

# 定义一个异步函数专门用于获取 client
async def get_client():
    return await init_llm()


async def main():
    client = await get_client()
    nl = await client._search('ManyBirds', NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    manybirds_node_uuid = nl.nodes[0].uuid
    print(manybirds_node_uuid)
    @tool
    async def get_shoe_data(query: str) -> str:
        """Search the graphiti graph for information about shoes"""
        edge_results = await client.search(
            query,
            center_node_uuid=manybirds_node_uuid,
            num_results=10,
        )
        return edges_to_facts_string(edge_results)## 将查询结果转换为自然语言字符串返回

    tools = [get_shoe_data]
    tool_node = ToolNode(tools)
    # await ingest_products_data(client)
    user_name = 'jess'
    # 存储数据，存储过就不用存了
    await client.add_episode(
        name='User Creation',
        episode_body=(f'{user_name} is interested in buying a pair of shoes'),
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
        source_description='SalesBot',
    )
    # 初始化Qwen模型（核心修改部分）
    llm = ChatOpenAI(
        model_name=QWEN_MODEL,
        temperature=0,
        openai_api_key=QWEN_API_KEY,
        openai_api_base=QWEN_BASE_URL
    ).bind_tools(tools)# 关键：将工具绑定到模型上

    # 执行查询
    # llm.ainvoke('wool shoes')：Qwen 模型接收 “羊毛鞋” 这个查询，分析是否需要调用工具。
    # 工具节点接收模型的输出:如果是工具调用指令，tool_node会自动执行对应的工具（get_shoe_data），传入查询参数。
    # 最终，模型会结合工具返回的结果，生成更精准的回答
    result = await tool_node.ainvoke({'messages': [await llm.ainvoke('wool shoes')]})
    # 提取文字内容
    if result.get('messages'):  # 确保 messages 列表存在
        tool_message = result['messages'][0]  # 获取第一个消息
        # 对于 ToolMessage 对象，直接通过 .content 属性访问
        text_content = tool_message.content if hasattr(tool_message, 'content') else '无内容'
        print(text_content)


if __name__ == '__main__':
    asyncio.run(main())