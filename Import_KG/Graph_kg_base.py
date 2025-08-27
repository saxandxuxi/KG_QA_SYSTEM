import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
# 将 graphiti-compat 目录添加到 Python 路径
# 在 Graph_kg_base.py 文件开头添加
import sys
import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.config import Config
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
    NODE_HYBRID_SEARCH_CROSS_ENCODER
)
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_compat_client import OpenAICompatClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

import nltk
# 下载 punkt_tab 资源
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
class GraphKnowledgeBase:
    """
    基于Graphiti的知识图谱知识库类，用于构建和检索知识图谱中的信息
    """

    def __init__(self, config: Any, logger: logging.Logger):
        """
        初始化知识图谱知识库

        Args:
            config: 配置对象，包含数据库连接和模型配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.graphiti: Optional[Graphiti] = None
        self.connected = False

    async def initialize(self):
        """
        初始化Graphiti客户端并建立连接
        """
        try:
            # 配置LLM客户端
            llm_config = LLMConfig(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
                temperature = 0.1
            )
            llm_client = OpenAICompatClient(llm_config)

            # 配置嵌入模型
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self.config.embedding_api_key,
                    embedding_model=self.config.embedding_model_name,
                    embedding_dim=self.config.embedding_dim,
                    base_url=self.config.embedding_base_url,
                )
            )

            graph_driver = Neo4jDriver(
                uri=self.config.neo4j_uri,  # 数据库URI
                user=self.config.neo4j_user,  # 正确的用户名（默认是'neo4j'）
                password=self.config.neo4j_password,  # 你的Neo4j密码（确保与数据库密码一致）
                database=self.config.neo4j_database  # 目标数据库名
            )
            # 初始化Graphiti
            self.graphiti = Graphiti(
                graph_driver=graph_driver,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)
            )

            # 建立索引和约束
            await self.graphiti.build_indices_and_constraints()
            self.connected = True
            self.logger.info("成功初始化知识图谱知识库")

        except Exception as e:
            self.logger.error(f"初始化知识图谱知识库失败: {e}")
            raise

    async def add_episode(
            self,
            name: str,
            content: str,
            source_description: str = "",
            reference_time: datetime = None,

    ):
        """
        向知识图谱中添加一个episode（片段）

        Args:
            name: episode名称
            content: episode内容
            source_description: 来源描述
            reference_time: 参考时间

        """


        try:
            reference_time = reference_time or datetime.now(timezone.utc)

            await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source_description,
                reference_time=reference_time,
                )

            self.logger.info(f"成功添加episode: {name}")

        except Exception as e:
            self.logger.error(f"添加episode失败: {e}")
            raise

    async def basic_search(
            self,
            query: str,
            limit: int = 10

    ) -> List[EntityNode]:
        results = await self.graphiti.search(query,num_results=limit)
        self.logger.info(f"搜索完成，找到{len(results)}个结果")
        return results

    async def search_edges(
            self,
            query: str,
            center_node_uuid: str = None,
            limit: int = 10
    ) -> List[EntityEdge]:
        """
        搜索知识图谱中的边（关系/事实）

        Args:
            query: 搜索查询
            center_node_uuid: 中心节点UUID，用于基于图距离的重排序
            limit: 返回结果数量限制

        Returns:
            EntityEdge列表
        """


        try:
            # 使用默认搜索配置
            results = await self.graphiti.search(
                query=query,
                center_node_uuid=center_node_uuid,
                num_results=limit
            )

            self.logger.info(f"边搜索完成，找到{len(results)}个结果")
            return results

        except Exception as e:
            self.logger.error(f"边搜索失败: {e}")
            raise

    async def search_nodes(
            self,
            query: str,
            limit: int = 10
    ) -> List[EntityNode]:
        """
        搜索知识图谱中的节点（实体）

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            EntityNode列表
        """

        try:
            # 使用节点混合搜索配置
            node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=node_search_config
            )

            self.logger.info(f"节点搜索完成，找到{len(results.nodes)}个结果")
            return results.nodes

        except Exception as e:
            self.logger.error(f"节点搜索失败: {e}")
            raise



    def load_and_split_documents(self):
        """
        加载并分割PDF文档

        Returns:
            list: 分割后的文档列表，每个元素包含文档名和片段内容
        """
        all_documents = []

        try:
            loader = DirectoryLoader(
                path=self.config.data_dir,
                glob="**/*.pdf",
                loader_cls=UnstructuredPDFLoader,
                recursive=True
            )
            documents = loader.load()
            all_documents.extend(documents)
            self.logger.info(f"成功加载{len(documents)}个PDF文件")
        except Exception as e:
            self.logger.error(f"加载PDF文件时出错: {e}")
            raise

        if not all_documents:
            raise ValueError("未加载到任何PDF文档，请检查文档路径和格式")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        split_docs = text_splitter.split_documents(all_documents)

        # 添加文件名信息到每个分割文档
        enriched_docs = []
        for doc in split_docs:
            source_path = doc.metadata.get('source', 'unknown')
            # 提取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(source_path))[0]
            enriched_docs.append({
                'filename': filename,
                'source_path': source_path,
                'document': doc
            })

        return enriched_docs

    async def _add_episode_with_retry(
            self,
            name: str,
            content: str,
            source_description: str,
            max_retries: int = 2
    ):
        """
        添加episode并包含重试机制

        Args:
            name: episode名称
            content: episode内容
            source_description: 来源描述
            max_retries: 最大重试次数
        """
        retries = 0
        while retries < max_retries:
            try:
                await self.add_episode(
                    name=name,
                    content=content,
                    source_description=source_description
                )
                self.logger.info(f"成功添加episode: {name}")
                return True
            except Exception as e:
                retries += 1
                self.logger.warning(f"添加episode失败 (尝试 {retries}/{max_retries}): {name}, 错误: {e}")

                # 检查是否是速率限制错误
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    # 对于速率限制错误，增加等待时间
                    wait_time = 15  # 固定等待60秒
                    self.logger.warning(f"遇到速率限制，等待{wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)

                if retries >= max_retries:
                    self.logger.error(f"添加episode最终失败: {name}")
                    raise
                else:
                    # 对于非速率限制错误，使用指数退避
                    if not ("rate limit" in error_str or "429" in error_str):
                        wait_time = min(2 ** retries, 5)  # 最多等待30秒
                        self.logger.info(f"等待{wait_time}秒后重试...")
                        await asyncio.sleep(wait_time)

    async def add_documents_from_directory(self):
        """
        从配置的目录加载文档，进行文本分割，并将每个分割后的文档片段作为episode添加到知识图谱中
        参考parser.py的处理方式，将文档内容分块处理
        """
        try:
            # 使用与KnowledgeBaseBuilder相同的文档加载和分割逻辑
            split_docs = self.load_and_split_documents()

            # 将分割后的文档添加为episode
            added_count = 0
            start_index = 0
            skipped_count = 0
            # 从start_index位置开始处理文档，但索引从0开始
            for i, doc_info in enumerate(split_docs[start_index:]):
                filename = doc_info['filename']
                source_path = doc_info['source_path']
                doc = doc_info['document']

                episode_name = f"{filename}_Part_{i + start_index}"
                source_description = source_path

                # 使用带重试机制的函数添加episode
                try:
                    await self._add_episode_with_retry(
                        name=episode_name,
                        content=doc.page_content,
                        source_description=source_description
                    )
                    added_count += 1
                except Exception as e:
                    # 如果重试次数用完仍然失败，跳过这个片段并记录
                    self.logger.error(f"跳过episode {episode_name}，错误: {e}")
                    skipped_count += 1
                    continue  # 继续处理下一个片段


            self.logger.info(f"成功从目录添加 {added_count} 个文档片段到知识图谱")
            return added_count

        except Exception as e:
            self.logger.error(f"从目录添加文档失败: {e}")
            raise

    async def close(self):
        """
        关闭知识库连接
        """
        if self.graphiti:
            await self.graphiti.close()
            self.connected = False
            self.logger.info("知识图谱知识库连接已关闭")





async def main():
    """
    主函数，演示如何使用GraphKnowledgeBase类
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    # 创建配置对象
    config = Config()

    # 创建知识库实例
    knowledge_base = GraphKnowledgeBase(config, logger)

    try:
        # 初始化知识库,只用运行一次
        await knowledge_base.initialize()

        # 添加示例episode
        # await knowledge_base.add_documents_from_directory()

    except Exception as e:
        logger.error(f"执行过程中出错: {e}")
    finally:
        # 关闭连接
        await knowledge_base.close()


if __name__ == "__main__":
    asyncio.run(main())
