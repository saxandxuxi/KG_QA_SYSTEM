import json
from typing import List, Dict, Any, Optional
from graphiti_core.llm_client.openai_compat_client import OpenAICompatClient
from graphiti_core.cross_encoder.openai_compat_reranker_client import OpenAICompatRerankerClient
from Import_KG.Graph_kg_base import GraphKnowledgeBase
from graphiti_core import Graphiti
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


class Graph_KG_Retriever:
    """
    知识图谱检索器类，提供多种检索方式并格式化检索结果
    """

    def __init__(self, GraphBase: GraphKnowledgeBase, logger: Any):
        """
        初始化知识图谱检索器

        Args:
            GraphBase: Graphiti实例
            logger: 日志记录器
        """
        self.GraphBase = GraphBase
        self.graphiti = self.GraphBase.graphiti
        self.logger = logger

    async def basic_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        基础搜索，返回边和节点

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含边和节点的字典
        """
        try:
            results = await self.GraphBase.basic_search(query, limit=limit)
            self.logger.info(f"基础搜索完成，找到{len(results)}个结果")

            return {
                "edges": results,
                "nodes": []  # 基础搜索主要返回边
            }
        except Exception as e:
            self.logger.error(f"基础搜索失败: {e}")
            raise

    async def center_node_search(self, query: str, center_node_uuid: str, limit: int = 10) -> Dict[str, Any]:
        """
        中心节点搜索，基于图距离重排序结果

        Args:
            query: 搜索查询
            center_node_uuid: 中心节点UUID
            limit: 返回结果数量限制

        Returns:
            包含边和节点的字典
        """
        try:
            results = await self.graphiti.search(
                query=query,
                center_node_uuid=center_node_uuid,
                num_results=limit
            )
            self.logger.info(f"中心节点搜索完成，找到{len(results)}个结果")

            return {
                "edges": results,
                "nodes": []
            }
        except Exception as e:
            self.logger.error(f"中心节点搜索失败: {e}")
            raise

    async def node_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        节点搜索，专门检索节点（实体）

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含节点的字典
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

            return {
                "edges": [],
                "nodes": results.nodes
            }
        except Exception as e:
            self.logger.error(f"节点搜索失败: {e}")
            raise

    async def combined_hybrid_search_rrf(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        执行混合搜索，并在边缘、节点和社区上重新排序 RRF

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含边和节点的字典
        """
        try:
            combined_search_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            combined_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=combined_search_config
            )

            self.logger.info(
                f"COMBINED_HYBRID_SEARCH_RRF搜索完成，找到{len(results.edges)}个边和{len(results.nodes)}个节点")

            return {
                "edges": results.edges,
                "nodes": results.nodes
            }
        except Exception as e:
            self.logger.error(f"COMBINED_HYBRID_SEARCH_RRF搜索失败: {e}")
            raise

    async def combined_hybrid_search_mmr(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        执行混合搜索，并在边缘、节点和社区上重新排序 MMR

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含边和节点的字典
        """
        try:
            combined_search_config = COMBINED_HYBRID_SEARCH_MMR.model_copy(deep=True)
            combined_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=combined_search_config
            )

            self.logger.info(
                f"COMBINED_HYBRID_SEARCH_MMR搜索完成，找到{len(results.edges)}个边和{len(results.nodes)}个节点")

            return {
                "edges": results.edges,
                "nodes": results.nodes
            }
        except Exception as e:
            self.logger.error(f"COMBINED_HYBRID_SEARCH_MMR搜索失败: {e}")
            raise

    async def edge_hybrid_search_node_distance(self, query: str, center_node_uuid,limit: int = 10) -> Dict[str, Any]:
        """
        使用节点距离重新排序对边执行混合搜索

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含边的字典
        """
        try:
            edge_search_config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
            edge_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=edge_search_config,
                center_node_uuid=center_node_uuid
            )

            self.logger.info(f"EDGE_HYBRID_SEARCH_NODE_DISTANCE搜索完成，找到{len(results.edges)}个边")

            return {
                "edges": results.edges,
                "nodes": []
            }
        except Exception as e:
            self.logger.error(f"EDGE_HYBRID_SEARCH_NODE_DISTANCE搜索失败: {e}")
            raise

    async def edge_hybrid_search_episode_mentions(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        通过剧集提及重新排序对边缘执行混合搜索

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含边的字典
        """
        try:
            edge_search_config = EDGE_HYBRID_SEARCH_EPISODE_MENTIONS.model_copy(deep=True)
            edge_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=edge_search_config
            )

            self.logger.info(f"EDGE_HYBRID_SEARCH_EPISODE_MENTIONS搜索完成，找到{len(results.edges)}个边")

            return {
                "edges": results.edges,
                "nodes": []
            }
        except Exception as e:
            self.logger.error(f"EDGE_HYBRID_SEARCH_EPISODE_MENTIONS搜索失败: {e}")
            raise

    async def node_hybrid_search_node_distance(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        对节点执行混合搜索，并重新排序节点距离

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含节点的字典
        """
        try:
            node_search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
            node_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=node_search_config
            )

            self.logger.info(f"NODE_HYBRID_SEARCH_NODE_DISTANCE搜索完成，找到{len(results.nodes)}个节点")

            return {
                "edges": [],
                "nodes": results.nodes
            }
        except Exception as e:
            self.logger.error(f"NODE_HYBRID_SEARCH_NODE_DISTANCE搜索失败: {e}")
            raise

    async def node_hybrid_search_episode_mentions(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        对节点执行混合搜索，并重新对剧集提及进行重新排名

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            包含节点的字典
        """
        try:
            node_search_config = NODE_HYBRID_SEARCH_EPISODE_MENTIONS.model_copy(deep=True)
            node_search_config.limit = limit

            results = await self.graphiti._search(
                query=query,
                config=node_search_config
            )

            self.logger.info(f"NODE_HYBRID_SEARCH_EPISODE_MENTIONS搜索完成，找到{len(results.nodes)}个节点")

            return {
                "edges": [],
                "nodes": results.nodes
            }
        except Exception as e:
            self.logger.error(f"NODE_HYBRID_SEARCH_EPISODE_MENTIONS搜索失败: {e}")
            raise





    def format_edge_results(self, edges: List[EntityEdge]) -> str:
        """
        格式化边检索结果为带编号的字符串

        Args:
            edges: 边列表

        Returns:
            格式化后的字符串
        """
        if not edges:
            return "未找到相关关系。"

        formatted_results = []
        for i, edge in enumerate(edges, 1):
            result_str = f"{i}. 关系: {edge.fact}\n"
            # if hasattr(edge, 'valid_at') and edge.valid_at:
            #     result_str += f"   有效期开始: {edge.valid_at}\n"
            # if hasattr(edge, 'invalid_at') and edge.invalid_at:
            #     result_str += f"   有效期结束: {edge.invalid_at}\n"
            # result_str += f"   UUID: {edge.uuid}"
            formatted_results.append(result_str)

        return "\n\n".join(formatted_results)

    def format_node_results(self, nodes: List[EntityNode]) -> str:
        """
        格式化节点检索结果为带编号的字符串

        Args:
            nodes: 节点列表

        Returns:
            格式化后的字符串
        """
        if not nodes:
            return "未找到相关实体。"

        formatted_results = []
        for i, node in enumerate(nodes, 1):
            result_str = f"{i}. 实体: {node.name}\n"
            node_summary = node.summary
            result_str += f"   描述: {node_summary}\n"
            result_str += f"   类型: {', '.join(node.labels) if node.labels else 'N/A'}\n"
            # result_str += f"   UUID: {node.uuid}"

            # 添加属性信息（如果存在）
            # if hasattr(node, 'attributes') and node.attributes:
            #     result_str += "\n   属性:"
            #     for key, value in node.attributes.items():
            #         result_str += f"\n     {key}: {value}"

            formatted_results.append(result_str)

        return "\n\n".join(formatted_results)

    def format_combined_results(self, search_results: Dict[str, Any]) -> str:
        """
        格式化组合检索结果（边和节点）为带编号的字符串

        Args:
            search_results: 包含edges和nodes的字典

        Returns:
            格式化后的字符串
        """
        edges = search_results.get("edges", [])
        nodes = search_results.get("nodes", [])

        result_parts = []

        # 格式化边结果
        if edges:
            result_parts.append("关系:")
            result_parts.append(self.format_edge_results(edges))

        # 格式化节点结果
        if nodes:
            result_parts.append("实体:")
            result_parts.append(self.format_node_results(nodes))

        if not result_parts:
            return "未找到相关结果。"

        return "\n\n".join(result_parts)


# 完整的测试示例（需要提供真实的依赖项）
if __name__ == "__main__":
    import asyncio
    from Config.config import Logger
    from Import_KG.Graph_kg_base import GraphKnowledgeBase
    from Config.config import Config


    async def test_graph_kg_retriever():
        # 初始化配置和日志
        config = Config()
        logger = Logger()

        # 初始化知识图谱知识库
        graph_base = GraphKnowledgeBase(config, logger)
        await graph_base.initialize()

        # 初始化知识图谱检索器
        graph_kg_retriever = Graph_KG_Retriever(graph_base, logger)

        # 测试查询
        test_queries = [

            "DGGS的核心数据模型",

        ]

        # 测试基础搜索功能
        print("=== 基础搜索测试 ===")
        for query in test_queries:
            print(f"查询：{query}")
            try:
                results = await graph_kg_retriever.basic_search(query, limit=5)
                # formatted_results = graph_kg_retriever.format_combined_results(results)
                # print(f"结果：\n{formatted_results}\n")
            except Exception as e:
                print(f"搜索失败: {e}\n")
            print("=== 边搜索测试 ===")
            center_node_uuid = results['edges'][0].source_node_uuid
            results = await graph_kg_retriever.edge_hybrid_search_episode_mentions(query,limit=3)
            formatted_results = graph_kg_retriever.format_combined_results(results)
            print(f"结果：\n{formatted_results}\n")
            print("=== 节点搜索测试 ===")
            results = await graph_kg_retriever.node_hybrid_search_episode_mentions(query, limit=3 )
            formatted_results = graph_kg_retriever.format_combined_results(results)
            print(f"结果：\n{formatted_results}\n")
            print("=== 组合搜索测试 ===")
            results = await graph_kg_retriever.combined_hybrid_search_rrf(query, limit=3)
            formatted_results = graph_kg_retriever.format_combined_results(results)
            print(f"结果：\n{formatted_results}\n")








    # 运行测试
    asyncio.run(test_graph_kg_retriever())

