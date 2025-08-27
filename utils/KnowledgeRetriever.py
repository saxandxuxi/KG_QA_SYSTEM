from langchain_chroma import Chroma
from Config.config import Config, Logger
from Import_KG.import_kg import KnowledgeBaseBuilder


class KnowledgeRetriever:
    def __init__(self, knowledge_base: KnowledgeBaseBuilder, logger):
        """
        初始化知识检索器

        Args:
            knowledge_base (Chroma): 已加载的Chroma向量库实例
            logger: 日志记录器
        """
        self.knowledge_base = knowledge_base.load_existing_knowledge_base()#加载本地向量库
        self.logger = logger

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> list:
        """
        从知识库中检索与查询相关的文档

        Args:
            query (str): 用户查询
            top_k (int): 检索返回的最大文档数

        Returns:
            list: 相关文档内容列表，每个元素包含文档内容和元数据
        """
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        try:
            # 执行相似性检索
            results = self.knowledge_base.similarity_search_with_score(query, k=top_k)

            # 提取文档内容和相似度分数
            relevant_docs = []
            for doc, score in results:
                relevant_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })

            self.logger.info(f"成功检索到 {len(relevant_docs)} 个相关文档")
            return relevant_docs

        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            raise

    def retrieve_relevant_docs_with_threshold(self, query: str, top_k: int = 3, threshold: float = 0.75) -> list:
        """
        从知识库中检索与查询相关的文档，并设置相似度阈值

        Args:
            query (str): 用户查询
            top_k (int): 检索返回的最大文档数
            threshold (float): 相似度阈值，低于此值的结果将被过滤掉

        Returns:
            list: 相关文档内容列表，每个元素包含文档内容、元数据和相似度分数
        """
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        if not (0 <= threshold <= 1):
            raise ValueError("threshold 必须在0到1之间")

        try:
            # 执行相似性检索
            results = self.knowledge_base.similarity_search_with_score(query, k=top_k)

            # 根据阈值过滤结果并提取文档信息
            relevant_docs = []
            for doc, score in results:
                # 转换为相似度分数 (score是距离，越小越相似)
                similarity = 1 - score
                if similarity >= threshold:
                    relevant_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity  # 返回相似度而不是距离
                    })

            self.logger.info(f"成功检索到 {len(relevant_docs)} 个相关文档 (阈值: {threshold})")
            return relevant_docs

        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            raise

    def get_knowledge_base_info(self) -> dict:
        """
        获取知识库信息

        Returns:
            dict: 包含知识库信息的字典
        """
        try:
            # 获取向量库中的文档数量
            collection = self.knowledge_base._collection
            doc_count = collection.count()

            info = {
                "document_count": doc_count,
                "persist_directory": self.knowledge_base._persist_directory
            }

            self.logger.info("成功获取知识库信息")
            return info

        except Exception as e:
            self.logger.error(f"获取知识库信息失败: {e}")
            raise

    def format_retrieved_docs(self, retrieved_docs: list) -> str:
        """
        将检索到的文档内容格式化为带编号的字符串

        Args:
            retrieved_docs (list): 检索到的文档列表

        Returns:
            str: 格式化后的文档内容字符串
        """
        if not retrieved_docs:
            return ""

        formatted_docs = []
        for i, doc in enumerate(retrieved_docs, 1):
            if isinstance(doc, dict) and "content" in doc:
                # 如果是包含content键的字典（来自retrieve_relevant_docs方法）
                formatted_docs.append(f"{i}.\n{doc['content']}")
            elif hasattr(doc, 'page_content'):
                # 如果是Document对象
                formatted_docs.append(f"{i}.\n{doc.page_content}")
            else:
                # 如果是其他类型的对象或字符串
                formatted_docs.append(f"{i}.\n{str(doc)}")

        return "\n\n".join(formatted_docs)



# 在主程序中使用
if __name__ == "__main__":


    # 初始化配置和日志
    config = Config()
    logger = Logger()

    # 加载知识库（使用KnowledgeBaseBuilder）
    knowledge_base = KnowledgeBaseBuilder(Config(), Logger())
    # knowledge_base = builder.load_existing_knowledge_base()

    # 创建检索器实例
    retriever = KnowledgeRetriever(knowledge_base, logger)

    # 基础检索
    docs = retriever.retrieve_relevant_docs("DGGS", top_k=10)
    print(retriever.format_retrieved_docs(docs))

    # 带阈值的检索
    filtered_docs = retriever.retrieve_relevant_docs_with_threshold(
        "GeoJSON zone list",
        top_k=5,
        threshold=0.7
    )

    # 获取知识库信息
    info = retriever.get_knowledge_base_info()
    print(f"知识库信息: {info}")
