import asyncio
from typing import Union, Optional, List
from Import_KG.Graph_kg_base import GraphKnowledgeBase
from Import_KG.import_kg import KnowledgeBaseBuilder
from Prompt.QA_Prompt import get_decomposition_template, get_final_rag_template, get_rag_template, get_prompt_template
from utils.KnowledgeRetriever import KnowledgeRetriever
from utils.Graph_KG_Retriever import Graph_KG_Retriever
from transformers import pipeline
import re
import jieba
import jieba.posseg as pseg
from utils.call_gpt import call_gpt_async
from langchain.prompts import ChatPromptTemplate
from Config.config import Config, Logger


class QASystem:
    """
    问答系统类，支持中英文问答，能够检索向量库和知识图谱
    """

    def __init__(self, knowledge_base: KnowledgeBaseBuilder, graphiti: GraphKnowledgeBase, logger, config: Config):
        """
        初始化问答系统

        Args:
            knowledge_base (KnowledgeBaseBuilder): 向量库实例
            graphiti (Graphiti, optional): Graphiti知识图谱实例
            logger: 日志记录器
            config: 配置对象
        """
        self.logger = logger
        self.config = config

        # 初始化检索器
        self.vector_retriever = KnowledgeRetriever(knowledge_base, logger)
        self.kg_retriever = Graph_KG_Retriever(graphiti, logger) if graphiti else None

        # 初始化翻译器 (需要安装 transformers 库)
        try:
            self.translator = pipeline("translation", model=self.config.translator_path)
            self.logger.info("翻译模型加载成功")
        except Exception as e:
            self.logger.error(f"翻译模型加载失败: {e}")
            self.translator = None

        # 初始化关键词提取器
        self.stop_words = {
            # 疑问词
            '什么', '如何', '为什么', '怎么', '哪里', '哪个', '是否', '可以', '怎样', '怎么样', '吗', '呢', '吧',
            # 虚词/助词
            '的', '得', '地', '了', '着', '过', '在', '是', '有', '就', '也', '还', '都', '只', '共','中'
            # 冗余修饰词
            '非常', '极其', '特别', '一定', '可能', '大概', '也许', '比如', '例如', '等等', '诸多', '与'
        }

        # 保留核心词性（参考结巴词性编码：n=名词, v=动词, vn=动名词, adj=形容词, ns=地名, nr=人名, nz=其他专有名词）
        self.keep_pos = {'n', 'v', 'vn', 'adj', 'ns', 'nr', 'nz'}



    def _extract_keywords(self, question: str) -> List[str]:
        """
        从用户问题中提取关键词列表用于检索

        Args:
            question (str): 用户问题

        Returns:
            List[str]: 提取的关键词列表
        """
        if not question.strip():
            return []

        # 1. 预处理：移除标点和多余空格
        processed = re.sub(r'[^\w\s]', ' ', question).strip()
        if not processed:
            # 如果处理后为空，尝试提取英文缩写
            english_abbrs = re.findall(r'\b[A-Z]{2,}\b', question)
            return english_abbrs if english_abbrs else [question]

        # 2. 提取英文术语（如DGGS, GeoTIFF等）
        # 匹配模式包括：
        # 1. 混合大小写包含大写字母的术语（如GeoTIFF, DiscreteGlobalGridSystems）
        # 2. 全大写缩写词（如DGGS, GPS）
        # 3. 普通英文单词（长度至少3个字符）
        english_terms = re.findall(r'[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}|[A-Za-z]{3,}', question)

        # 3. 对英文术语进行去重但保持顺序
        seen = set()
        unique_english_terms = []
        for term in english_terms:
            if term not in seen:
                seen.add(term)
                unique_english_terms.append(term)
        # 11. 返回英文术语列表
        return unique_english_terms if unique_english_terms else [question]




    def _translate_to_english(self, keywords: List[str]) -> List[str]:
        """
        将中文关键词列表翻译为英文

        Args:
            keywords (List[str]): 中文关键词列表

        Returns:
            List[str]: 英文关键词列表
        """
        if not self.translator:
            self.logger.info("翻译模型未加载，返回原文本")
            return keywords

        english_keywords = []
        for keyword in keywords:
            try:
                if re.fullmatch(r'[A-Za-z0-9\- ]+', keyword) and re.search(r'[A-Z]{2,}|[A-Za-z]{3,}', keyword):
                    english_keywords.append(keyword)
                else:
                    result = self.translator(keyword, src_lang="zh", tgt_lang="en", clean_up_tokenization_spaces=True)
                    translated_text = result[0]['translation_text']
                    english_keywords.append(translated_text)
                    self.logger.info(f"翻译结果: {keyword} -> {translated_text}")
            except Exception as e:
                self.logger.info(f"翻译失败: {e}")
                english_keywords.append(keyword)

        return english_keywords

    def query_vector_database(self, keywords: List[str], top_k: int = 3, threshold: float = 0.7) -> list:
        """
        根据关键词列表查询向量数据库并返回结果

        Args:
            keywords (List[str]): 关键词列表
            top_k (int): 每个关键词返回结果数量
            threshold (float): 相似度阈值

        Returns:
            list: 检索到的文档列表
        """
        try:
            all_docs = []
            seen_contents = set()  # 用于去重

            self.logger.info(f"提取的关键词: {keywords}")

            # 翻译为英文
            english_keywords = self._translate_to_english(keywords)
            self.logger.info(f"英文关键词: {english_keywords}")

            # 对每个关键词进行检索
            for keyword in english_keywords:
                # 检索相关文档
                docs = self.vector_retriever.retrieve_relevant_docs(
                    query=keyword,
                    top_k=top_k,

                )

                # 去重并添加到结果中
                for doc in docs:
                    content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
                    if content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(content)

            return all_docs

        except Exception as e:
            self.logger.info(f"向量数据库查询失败: {e}")
            return []

    async def query_knowledge_graph(self, keywords: List[str], search_type: str = "basic", limit: int = 5) -> dict:
        """
        根据关键词列表查询知识图谱并返回结果

        Args:
            keywords (List[str]): 关键词列表
            search_type (str): 搜索类型 ("basic", "node", "center")
            limit (int): 每个关键词返回结果数量

        Returns:
            dict: 检索到的知识图谱结果
        """
        if not self.kg_retriever:
            return {"edges": [], "nodes": []}

        try:
            all_edges = []
            all_nodes = []
            seen_edges = set()
            seen_nodes = set()

            # 翻译为英文
            english_keywords = self._translate_to_english(keywords)

            # 对每个关键词进行检索
            for keyword in english_keywords:
                # 根据搜索类型执行不同搜索
                if search_type == "basic":
                    results = await self.kg_retriever.basic_search(keyword, limit)
                elif search_type == "node":
                    results = await self.kg_retriever.node_search(keyword, limit)
                elif search_type == "center":
                    # 先进行基础搜索获取中心节点
                    basic_results = await self.kg_retriever.basic_search(keyword, 1)
                    if basic_results["edges"]:
                        center_node_uuid = basic_results["edges"][0].source_node_uuid
                        results = await self.kg_retriever.center_node_search(
                            keyword, center_node_uuid, limit
                        )
                    else:
                        results = {"edges": [], "nodes": []}
                else:
                    results = {"edges": [], "nodes": []}

                # 去重并添加到结果中
                for edge in results.get("edges", []):
                    edge_id = getattr(edge, 'uuid', str(edge))
                    if edge_id not in seen_edges:
                        all_edges.append(edge)
                        seen_edges.add(edge_id)

                for node in results.get("nodes", []):
                    node_id = getattr(node, 'uuid', str(node))
                    if node_id not in seen_nodes:
                        all_nodes.append(node)
                        seen_nodes.add(node_id)

            return {"edges": all_edges, "nodes": all_nodes}

        except Exception as e:
            self.logger.info(f"知识图谱查询失败: {e}")
            return {"edges": [], "nodes": []}

    async def query_knowledge_graph_multi_methods(self, keywords: List[str], limit: int = 5) -> dict:
        """
        使用多种检索方式查询知识图谱并合并去重结果

        Args:
            keywords (List[str]): 关键词列表
            limit (int): 每个关键词返回结果数量

        Returns:
            dict: 合并去重后的知识图谱结果
        """
        if not self.kg_retriever:
            return {"edges": [], "nodes": []}

        try:
            all_edges = []
            all_nodes = []
            seen_edges = set()
            seen_nodes = set()

            # 翻译为英文
            english_keywords = self._translate_to_english(keywords)

            # 对每个关键词进行多种方式的检索
            for keyword in english_keywords:
                # 收集所有检索方法的结果
                search_methods = [
                    self.kg_retriever.basic_search(keyword, limit),
                    self.kg_retriever.node_search(keyword, limit),
                    self.kg_retriever.edge_hybrid_search_episode_mentions(keyword, limit),
                    self.kg_retriever.node_hybrid_search_episode_mentions(keyword, limit),
                    self.kg_retriever.combined_hybrid_search_rrf(keyword, limit),
                    self.kg_retriever.combined_hybrid_search_mmr(keyword, limit),

                ]

                # 并行执行所有检索方法
                results = await asyncio.gather(*search_methods, return_exceptions=True)

                # 处理每个检索结果
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.info(f"知识图谱检索出错: {result}")
                        continue

                    # 去重并添加到结果中
                    for edge in result.get("edges", []):
                        edge_id = getattr(edge, 'uuid', str(edge))
                        if edge_id not in seen_edges:
                            all_edges.append(edge)
                            seen_edges.add(edge_id)

                    for node in result.get("nodes", []):
                        node_id = getattr(node, 'uuid', str(node))
                        if node_id not in seen_nodes:
                            all_nodes.append(node)
                            seen_nodes.add(node_id)

            return {"edges": all_edges, "nodes": all_nodes}

        except Exception as e:
            self.logger.info(f"知识图谱多方式查询失败: {e}")
            return {"edges": [], "nodes": []}

    def _merge_contexts(self, vector_docs: list, kg_results: dict) -> str:
        """
        合并向量库和知识图谱的检索结果作为上下文

        Args:
            vector_docs (list): 向量库检索结果
            kg_results (dict): 知识图谱检索结果

        Returns:
            str: 合并后的上下文
        """
        context_parts = []

        # 处理向量库结果
        if vector_docs:
            vector_context = self.vector_retriever.format_retrieved_docs(vector_docs)
            context_parts.append(f"文档信息:\n{vector_context}")

        # 处理知识图谱结果
        if self.kg_retriever and (kg_results.get("edges") or kg_results.get("nodes")):
            kg_context = self.kg_retriever.format_combined_results(kg_results)
            context_parts.append(f"知识图谱信息:\n{kg_context}")

        return "\n\n".join(context_parts)

    async def _decompose_question(self, question: str) -> List[str]:
        """
        将复杂问题分解为多个子问题

        Args:
            question (str): 原始问题

        Returns:
            List[str]: 子问题列表
        """
        try:


            prompt = get_decomposition_template(question=question)


            response, _, _ = await call_gpt_async(
                prompt=prompt,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                if_print=True,
                temperature=0.1
            )

            # 将响应分割为子问题列表
            if '/n' in response:
                sub_questions = [q.strip() for q in response.split('/n') if q.strip()]
            else:
                # 使用默认的换行符分割
                sub_questions = [q.strip() for q in response.split('\n') if q.strip()]

            return sub_questions
        except Exception as e:
            self.logger.info(f"问题分解失败: {e}")
            return [question]

    async def _retrieve_and_rag(self, sub_questions: List[str]) -> tuple:
        """
        对子问题进行检索和初步回答

        Args:
            sub_questions (List[str]): 子问题列表

        Returns:
            tuple: (answers, questions)
        """
        try:


            # 存储结果
            rag_results = []

            # 循环获取结果
            for sub_question in sub_questions:
                # 查询子问题的文档内容

                vector_doc = self.query_vector_database([sub_question],top_k=self.config.vector_limit)
                kg_result = await self.query_knowledge_graph_multi_methods([sub_question], limit=self.config.max_limit)
                retrieved_docs = self._merge_contexts(vector_doc, kg_result)

                # 构造RAG提示
                prompt = get_rag_template(context=retrieved_docs, question=sub_question)

                # 获取答案
                answer, _, _ = await call_gpt_async(
                    prompt=prompt,
                    model=self.config.llm_model,
                    api_key=self.config.llm_api_key,
                    base_url=self.config.llm_base_url,
                    if_print=True,
                    temperature=0.1
                )

                rag_results.append(answer)

            return "\n\n".join(rag_results), sub_questions
        except Exception as e:
            self.logger.info(f"检索和RAG过程失败: {e}")
            return [], sub_questions

    def _format_qa_pairs(self, sub_questions: List[str], sub_answers: List[str]) -> str:
        """
        格式化问题和答案对

        Args:
            sub_questions (List[str]): 子问题列表
            sub_answers (List[str]): 子答案列表

        Returns:
            str: 格式化后的问题答案对
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(sub_questions, sub_answers), start=1):
            formatted_string += f"问题 {i}: {question}\n答案 {i}: {answer}\n\n"
        return formatted_string.strip()

    async def generate_answer(self, question: str, context: str) -> str:
        """
        根据检索到的上下文生成答案

        Args:
            question (str): 用户问题
            context (str): 检索到的上下文

        Returns:
            str: 生成的答案
        """
        if not context or context.strip() == "":
            return "抱歉，我没有找到相关的信息来回答您的问题。"

        try:
            # 使用您已有的提示模板
            prompt = get_prompt_template(context=context, question=question)

            response, _, _ = await call_gpt_async(
                prompt=prompt,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                if_print=False,
                temperature=0.1
            )

            return response
        except Exception as e:
            self.logger.info(f"生成答案失败: {e}")
            return "抱歉，我在生成答案时遇到了一些问题。"

    async def answer_question_decomposed(self, question: str) -> str:
        """
        通过问题分解方式回答用户问题

        Args:
            question (str): 用户问题

        Returns:
            str: 最终答案
        """
        self.logger.info(f"通过问题分解方式回答用户问题: {question}")

        try:
            # 1. 分解问题
            sub_questions = await self._decompose_question(question)
            self.logger.info(f"分解后的问题: {sub_questions}")

            # 2. 对每个子问题进行检索和回答
            answers, questions = await self._retrieve_and_rag(sub_questions)

            # 3. 组织问题答案对
            context = self._format_qa_pairs(questions, answers)

            # # 4. 生成最终答案
            # final_answer = await self.generate_answer(question, context)
            return  context

        except Exception as e:
            self.logger.info(f"问题分解回答过程出错: {e}")
            return "抱歉，我在回答您的问题时遇到了一些问题。"

    async def answer_question(self, question: str) -> str:
        """
        回答用户问题的主方法，结合向量库和知识图谱进行检索

        Args:
            question (str): 用户问题

        Returns:
            str: 最终答案
        """
        self.logger.info(f"收到用户问题: {question}")

        try:
            # 提取关键词列表
            keywords = self._extract_keywords(question)
            self.logger.info(f"提取的关键词列表: {keywords}")


            # 分解问题获取子问题
            sub_questions = await self._decompose_question(question)
            self.logger.info(f"分解后的问题: {sub_questions}")


            # 1.针对关键词列表中的关键词，并行查询向量数据库和知识图谱（使用多种检索方式）
            vector_docs = self.query_vector_database(keywords, top_k=3, threshold=0.7)
            kg_results = await self.query_knowledge_graph_multi_methods(keywords,
                                                                        limit=10) if self.kg_retriever else {"edges": [],
                                                                                                            "nodes": []}
            #2.针对子问题，进行检索并回答，然后生成问题答案对的上下文
            sub_answers_context, sub_questions = await self._retrieve_and_rag(sub_questions)
            sub_answers_context = self._format_qa_pairs(sub_questions, sub_answers_context)

            # 合并检索结果作为上下文
            keywords_context = self._merge_contexts(vector_docs, kg_results)

            combined_context = sub_answers_context + keywords_context


            # 生成最终答案
            answer = await self.generate_answer(question, combined_context)
            return answer

        except Exception as e:
            self.logger.info(f"回答问题时出现错误: {e}")
            return "抱歉，我在回答您的问题时遇到了一些问题。"


# 完整的测试示例（需要提供真实的依赖项）
if __name__ == "__main__":


    async def test_qa_system():
        # 初始化配置和日志
        config = Config()
        logger = Logger()
        vector_db = KnowledgeBaseBuilder(config,logger)

        graph_db = GraphKnowledgeBase(config,logger)
        await graph_db.initialize()

        # 加载知识库和知识图谱实例
        # 这里需要根据您的实际代码进行调整
        # knowledge_base = load_chroma_instance()
        # graphiti = load_graphiti_instance()

        # 初始化问答系统
        qa_system = QASystem(vector_db, graph_db, logger, config)

        # 测试问题
        test_questions = [
            "在 OGC API-DGGS 标准中，ISEA3H DGGRS 的六边形子区与五边形子区在面积上存在差异，该标准对这种面积差异的处理逻辑是什么？且这种差异会对基于该 DGGRS 的空间数据统计分析（如区域平均海拔计算）产生哪些影响？",
        #     "文档中提到 DGGS-JSON-FG 编码通过特殊子区索引值 0 处理跨区几何拼接的人工边缘问题，具体而言，当矢量几何（如道路线）跨越两个相邻 DGGRS 分区时，该编码如何通过子区索引 0 标识人工边缘，且客户端应如何利用此标识实现几何的无缝拼接？",
        #     "在 Zone Query 功能中，compact-zones 参数设为 true 时会递归用父区替代完全包含的子区，若某 DGGRS 层级中存在部分子区数据缺失的情况（如部分子区无高程数据），该标准规定服务器应如何处理 compact-zones 逻辑以确保返回的紧凑区列表能准确反映数据可用性？",
        #     "文档定义的 Data Custom Depths 需求类支持通过 zone-depth 参数请求多深度子区数据，当请求的 zone-depth 超出 DGGRS 描述中 maxRelativeDepth 限制时，服务器有两种处理方式（返回 4xx 错误或过采样），这两种处理方式各自的适用场景是什么？且过采样处理时需遵循哪些数据精度保障规则？",
        #     "在 Filtering Zone Data with CQL2 需求类中，针对栅格数据和矢量数据，CQL2 表达式过滤的执行逻辑存在差异（栅格设 NODATA、矢量过滤特征），若某数据集同时包含栅格高程数据和矢量道路数据，当使用 S_INTERSECTS () 函数结合高程条件（如 Elevation>500）进行过滤时，服务器需如何协调两种数据类型的过滤逻辑以确保结果一致性？"
        ]

        # 测试关键词提取
        print("=== 关键词提取测试 ===")
        for question in test_questions:
            print(f"原问题：{question}")
            keywords = qa_system._extract_keywords(question)
            print(f"提取关键词：{keywords}\n")

        # 测试问答功能
        print("=== 问答功能测试 ===")
        for question in test_questions:
            print(f"问题：{question}")
            answer = await qa_system.answer_question(question)
            print(f"回答：{answer}\n")


    # 运行测试
    asyncio.run(test_qa_system())
