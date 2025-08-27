# 这是一个专门用于提问的LLM类，并使用QASystem进行解答

import asyncio
import json
import sys
import random
from typing import List

from Import_KG.import_kg import KnowledgeBaseBuilder
from Import_KG.Graph_kg_base import GraphKnowledgeBase
from utils.QASystem import QASystem
from Config.config import Config, Logger
from utils.call_gpt import call_gpt_async


class QuestionAsker:
    """
    专门用于提问的LLM类
    """

    def __init__(self):
        """
        初始化提问系统
        """
        self.config = Config()
        self.logger = Logger()
        self.qa_system = None
        # 添加对话历史记录
        self.conversation_history = []

    async def initialize(self):
        """
        初始化问答系统
        """
        try:
            # 初始化向量知识库
            self.logger.info("正在初始化向量知识库...")
            self.vector_db = KnowledgeBaseBuilder(self.config, self.logger)
            self.vector_db.load_existing_knowledge_base()

            # 初始化图知识库
            self.logger.info("正在初始化图知识库...")
            self.graph_db = GraphKnowledgeBase(self.config, self.logger)
            await self.graph_db.initialize()

            # 初始化问答系统
            self.logger.info("正在初始化问答系统...")
            self.qa_system = QASystem(self.vector_db, self.graph_db, self.logger, self.config)

            self.logger.info("系统初始化完成！")
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    async def ask_question_without_memory(self, question: str) -> str:
        """
        提出问题并获取答案

        Args:
            question (str): 要提出的问题

        Returns:
            str: 问题的答案
        """
        if not self.qa_system:
            raise RuntimeError("问答系统未初始化，请先调用initialize()方法")

        try:
            self.logger.info(f"正在处理问题: {question}")
            answer = await self.qa_system.answer_question(question)
            return answer
        except Exception as e:
            self.logger.error(f"回答问题时出错: {e}")
            return f"抱歉，回答问题时出现错误: {str(e)}"

    async def ask_question(self, question: str) -> str:
        """
        提出问题并获取答案

        Args:
            question (str): 要提出的问题

        Returns:
            str: 问题的答案
        """
        if not self.qa_system:
            raise RuntimeError("问答系统未初始化，请先调用initialize()方法")

        try:
            self.logger.info(f"正在处理问题: {question}")
            # 将当前问题添加到对话历史中
            self.conversation_history.append({"role": "user", "content": question})

            # 限制对话历史长度，只保留最近的10轮对话
            if len(self.conversation_history) > 20:  # 10轮对话包含问答
                self.conversation_history = self.conversation_history[-20:]

            answer = await self.qa_system.answer_question_with_context(question, self.conversation_history)

            # 将回答添加到对话历史中
            self.conversation_history.append({"role": "assistant", "content": answer})

            return answer
        except Exception as e:
            self.logger.error(f"回答问题时出错: {e}")
            return f"抱歉，回答问题时出现错误: {str(e)}"

    async def generate_questions_from_context(self, num_questions: int = 3) -> List[str]:
        """
                    基于知识库内容自动生成问题

                    Args:
                        num_questions (int): 生成问题的数量

                    Returns:
                        List[str]: 生成的问题列表
                    """
        try:
            # 从向量数据库中随机获取一些文档作为上下文
            sample_docs = self.vector_db.load_and_split_documents()
            # 随机选取k个索引
            k = 3  # 可以根据需要调整k的值
            if len(sample_docs) >= k:
                random_indices = random.sample(range(len(sample_docs)), k)
                selected_docs = [sample_docs[i] for i in random_indices]
            else:
                selected_docs = sample_docs

            if not sample_docs:
                self.logger.warning("没有找到足够的文档来生成问题")
                return []

            # 合并文档内容作为上下文
            context = "\n\n".join([doc.page_content for doc in selected_docs])

            # 构建提示词
            prompt = f"""
                        基于以下文档内容，生成{num_questions}个相关的问题。
                        这些问题应该涵盖文档中的关键概念和重要信息，并且问题应该具有挑战性。

                        文档内容：
                        {context}  

                        每个问题之间，用换行符'/n'分隔,不准使用其他符号，问题用中文论述,不准生成无关的问题和内容；
                        """

            # 调用LLM生成问题
            response, _, _ = await call_gpt_async(
                prompt=prompt,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                if_print=False,
                temperature=0.7  # 适当提高创造性
            )

            if '/n' in response:
                questions = [q.strip() for q in response.split('/n') if q.strip()]
            else:
                # 使用默认的换行符分割
                questions = [q.strip() for q in response.split('\n') if q.strip()]



            return questions  # 返回指定数量的问题

        except Exception as e:
            self.logger.error(f"从上下文中生成问题失败: {e}")
            return []

    async def auto_qa_session(self, num_questions: int = 5) -> List[dict]:
        """
        自动生成问题并回答的完整会话

        Args:
            num_questions (int): 生成问题的数量

        Returns:
            List[dict]: 包含问题和答案的字典列表
        """
        qa_pairs = []

        try:
            # 生成问题
            self.logger.info("正在基于知识库内容生成问题...")
            questions = await self.generate_questions_from_context(num_questions)

            if not questions:
                self.logger.info("未能生成任何问题")
                return qa_pairs

            self.logger.info(f"生成了{len(questions)}个问题: {questions}")

            # 回答每个问题
            for i, question in enumerate(questions, 1):
                self.logger.info(f"正在回答第{i}个问题: {question}")
                try:
                    answer = await self.ask_question(question)
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
                    self.logger.info(f"第{i}个问题回答完成")
                except Exception as e:
                    self.logger.error(f"回答问题 '{question}' 时出错: {e}")
                    qa_pairs.append({
                        "question": question,
                        "answer": f"回答问题时出错: {str(e)}"
                    })

            return qa_pairs

        except Exception as e:
            self.logger.error(f"自动生成问答会话失败: {e}")
            return qa_pairs

    async def interactive_qa(self):
        """
        交互式问答模式
        """
        print("欢迎使用问答系统！输入 'quit' 或 'exit' 退出程序。")
        print("系统现在支持对话记忆功能，可以基于上下文理解您的问题。")
        print("-" * 50)

        while True:
            try:
                question = input("\n请输入您的问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用，再见！")
                    break

                if not question:
                    print("请输入有效问题。")
                    continue

                print("正在思考中，请稍候...")
                answer = await self.ask_question(question)
                print(f"\n答案: {answer}")

            except KeyboardInterrupt:
                print("\n\n程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"处理问题时出现错误: {e}")




async def main():
    """
    主函数

    """
    try:
        # 创建提问系统实例
        asker = QuestionAsker()

        # 初始化系统
        await asker.initialize()

        num_questions = 2
        qa_pairs = await asker.auto_qa_session(num_questions)
        print("\n" + "=" * 60)
        print("自动生成的问答:")
        print("=" * 60)

        for i, qa_pair in enumerate(qa_pairs, 1):
            print(f"\n问题 {i}: {qa_pair['question']}")
            print(f"答案 {i}: {qa_pair['answer']}")
            print("-" * 60)

        with open("qa_pairs.json", "w", encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

        # 交互模式
        # await asker.interactive_qa()



    except Exception as e:
        print(f"程序运行出错: {e}")



if __name__ == '__main__':
    asyncio.run(main())