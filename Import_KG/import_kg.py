import torch
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import shutil
from langchain.vectorstores import utils as chromautils

from modelscope import snapshot_download  # 新增：用于下载模型

from Config.config import Config, Logger

# 下载英文嵌入模型（BGE英文版本）
# model_dir = snapshot_download(
#     model_id='BAAI/bge-large-en-v1.5',  # 英文版本模型ID
#     cache_dir='../models'
# )
# print(f"英文模型下载完成，路径：{model_dir}")



class KnowledgeBaseBuilder:
    def __init__(self, config, logger):
        """
        初始化知识库构建器

        Args:
            config: 配置对象，包含路径和参数设置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.db = None

    def load_and_split_documents(self):
        """
        加载并分割PDF文档

        Returns:
            list: 分割后的文档列表
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

        return split_docs

    def _create_embedding_function(self):
        """
        创建嵌入函数

        Returns:
            HuggingFaceEmbeddings: 嵌入模型实例
        """
        # 自动检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {device}")

        # 初始化嵌入模型
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': device}
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"加载嵌入模型失败: {e}")
            raise

    def _clear_existing_database(self, db_path):
        """
        清除已有的向量库目录

        Args:
            db_path (str): 向量库存储路径
        """
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                self.logger.info(f"已删除原有向量库目录: {db_path}")
            except Exception as e:
                self.logger.error(f"删除原有向量库目录失败: {e}")
                raise

    def build_local_knowledge_base(self):
        """
        构建本地知识库

        Returns:
            Chroma: 构建好的向量库实例
        """
        try:
            # 清除已有的向量库
            self._clear_existing_database(self.config.chroma_db_path)

            # 加载并分割文档
            split_docs = self.load_and_split_documents()

            # 创建嵌入函数
            embeddings = self._create_embedding_function()

            # 构建向量库
            self.db = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=self.config.chroma_db_path
            )

            self.logger.info(f"知识库构建完成！向量库路径：{self.config.chroma_db_path}")
            self.logger.info(f"加载文档片段数：{len(split_docs)}")

            return self.db

        except Exception as e:
            self.logger.error(f"构建知识库失败: {e}")
            raise

    def load_existing_knowledge_base(self):
        """
        加载已构建的本地知识库
        """
        try:
            # 使用传入的参数或配置中的默认值
            db_path = self.config.chroma_db_path
            model_path = self.config.embedding_model

            # 检查向量库路径是否存在
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"向量库路径不存在: {db_path}")

            # 初始化嵌入模型（需与构建知识库时使用的模型一致）
            embeddings = self._create_embedding_function()

            # 加载已存在的向量库
            self.db = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )

            self.logger.info(f"成功加载现有知识库，路径: {db_path}")
            return self.db

        except Exception as e:
            self.logger.error(f"加载现有知识库失败: {e}")
            raise
    def get_knowledge_base(self):
        """
        获取已构建的知识库实例

        Returns:
            Chroma: 向量库实例
        """
        return self.db


if __name__ == "__main__":
    config = Config()
    logger = Logger()
    config.data_dir = '../KG_Base'
    config.chroma_db_path = '../chroma_db'
    config.embedding_model = '../models/BAAI/bge-large-en-v1___5'
    kg_db = KnowledgeBaseBuilder(config,logger)
    kg_db = kg_db.build_local_knowledge_base()

