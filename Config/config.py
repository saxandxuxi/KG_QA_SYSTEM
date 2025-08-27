import logging
import os
DOUBAO_API_KEY = os.environ.get('DOUBAO_API_KEY')
DOUBAO_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'

QWEN_API_KEY = os.environ.get('QWEN_API_KEY')
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY')
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

DEEPSEEK_API_KEY = os.environ.get('deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_Model: str = 'deepseek-chat'

BAISHAN_API_KEY = os.environ.get('BAISHAN_API_KEY')
BAISHAN_BASE_URL = "https://api.edgefn.net/v1"
BAISHAN_Model: str = 'DeepSeek-R1-0528'

siliconflow_API_KEY = 'sk-ioqnrsrufdlqntzrsgyfwnuyrsndxbmsjhvuoobkydwemirn'
siliconflow_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"

class Logger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("QASystem")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

class Config:
    def __init__(self):
        # TODO: 请修改放置markdown文件的目录
        self.data_dir = "./KG_Base"  # 放置PDF文件的目录
        self.chroma_db_path = './chroma_db'
        # 文本切分相关参数
        self.chunk_size = 1500
        self.chunk_overlap = 200

        # 向量数据库相关参数
        self.embedding_model = "./models/BAAI/bge-large-en-v1___5"  # embedding模型
        self.collection_name = "OGC"
        self.max_results = 5  # 返回top5相似性结果
        self.distance_threshold = 0.25  # 相似性大于等于0.75； distance的含义为：（1-余弦相似度）

        # llm相关参数  # TODO: 请修改大模型接口相关参数
        self.llm_api_key = os.environ.get('ZHIPU_API_KEY')
        self.llm_base_url = ZHIPU_BASE_URL
        self.llm_model = 'GLM-4.5-AirX'

        self.translator_path = './models/opus-mt-zh-en'

        # 知识图谱限制
        self.max_limit = 8
        # 向量库限制
        self.vector_limit = 3


        # Neo4j配置
        self.neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.environ.get('NEO4J_PASSWORD', '12345678')
        self.neo4j_database = os.environ.get('NEO4J_DATABASE', 'companypaper2')
        #companypaper终止片段：435，切分 500， overlap 50
        #companypaper2终止片段：？，切分 1500， overlap 200


        # ollama嵌入模型配置
        self.embedding_api_key = os.environ.get('EMBEDDING_API_KEY', 'abc')
        self.embedding_base_url = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:11434/v1')
        self.embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME', 'qllama/bge-large-en-v1.5:latest')
        self.embedding_dim = int(os.environ.get('EMBEDDING_DIM', '768'))



