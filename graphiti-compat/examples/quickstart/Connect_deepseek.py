import os
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
DEEPSEEK_API_KEY = os.environ.get('deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_Model: str = 'deepseek-chat'
llmconfig = LLMConfig(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL,model=DEEPSEEK_Model)



from graphiti_core.llm_client.openai_compat_client import LLMClient,OpenAICompatClient
llm_client = OpenAICompatClient(llmconfig)


from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="abc",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            base_url="http://localhost:11434/v1",
        )
    )

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "12345678",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llmconfig),
)
