import asyncio
import uuid
from contextlib import suppress
from datetime import datetime, timezone

from IPython.core.display_functions import display
from ipywidgets import Image, widgets
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver# 对话记忆存储
from langgraph.graph import END, START, StateGraph, add_messages # 状态图构建
from langgraph.prebuilt import ToolNode
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS
from graphiti_core.nodes import EpisodeType
from typing_extensions import TypedDict
from typing import Annotated
from agent import get_client
from agent import edges_to_facts_string, QWEN_MODEL, QWEN_BASE_URL, QWEN_API_KEY
llm = None
client = None

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表（带自动追加功能）
    user_name: str  # 用户名
    user_node_uuid: str  # 用户在Graphiti中的节点UUID（用于关联用户信息）


async def chatbot(state: State):
    # 从Graphiti中检索与当前对话相关的用户信息
    facts_string = None
    if len(state['messages']) > 0:
        last_message = state['messages'][-1]
        graphiti_query = f'{"SalesBot" if isinstance(last_message, AIMessage) else state["user_name"]}: {last_message.content}'
        # search graphiti using Jess's node uuid as the center node
        # graph edges (facts) further from the Jess node will be ranked lower
        edge_results = await client.search( # 搜索用户相关的知识图谱边（事实）
            graphiti_query, center_node_uuid=state['user_node_uuid'], num_results=5
        )
        facts_string = edges_to_facts_string(edge_results) # 转换为自然语言事实
    #构建系统提示（定义销售机器人角色和目标）
    system_message = SystemMessage(
        content=f"""You are a skillfull shoe salesperson working for ManyBirds. Review information about the user and their prior conversation below and respond accordingly.
        Keep responses short and concise. And remember, always be selling (and helpful!)

        Things you'll need to know about the user in order to close a sale:
        - the user's shoe size
        - any other shoe needs? maybe for wide feet?
        - the user's preferred colors and styles
        - their budget

        Ensure that you ask the user for the above if you don't already know.

        Facts about the user and their conversation:
        {facts_string or 'No facts about the user and their conversation'}"""
    )

    # 调用LLM生成响应
    messages = [system_message] + state['messages']
    response = await llm.ainvoke(messages)

    #  将对话记录添加到Graphiti知识图谱（异步执行，不阻塞流程）
    asyncio.create_task(
        client.add_episode(
            name='Chatbot Response',
            episode_body=f'{state["user_name"]}: {state["messages"][-1]}\nSalesBot: {response.content}',
            source=EpisodeType.message,
            reference_time=datetime.now(datetime.timezone.utc),
            source_description='Chatbot',
        )
    )

    return {'messages': [response]}

# Define the function that determines whether to continue or not
async def should_continue(state, config):
    messages = state['messages']
    last_message = messages[-1]
    # # 若最后一条消息无工具调用，则结束流程；否则继续调用工具
    if not last_message.tool_calls:
        return 'end'
    # Otherwise if there is, we continue
    else:
        return 'continue'
# 处理用户输入并更新对话输出
async def process_input(user_state: State, user_input: str,conversation_output,graph,config):
    conversation_output.append_stdout(f'\nUser: {user_input}\n')
    conversation_output.append_stdout('\nAssistant: ')

    graph_state = {
        'messages': [{'role': 'user', 'content': user_input}],
        'user_name': user_state['user_name'],
        'user_node_uuid': user_state['user_node_uuid'],
    }

    try:
        async for event in graph.astream(
            graph_state,
            config=config,
        ):
            for value in event.values():
                if 'messages' in value:
                    last_message = value['messages'][-1]
                    if isinstance(last_message, AIMessage) and isinstance(
                        last_message.content, str
                    ):
                        conversation_output.append_stdout(last_message.content)
    except Exception as e:
        conversation_output.append_stdout(f'Error: {e}')

# # 提交按钮回调函数
def on_submit(input_box,user_state,conversation_output,graph,config):
    user_input = input_box.value
    input_box.value = ''
    asyncio.create_task(process_input(user_state,user_input,conversation_output,graph,config))

async def main():
    client = await get_client()
    nl = await client._search('ManyBirds', NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    manybirds_node_uuid = nl.nodes[0].uuid

    @tool
    async def get_shoe_data(query: str) -> str:
        """Search the graphiti graph for information about shoes"""
        edge_results = await client.search(
            query,
            center_node_uuid=manybirds_node_uuid,
            num_results=10,
        )
        return edges_to_facts_string(edge_results)  ## 将查询结果转换为自然语言字符串返回

    tools = [get_shoe_data]
    tool_node = ToolNode(tools)
    llm = ChatOpenAI(
        model_name=QWEN_MODEL,
        temperature=0,
        openai_api_key=QWEN_API_KEY,
        openai_api_base=QWEN_BASE_URL
    ).bind_tools(tools)  # 关键：将工具绑定到模型上

    graph_builder = StateGraph(State)
    memory = MemorySaver()# # 存储对话历史
    graph_builder.add_node('agent', chatbot)
    graph_builder.add_node('tools', tool_node)
    #定义边：START -> agent；agent根据should_continue到tools或END；tools -> agent
    graph_builder.add_edge(START, 'agent')
    graph_builder.add_conditional_edges('agent', should_continue, {'continue': 'tools', 'end': END})
    graph_builder.add_edge('tools', 'agent')
    graph = graph_builder.compile(checkpointer=memory)
    user_name = 'jess'
    # let's get Jess's node uuid
    user_node_uuid = await client._search(user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    await graph.ainvoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What sizes do the TinyBirds Wool Runners in Natural Black come in?',
                }
            ],
            'user_name': user_name,
            'user_node_uuid': user_node_uuid,
        },
        config={'configurable': {'thread_id': uuid.uuid4().hex}},
    )

    conversation_output = widgets.Output()
    config = {'configurable': {'thread_id': uuid.uuid4().hex}}
    user_state = {'user_name': user_name, 'user_node_uuid': user_node_uuid}
    input_box = widgets.Text(placeholder='Type your message here...')
    submit_button = widgets.Button(description='Send')
    submit_button.on_click(on_submit,conversation_output,graph,config)

    conversation_output.append_stdout('Asssistant: Hello, how can I help you find shoes today?')

    display(widgets.VBox([input_box, submit_button, conversation_output]))

if __name__ == '__main__':
    asyncio.run(main())