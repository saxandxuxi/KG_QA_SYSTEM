from langgraph.prebuilt import ToolNode
from langchain_community.llms import Qwen
from langchain_core.messages import HumanMessage, ToolMessage
from typing import List, Dict, Any
import os

# 3. Qwen模型初始化（关键修改）
class QwenWithTools(Qwen):
    """Qwen模型增强版，支持工具调用和异步处理"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = tools  # 注册可用工具

    async def ainvoke(self, input: str) -> Dict[str, Any]:
        """异步调用接口，支持工具绑定"""
        try:
            # 生成工具调用
            tool_call = await self._generate_tool_call(input)
            # 执行工具
            tool_response = await self._execute_tool(tool_call)
            # 组合最终响应
            return self._format_response(tool_response)
        except Exception as e:
            return {"error": f"Model invocation failed: {str(e)}"}

    async def _generate_tool_call(self, query: str) -> Dict:
        """生成工具调用请求"""
        response = await self.agenerate(
            prompts=[query],
            stop=["</tool_call>"]  # 根据Qwen的tool_call格式调整
        )
        return self._parse_tool_call(response.generations[0].text)

    async def _execute_tool(self, tool_call: Dict) -> str:
        """执行具体工具"""
        tool = next(t for t in self.tools if t.name == tool_call["name"])
        return await tool(tool_call["arguments"])

    def _parse_tool_call(self, response: str) -> Dict:
        """解析Qwen的tool call格式"""
        # 示例解析逻辑（需根据实际响应格式调整）
        return {
            "name": "get_shoe_data",
            "arguments": {"query": "wool shoes"}
        }

    def _format_response(self, tool_response: str) -> Dict:
        """格式化最终响应"""
        return {
            "content": tool_response,
            "type": "tool_response"
        }


# 4. 初始化组件
# 配置Qwen参数
qwen_config = {
    "model_name": "qwen-max",  # 或 qwen-turbo/qwen-plus
    "api_key": os.getenv("QWEN_API_KEY"),
    "base_url": os.getenv("QWEN_BASE_URL"),
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": False
}
# 创建增强模型实例
qwen_llm = QwenWithTools(**qwen_config)
# 创建工具节点
tool_node = ToolNode(tools)


# 5. 构建执行流程
async def run_workflow(query: str) -> str:
    """完整执行流程"""
    # 1. 用户查询
    user_msg = HumanMessage(content=query)
    # 2. 模型生成工具调用
    tool_call = await qwen_llm.ainvoke(user_msg.content)
    # 3. 执行工具
    tool_response = await tool_node.ainvoke({
        "messages": [tool_call]
    })
    # 4. 合并响应
    final_response = f"""
    Analysis Results:
    {tool_response}
    """
    return final_response


# 6. 测试执行
async def main():
    try:
        result = await run_workflow("Find eco-friendly wool shoes under $100")
        print("Final Answer:")
        print(result)
    except Exception as e:
        print(f"Workflow failed: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())