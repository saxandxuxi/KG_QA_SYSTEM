import openai
import re
import pandas as pd
import os
import asyncio  # 为异步示例添加导入

# ==============================================================================
# 用户配置区域 START
# 请在此处填入您的 DeepSeek API Key
# ==============================================================================
QWEN_API_KEY = os.environ.get('QWEN_API_KEY')
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DOUBAO_API_KEY = 'accbe191-e8b5-48ce-958b-25cfa3d224a1'
DOUBAO_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
# ==============================================================================
# 用户配置区域 END
# ==============================================================================




def call_gpt(prompt: str,
             api_key: str = QWEN_API_KEY,
             base_url: str = QWEN_BASE_URL,
             model: str = 'qwen3-30b-a3b-instruct-2507',
             if_print: bool = False,
             system_prompt: str = "You are a helpful assistant.",
             temperature: float = 0
             ) -> tuple[str, str, openai.types.CompletionUsage | None]:
    """
    本函数用于同步获取大语言模型的回答 (通过 OpenAI SDK)。
    API Key 和 Base URL 从脚本顶部的全局配置中读取。

    参数:
        prompt (str): 用户的输入提示。
        model (str): 使用的语言模型ID。
        if_print (bool): 是否打印输入和输出。
        system_prompt (str): 给模型的系统级指令。
        temperature (float): 生成文本的随机性。

    返回:
        Tuple[str, str, openai.types.CompletionUsage | None]:
            - content (str): 模型回复的主要内容。
            - reasoning_content (str): 模型的思考过程或相关元数据 (如果可用，否则为提示信息)。
            - api_usage (openai.types.CompletionUsage | None): API 调用用量信息，如果API不返回则为None。
    """
    try:
        if api_key is None:
            api_key = QWEN_API_KEY

        if base_url is None:
            base_url = QWEN_BASE_URL

        client = openai.Client(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model=model,
            stream=False,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content
        api_usage = completion.usage

        reasoning_content_value = None
        try:
            # 尝试获取 reasoning_content，不同模型或API版本可能存在差异
            reasoning_content_value = getattr(completion.choices[0].message, 'reasoning_content', None)
            if reasoning_content_value is None:
                # 有些模型可能将类似思考过程的内容放在 tool_calls 中，这里简化处理
                # 如果您的模型有特定结构，请调整此处
                if completion.choices[0].message.tool_calls:
                    reasoning_content_value = f"Tool calls: {completion.choices[0].message.tool_calls}"
                else:
                    reasoning_content_value = "reasoning_content not explicitly found or is None."
        except AttributeError:
            reasoning_content_value = "reasoning_content attribute not found in response."
        except Exception as e:
            reasoning_content_value = f"Error accessing reasoning_content: {str(e)}"

        if if_print:
            print('=' * 20 + " PROMPT " + '=' * 20)
            print(prompt)
            print('=' * 20 + " RESPONSE " + '=' * 20)
            print(content)
            if reasoning_content_value and "not found" not in reasoning_content_value and "is None" not in reasoning_content_value:
                print('-' * 20 + " REASONING " + '-' * 20)
                print(reasoning_content_value)
            print('=' * 50)

        return content, reasoning_content_value, api_usage

    except openai.APIConnectionError as e:
        print(f"API 连接错误: {e.__cause__}")
        return f"Error: API Connection Error - {e}", "Connection Error", None
    except openai.RateLimitError as e:
        print(f"API 请求超过速率限制: {e.response.text}")
        return f"Error: Rate Limit Exceeded - {e}", "Rate Limit Error", None
    except openai.APIStatusError as e:
        print(f"API 状态错误 (HTTP Code: {e.status_code}): {e.response.text}")
        return f"Error: API Status Error {e.status_code} - {e}", "API Status Error", None
    except Exception as e:
        print(f"调用 DeepSeek API 时发生未知错误: {e}")
        return f"Error: An unexpected error occurred - {e}", "Unexpected Error", None


async def call_gpt_async(prompt: str,
                         model: str = 'doubao-1-5-pro-32k-250115',
                         api_key: str = DOUBAO_API_KEY,
                         base_url: str = DOUBAO_BASE_URL,
                         if_print: bool = False,
                         temperature: float = 0,
                         system_prompt: str = "You are a helpful assistant."
                         ) -> tuple[str, str, openai.types.CompletionUsage | None]:
    """
    异步版本 - 本函数用于异步获取大语言模型的回答 (通过 OpenAI SDK)。
    API Key 和 Base URL 从脚本顶部的全局配置中读取。

    参数:
        prompt (str): 用户的输入提示。
        model (str): 使用的语言模型ID。
        if_print (bool): 是否打印输入和输出。
        temperature (float): 生成文本的随机性。
        system_prompt (str): 给模型的系统级指令。

    返回:
        Tuple[str, str, openai.types.CompletionUsage | None]:
            - content (str): 模型回复的主要内容。
            - reasoning_content (str): 模型的思考过程或相关元数据 (如果可用，否则为提示信息)。
            - api_usage (openai.types.CompletionUsage | None): API 调用用量信息，如果API不返回则为None。
    """
    try:
        if api_key is None:
            api_key = DOUBAO_API_KEY

        if base_url is None:
            base_url = DOUBAO_BASE_URL

        client = openai.AsyncClient(api_key=api_key, base_url=base_url)

        completion = await client.chat.completions.create(
            model=model,
            stream=False,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content
        api_usage = completion.usage

        reasoning_content_value = None
        try:
            reasoning_content_value = getattr(completion.choices[0].message, 'reasoning_content', None)
            if reasoning_content_value is None:
                if completion.choices[0].message.tool_calls:
                    reasoning_content_value = f"Tool calls: {completion.choices[0].message.tool_calls}"
                else:
                    reasoning_content_value = "reasoning_content not explicitly found or is None."
        except AttributeError:
            reasoning_content_value = "reasoning_content attribute not found in response."
        except Exception as e:
            reasoning_content_value = f"Error accessing reasoning_content: {str(e)}"

        if if_print:
            print('=' * 20 + " ASYNC PROMPT " + '=' * 20)
            print(prompt)
            print('=' * 20 + " ASYNC RESPONSE " + '=' * 20)
            print(content)
            if reasoning_content_value and "not found" not in reasoning_content_value and "is None" not in reasoning_content_value:
                print('-' * 20 + " ASYNC REASONING " + '-' * 20)
                print(reasoning_content_value)
            print('=' * 50)

        return content, reasoning_content_value, api_usage

    except openai.APIConnectionError as e:
        print(f"异步 API 连接错误: {e.__cause__}")
        return f"Error: API Connection Error - {e}", "Connection Error", None
    except openai.RateLimitError as e:
        print(f"异步 API 请求超过速率限制: {e.response.text}")
        return f"Error: Rate Limit Exceeded - {e}", "Rate Limit Error", None
    except openai.APIStatusError as e:
        print(f"异步 API 状态错误 (HTTP Code: {e.status_code}): {e.response.text}")
        return f"Error: API Status Error {e.status_code} - {e}", "API Status Error", None
    except Exception as e:
        print(f"异步调用 DeepSeek API 时发生未知错误: {e}")
        return f"Error: An unexpected error occurred - {e}", "Unexpected Error", None


if __name__ == "__main__":
    # --- 同步调用  示例 ---
    print("--- 同步调用 QWEN 示例 ---")
    sync_prompt = "请用中文解释一下什么是量子纠缠，要通俗易懂。"
    try:
        s_content, s_reasoning, s_usage = call_gpt(
            prompt=sync_prompt,
            model='qwen-turbo',
            if_print=True,  # 函数内部会打印
            temperature=0.7
        )
        # if_print=True 已经在函数内打印，这里可以打印额外信息或进行后续处理
        print(f"\n同步调用完成。")
        # print(f"同步 - 内容:\n{s_content}")
        if s_reasoning and "not found" not in s_reasoning and "is None" not in s_reasoning:
            print(f"同步 - 返回的思考过程/元数据:\n{s_reasoning}")
        else:
            print(f"同步 - 未找到明确的思考过程/元数据。")

        if s_usage:
            print(f"同步 - API 使用情况: Prompt Tokens: {s_usage.prompt_tokens}, Completion Tokens: {s_usage.completion_tokens}, Total Tokens: {s_usage.total_tokens}")
        else:
            print("同步 - 未获取到 API 使用情况。")

    except Exception as e:
        print(f"同步调用主程序出错: {e}")

    print("\n" + "=" * 60 + "\n")

    # --- 异步调用 示例 ---
    async def run_async_example():
        print("--- 异步调用 QWEN 示例 ---")
        async_prompt = "讲一个北极熊笑话"
        try:
            a_content, a_reasoning, a_usage = await call_gpt_async(
                prompt=async_prompt,
                if_print=True,  # 函数内部会打印
                temperature=0.6
            )
            print(f"\n异步调用完成。")
            # print(f"异步 - 内容:\n{a_content}")
            if a_reasoning and "not found" not in a_reasoning and "is None" not in a_reasoning:
                print(f"异步 - 返回的思考过程/元数据:\n{a_reasoning}")
            else:
                print(f"异步 - 未找到明确的思考过程/元数据。")

            if a_usage:
                print(f"异步 - API 使用情况: Prompt Tokens: {a_usage.prompt_tokens}, Completion Tokens: {a_usage.completion_tokens}, Total Tokens: {a_usage.total_tokens}")
            else:
                print("异步 - 未获取到 API 使用情况。")

        except Exception as e:
            print(f"异步调用主程序出错: {e}")

    asyncio.run(run_async_example())