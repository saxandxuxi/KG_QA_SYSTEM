def get_prompt_template(context, question):
    PROMPT_TEMPLATE = f"""  
    你是一个地理信息方面的专家，需要根据提供的全球离散格网系统（Discrete Global Grid Systems，DGGS）文档信息回答问题。请严格遵守以下规则：  
    1. 只使用提供的上下文信息回答  
    2. 不要透露任何文档来源或元数据  
    3. 如果信息不完整请明确说明  

    <上下文>  
    {context}  
    </上下文>  

    <用户问题>  
    {question}  
    </用户问题>  

    请直接给出答案：  
    """
    return PROMPT_TEMPLATE


def get_decomposition_template(question):
    """获取问题分解提示词模板"""
    return f"""您是一个分解问题的助手，需将输入问题:{question};
    拆分为最小单位、不可再拆分的子问题，作为检索词进行检索（如"DGGS是什么"）。
    要求：1. 子问题与输入问题相关；
         2. 每个不超过20字；
         3. 数量不超过5个；
         4. 用换行符'/n'分隔,不准使用其他符号；
         5. 仅输出子问题，无需其他说明。
    输出子问题："""


def get_rag_template(question, context):
    """获取RAG提示词模板"""
    return f'''你是负责回答问题的助手。请使用以下检索到的上下文信息（可能包含英文），用中文回答问题。回答需基于上下文内容，若无法从上下文中找到答案，请直接说明"不知道"。回答要简明扼要，字数不超过300字。
问题: {question}
上下文: {context}
答案：'''


def get_final_rag_template(context, question):
    """获取最终RAG提示词模板"""
    return f"""下面是一组问题+答案对:

{context}

使用上述问题+答案对来生成问题的答案: {question}
"""
