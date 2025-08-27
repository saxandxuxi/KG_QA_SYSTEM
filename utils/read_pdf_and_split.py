import PyPDF2
import re


def read_pdf_and_split(file_path, chunk_size=500, overlap=50):
    """
    读取PDF文件内容并分割为指定长度的文本片段

    参数:
        file_path (str): PDF文件路径
        chunk_size (int): 每个文本片段的大致长度（字符数）
        overlap (int): 相邻片段的重叠字符数，增强上下文连贯性

    返回:
        list: 分割后的文本片段列表
    """
    # 读取PDF内容
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # 遍历所有页面并提取文本
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # 文本预处理：去除多余空行和空格
        text = re.sub(r'\n+', '\n', text).strip()
        text = re.sub(r' +', ' ', text)

        # 文本分割
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算当前片段结束位置
            end = start + chunk_size
            # 确保不超过文本总长度
            if end > text_length:
                end = text_length

            # 提取当前片段
            chunk = text[start:end]
            chunks.append(chunk)

            # 计算下一段的起始位置（考虑重叠）
            start = end - overlap
            # 避免重复处理（当剩余文本小于重叠长度时）
            if start >= text_length:
                break

        return chunks

    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        return []


def remove_pages_and_save(input_path, output_path, start_page, end_page):
    """
    删除PDF文件中指定范围的页面并保存为新文件

    参数:
        input_path (str): 输入PDF文件路径
        output_path (str): 输出PDF文件路径
        start_page (int): 要删除的起始页码（从1开始）
        end_page (int): 要删除的结束页码（从1开始）
    """
    try:
        # 读取输入PDF文件
        with open(input_path, 'rb') as input_file:
            reader = PyPDF2.PdfReader(input_file)
            writer = PyPDF2.PdfWriter()

            # 遍历所有页面
            for i in range(len(reader.pages)):
                page_num = i + 1  # 页码从1开始计算
                # 如果当前页面不在要删除的范围内，则添加到输出文件中
                if page_num < start_page or page_num > end_page:
                    writer.add_page(reader.pages[i])

            # 将结果写入新文件
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

        print(f"成功删除第{start_page}到{end_page}页，已保存到: {output_path}")
        return True

    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        return False


# 使用示例
if __name__ == "__main__":
    # 原有功能示例
    pdf_path = "../KG_Base/OGC-API-DISCRETE-GLOBAL-GRID-SYSTEMS.pdf"  # 替换为你的PDF文件路径
    # segments = read_pdf_and_split(pdf_path, chunk_size=600, overlap=100)
    #
    # print(f"分割完成，共得到 {len(segments)} 个文本片段：")
    # for i, seg in enumerate(segments, 1):
    #     print(f"\n片段 {i}（长度：{len(seg)}）：")
    #     print(seg[:100] + "...")  # 只显示前100个字符

    # 新功能示例 - 删除第3到17页并保存
    remove_pages_and_save(pdf_path, "../KG_Base/OGC-API-DISCRETE-GLOBAL-GRID-SYSTEMS.pdf", 3, 17)
