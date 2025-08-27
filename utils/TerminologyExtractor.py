import re
import os
from collections import Counter
from typing import List, Dict, Set
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import jieba
import jieba.posseg as pseg


class TerminologyExtractor:
    """
    从PDF文档中提取术语库的工具类
    """

    def __init__(self, config):
        """
        初始化术语提取器

        Args:
            config: 配置对象，需要包含data_dir, chunk_size, chunk_overlap等属性
        """
        self.config = config
        self.technical_terms = set()
        self.term_frequency = Counter()

        # 定义常见的技术术语词性
        self.technical_pos = {'n', 'vn', 'nz', 'nt', 'nl'}  # 名词、动名词、其他专有名词、机构团体名、名词性惯用语

        # 常见的英文技术术语模式
        self.technical_patterns = [
            r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b',  # 驼峰命名，如GeoTIFF
            r'\b[A-Z]{2,}\b',  # 全大写缩写，如DGGS
            r'\b[A-Z][A-Z0-9]{2,}\b',  # 大写字母和数字组合
            r'\b[a-z]+[-][a-z]+\b',  # 连字符连接的术语，如discrete-global\

        ]

        # 已知的技术术语列表（可以扩展）
        self.known_terms = {
            'DGGS', 'GeoTIFF', 'coordinate', 'transformation', 'compatibility',
            'conflict', 'grid', 'cell', 'resolution', 'projection', 'CRS',
            'discrete', 'global', 'system', 'model', 'algorithm', 'encoding',
            'decoding', 'interoperability', 'standard', 'specification'
        }

    def load_and_split_documents(self) -> List:
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
            print(f"成功加载{len(documents)}个PDF文件")
        except Exception as e:
            print(f"加载PDF文件时出错: {e}")
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

    def extract_english_terms(self, text: str) -> Set[str]:
        """
        从文本中提取英文术语

        Args:
            text: 输入文本

        Returns:
            Set[str]: 提取的英文术语集合
        """
        terms = set()

        # 使用预定义的模式提取术语
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)

        # 过滤掉太短或太长的术语
        filtered_terms = {term for term in terms if 2 <= len(term) <= 30}

        return filtered_terms

    def extract_chinese_terms(self, text: str) -> Set[str]:
        """
        从文本中提取中文术语

        Args:
            text: 输入文本

        Returns:
            Set[str]: 提取的中文术语集合
        """
        terms = set()

        # 使用jieba进行分词和词性标注
        words = pseg.cut(text)

        for word, pos in words:
            # 如果词性是技术相关词性且长度合适
            if pos in self.technical_pos and 2 <= len(word) <= 10:
                # 过滤掉常见的停用词
                if not self._is_stop_word(word):
                    terms.add(word)

        return terms

    def _is_stop_word(self, word: str) -> bool:
        """
        判断是否为停用词

        Args:
            word: 待判断的词

        Returns:
            bool: 是否为停用词
        """
        stop_words = {
            '问题', '方法', '技术', '系统', '应用', '研究', '分析', '设计', '实现',
            '基于', '使用', '通过', '可以', '能够', '提供', '支持', '包括', '以及'
        }
        return word in stop_words

    def extract_terms_from_documents(self) -> Dict[str, int]:
        """
        从所有文档中提取术语并统计频率

        Returns:
            Dict[str, int]: 术语及其频率的字典
        """
        # 加载并分割文档
        documents = self.load_and_split_documents()

        print(f"开始从{len(documents)}个文档片段中提取术语...")

        # 遍历所有文档片段提取术语
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(documents)} 个文档片段")

            content = doc.page_content

            # 提取英文术语
            english_terms = self.extract_english_terms(content)

            # 提取中文术语
            chinese_terms = self.extract_chinese_terms(content)

            # 合并术语
            all_terms = english_terms.union(chinese_terms)

            # 更新术语频率统计
            for term in all_terms:
                self.term_frequency[term] += 1

        print(f"术语提取完成，共提取到 {len(self.term_frequency)} 个唯一术语")
        return dict(self.term_frequency)

    def filter_terms_by_frequency(self, min_frequency: int = 2) -> Dict[str, int]:
        """
        根据频率过滤术语

        Args:
            min_frequency: 最小出现频率

        Returns:
            Dict[str, int]: 过滤后的术语及其频率
        """
        filtered_terms = {
            term: freq for term, freq in self.term_frequency.items()
            if freq >= min_frequency
        }
        return filtered_terms

    def save_terminology_library(self, filepath: str, min_frequency: int = 2):
        """
        保存术语库到文件

        Args:
            filepath: 保存文件路径
            min_frequency: 最小出现频率
        """
        # 过滤术语
        filtered_terms = self.filter_terms_by_frequency(min_frequency)

        # 按频率排序
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 术语库\n")
            f.write("# 术语\t频率\n")
            for term, freq in sorted_terms:
                f.write(f"{term}\t{freq}\n")

        print(f"术语库已保存到 {filepath}，包含 {len(sorted_terms)} 个术语")

    def get_top_terms(self, n: int = 100) -> List[tuple]:
        """
        获取频率最高的前N个术语

        Args:
            n: 返回术语数量

        Returns:
            List[tuple]: 术语及其频率的列表
        """
        return self.term_frequency.most_common(n)


# 使用示例
if __name__ == "__main__":
    from Config.config import Config

    # 创建配置对象
    config = Config()

    # 创建术语提取器
    extractor = TerminologyExtractor(config)

    # 提取术语
    try:
        terms = extractor.extract_terms_from_documents()
        print(f"总共提取到 {len(terms)} 个术语")

        # 显示频率最高的术语
        top_terms = extractor.get_top_terms(20)
        print("\n频率最高的20个术语:")
        for term, freq in top_terms:
            print(f"{term}: {freq}")

        # 保存术语库
        extractor.save_terminology_library("../Terms/terminology_library.txt", min_frequency=2)

    except Exception as e:
        print(f"提取术语时出错: {e}")
