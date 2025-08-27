import re
import jieba
import jieba.posseg as pseg  # 用于词性标注


# 确保结巴分词已加载词典（可选：添加领域专属词典提升分词准确性）
# jieba.load_userdict("domain_dict.txt")  # 若有领域术语，可加载自定义词典

class KeywordExtractor:
    def __init__(self):
        # 扩展停用词表：包含疑问词、虚词、常见冗余词
        self.stop_words = {
            # 疑问词
            '什么', '如何', '为什么', '怎么', '哪里', '哪个', '是否', '可以', '怎样', '怎么样', '吗', '呢', '吧',
            # 虚词/助词
            '的', '得', '地', '了', '着', '过', '在', '是', '有', '就', '也', '还', '都', '只', '共',
            # 冗余修饰词
            '非常', '极其', '特别', '一定', '可能', '大概', '也许', '比如', '例如', '等等', '诸多','与'
        }

        # 保留核心词性（参考结巴词性编码：n=名词, v=动词, vn=动名词, adj=形容词, ns=地名, nr=人名, nz=其他专有名词）
        self.keep_pos = {'n', 'v', 'vn', 'adj', 'ns', 'nr', 'nz'}

    def _extract_keywords(self, question: str) -> str:
        """从用户问题中提取精准关键词，提升检索相关性"""
        if not question.strip():
            return ""

        # 1. 预处理：移除标点和多余空格
        processed = re.sub(r'[^\w\s]', '', question).strip()
        if not processed:
            return question

        # 2. 分词并标注词性
        words = pseg.cut(processed)  # 返回 (词, 词性) 元组

        # 3. 过滤：保留核心词性且非停用词的词
        filtered = []
        for word, pos in words:
            if word not in self.stop_words and pos in self.keep_pos:
                filtered.append((word, pos))

        if not filtered:
            return processed  # 若过滤后为空，返回预处理后的原问题

        # 4. 合并连续核心词为短语（提升语义完整性，如"离散全球网格系统"而非单个词）
        phrases = []
        current_phrase = []
        for word, pos in filtered:
            # 连续名词/动词/形容词合并为短语
            if pos in {'n', 'v', 'vn', 'adj'}:
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrases.append(''.join(current_phrase))
                    current_phrase = []
                phrases.append(word)
        # 处理最后一组短语
        if current_phrase:
            phrases.append(''.join(current_phrase))

        # 5. 去重（保留首次出现的词）
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)

        # 6. 最终关键词：用空格连接短语（适配多数检索引擎的分词习惯）
        keywords = ' '.join(unique_phrases)

        return keywords if keywords else question


# 使用示例
if __name__ == "__main__":
    extractor = KeywordExtractor()
    test_questions = [
        "什么是离散全球网格系统（DGGS）的核心数据模型？",
        "如何解决DGGS与GeoTIFF在坐标转换中的兼容性冲突？",
        "为什么网格单元的邻接关系判断需要特殊处理赤道区域？"
    ]
    for q in test_questions:
        print(f"原问题：{q}")
        print(f"提取关键词：{extractor._extract_keywords(q)}\n")
