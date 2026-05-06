import re
import jieba
from base.logger import logger


class Preprocessor:
    def __init__(self):
        self.logger = logger
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self):
        default_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '他', '她', '吗', '什么',
            '那', '里', '又', '把', '为', '而', '等', '但', '与', '及', '等',
            '被', '从', '它', '们', '之', '对', '中', '可以', '可', '能',
            '得', '地', '过', '么', '些', '还', '将', '此', '其', '或'
        }
        self.logger.info(f'加载停用词 {len(default_stop_words)} 个')
        return default_stop_words

    def clean_text(self, text):
        if not text:
            return ''
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        words = jieba.lcut(cleaned)
        words = [w.strip() for w in words if w.strip() and w not in self.stop_words]
        return words

    def load_stop_words_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = set(line.strip() for line in f if line.strip())
            self.stop_words = words
            self.logger.info(f'从文件加载停用词 {len(words)} 个')
        except FileNotFoundError as e:
            self.logger.error(f'停用词文件不存在：{e}')
            raise

    def normalize_text(self, text):
        text = self.clean_text(text)
        text = text.lower()
        return text


if __name__ == '__main__':
    preprocessor = Preprocessor()
    print(preprocessor.tokenize('什么是机器学习？机器学习是人工智能的一个分支。'))
    print(preprocessor.clean_text('<p>测试文本</p> https://example.com 去掉特殊字符！'))