import jieba
from rank_bm25 import BM25Okapi
from base.logger import logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.utils.preprocess import Preprocessor


class BM25Search:
    def __init__(self, table='qa_pairs', question_col='question', answer_col='answer'):
        self.logger = logger
        self.table = table
        self.question_col = question_col
        self.answer_col = answer_col
        self.preprocessor = Preprocessor()
        self.bm25 = None
        self.corpus = []
        self.raw_data = []
        self._load_and_build()

    def _load_and_build(self):
        try:
            with MySQLClient() as client:
                sql = f"SELECT {self.question_col}, {self.answer_col} FROM {self.table}"
                df = client.query_to_dataframe(sql)
                if df.empty:
                    self.logger.warning('MySQL 中无 QA 数据')
                    return
                self.raw_data = df.to_dict('records')
                self.corpus = df[self.question_col].tolist()
                tokenized_corpus = [self.preprocessor.tokenize(text) for text in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.logger.info(f'BM25 模型构建成功，共 {len(self.corpus)} 条数据')
        except Exception as e:
            self.logger.error(f'BM25 模型构建失败：{e}')
            raise

    def search(self, query, top_k=5):
        if self.bm25 is None or not self.corpus:
            self.logger.warning('BM25 模型未初始化')
            return []
        tokenized_query = self.preprocessor.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'question': self.raw_data[idx][self.question_col],
                    'answer': self.raw_data[idx][self.answer_col],
                    'score': float(scores[idx])
                })
        self.logger.info(f'BM25 搜索完成，查询：{query}，返回 {len(results)} 条结果')
        return results

    def rebuild(self):
        self.logger.info('重新构建 BM25 模型...')
        self._load_and_build()


if __name__ == '__main__':
    searcher = BM25Search()
    results = searcher.search('什么是机器学习', top_k=3)
    for r in results:
        print(r)