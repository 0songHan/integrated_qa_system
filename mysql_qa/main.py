import hashlib
from base.logger import logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search


class MySQLQA:
    def __init__(self, table='qa_pairs', question_col='question', answer_col='answer'):
        self.logger = logger
        self.redis_client = RedisClient()
        self.bm25_searcher = BM25Search(
            table=table,
            question_col=question_col,
            answer_col=answer_col
        )
        self.cache_expire = 3600

    def _generate_cache_key(self, question):
        md5 = hashlib.md5(question.encode('utf-8')).hexdigest()
        return f'qa:{md5}'

    def query(self, question, top_k=5):
        cache_key = self._generate_cache_key(question)
        cached = self.redis_client.get_cache(cache_key)
        if cached is not None:
            self.logger.info(f'命中缓存，查询：{question}')
            return cached

        results = self.bm25_searcher.search(question, top_k=top_k)
        if results:
            self.redis_client.set_cache(cache_key, results, self.cache_expire)
            self.logger.info(f'BM25 搜索完成并缓存，查询：{question}')
        else:
            self.logger.warning(f'未找到相关答案，查询：{question}')

        return results

    def rebuild_index(self):
        self.bm25_searcher.rebuild()
        self.logger.info('索引重建完成')

    def close(self):
        self.redis_client.close()
        self.logger.info('QA 系统资源释放完成')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    with MySQLQA() as qa:
        results = qa.query('什么是机器学习', top_k=3)
        for r in results:
            print(f"问题: {r['question']}")
            print(f"答案: {r['answer']}")
            print(f"分数: {r['score']}")
            print('---')