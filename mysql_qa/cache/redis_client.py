import json
import redis
from base.config import Config
from base.logger import logger


class RedisClient:
    def __init__(self):
        self.logger = logger
        self.client = None
        self._connect()

    def _connect(self):
        try:
            config = Config()
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.client.ping()
            self.logger.info('Redis 连接成功')
        except redis.RedisError as e:
            self.logger.error(f'Redis 连接失败：{e}')
            raise

    def set_cache(self, key, value, expire_seconds=3600):
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            self.client.setex(key, expire_seconds, serialized)
            self.logger.info(f'缓存写入成功：{key}')
            return True
        except redis.RedisError as e:
            self.logger.error(f'缓存写入失败：{e}')
            return False

    def get_cache(self, key):
        try:
            value = self.client.get(key)
            if value is None:
                self.logger.info(f'缓存未命中：{key}')
                return None
            result = json.loads(value)
            self.logger.info(f'缓存命中：{key}')
            return result
        except redis.RedisError as e:
            self.logger.error(f'缓存读取失败：{e}')
            return None

    def delete_cache(self, key):
        try:
            self.client.delete(key)
            self.logger.info(f'缓存删除成功：{key}')
            return True
        except redis.RedisError as e:
            self.logger.error(f'缓存删除失败：{e}')
            return False

    def batch_set_cache(self, items, expire_seconds=3600):
        try:
            pipe = self.client.pipeline()
            for key, value in items:
                serialized = json.dumps(value, ensure_ascii=False)
                pipe.setex(key, expire_seconds, serialized)
            pipe.execute()
            self.logger.info(f'批量缓存写入成功：{len(items)} 条')
            return True
        except redis.RedisError as e:
            self.logger.error(f'批量缓存写入失败：{e}')
            return False

    def exists(self, key):
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            self.logger.error(f'缓存检查失败：{e}')
            return False

    def close(self):
        if self.client:
            self.client.close()
            self.logger.info('Redis 连接已关闭')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    with RedisClient() as client:
        client.set_cache('test_key', {'question': 'test', 'answer': 'hello'})
        result = client.get_cache('test_key')
        print(result)