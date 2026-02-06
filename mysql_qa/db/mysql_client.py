
import pymysql
import pandas as pd
from base import Config
from base.logger import logger


class MySQLClient:
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                database=Config().MYSQL_DATABASE
            )
            self.cursor = self.connection.cursor()
            self.logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL 连接失败：{e}')
            raise

if __name__ == '__main__':
    mysql_client = MySQLClient()