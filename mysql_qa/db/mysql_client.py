import pymysql
import pandas as pd
from base.config import Config
from base.logger import logger


class MySQLClient:
    def __init__(self):
        self.logger = logger
        self.connection = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            config = Config()
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
            self.logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL 连接失败：{e}')
            raise

    def execute_query(self, sql, params=None):
        try:
            self.cursor.execute(sql, params)
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            self.logger.error(f'查询执行失败：{e}')
            self.connection.rollback()
            raise

    def execute_update(self, sql, params=None):
        try:
            self.cursor.execute(sql, params)
            self.connection.commit()
            return self.cursor.rowcount
        except pymysql.MySQLError as e:
            self.logger.error(f'更新执行失败：{e}')
            self.connection.rollback()
            raise

    def query_to_dataframe(self, sql, params=None):
        try:
            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame()
        except pymysql.MySQLError as e:
            self.logger.error(f'查询转为DataFrame失败：{e}')
            raise

    def insert_batch(self, table, columns, data):
        column_str = ', '.join(columns)
        placeholder = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO {table} ({column_str}) VALUES ({placeholder})"
        try:
            self.cursor.executemany(sql, data)
            self.connection.commit()
            self.logger.info(f'批量插入 {self.cursor.rowcount} 条数据到 {table}')
            return self.cursor.rowcount
        except pymysql.MySQLError as e:
            self.logger.error(f'批量插入失败：{e}')
            self.connection.rollback()
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.logger.info('MySQL 连接已关闭')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    with MySQLClient() as client:
        result = client.execute_query("SHOW TABLES")
        print(result)