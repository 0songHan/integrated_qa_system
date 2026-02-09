import configparser
import os

class Config:
    def __init__(self,config_file='../config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.MYSQL_HOST = self.config.get('mysql','host',fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='hcs')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='rag_demo')  

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost') 
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)  
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234') 
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)  
        # 日志文件路径
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log') 

if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_DATABASE)
