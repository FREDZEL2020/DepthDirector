import redis
import os
import sys
import json
from multiprocessing import Pool
import subprocess
from tqdm import tqdm

class REDIS(object):
    TO_BE_PROCESSED_KEY: str
    FAILED_KEY: str
    BUILT_KEY: str
    DONE_KEY: str

    def __init__(self, REDIS_URL, REDIS_PASSWD, TO_BE_PROCESSED_KEY, FAILED_KEY, BUILT_KEY, DONE_KEY, db=30) -> None:
        REDIS_DB = db
        redis_inst_pool = redis.ConnectionPool.from_url(
            REDIS_URL, max_connections=20, password=REDIS_PASSWD, encoding="utf-8", decode_responses=True, db=REDIS_DB)
        self.redis_inst = redis.Redis(connection_pool=redis_inst_pool)

        self.TO_BE_PROCESSED_KEY = TO_BE_PROCESSED_KEY
        self.FAILED_KEY = FAILED_KEY
        self.DONE_KEY = DONE_KEY
        self.BUILT_KEY = BUILT_KEY

    def submit(self, data, num_workers=10):
        if not isinstance(data, list):
            data = [data]

        with self.redis_inst.pipeline() as pipe:
            for d in tqdm(data):
                pipe.sadd(self.TO_BE_PROCESSED_KEY, json.dumps(d))
            pipe.execute()
    
    def pop(self, num):
        return self.redis_inst.spop(self.TO_BE_PROCESSED_KEY, num)
    
    def add(self, data, status):
        if status=="success":
            key = self.DONE_KEY
        elif status=="built":
            key = self.BUILT_KEY
        else:
            key = self.FAILED_KEY

        self.redis_inst.sadd(key, json.dumps(data))
 