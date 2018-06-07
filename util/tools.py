# encoding:utf-8
from pymongo import MongoClient
import ConfigParser
import logging
import redis
import json


def get_redis_client(config):
    cf = ConfigParser.ConfigParser()
    cf.read(config)
    address = cf.get('redis', 'address')
    port = int(cf.get('redis', 'port'))
    client = redis.StrictRedis(address, port)
    return client


def get_collection(config):
    """
    get connection from mongoDB
    """
    cf = ConfigParser.ConfigParser()
    cf.read(config)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')

    con = MongoClient(address, port)
    EPGInfo = con[database]
    EPGInfo.authenticate(username, password)
    collection = EPGInfo[collection]
    return collection


def get_logger(filename, debug=False):
    format = '''[ %(levelname)s %(asctime)s @ %(process)d] (%(filename)s:%(lineno)d) - %(message)s'''
    formatter = logging.Formatter(format)
    logger = logging.getLogger("vav_result")
    handler = logging.FileHandler(filename, 'a')
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def write_into_redis(key_pattern, item_dict, verbose=True):

    proxy_ip = '10.66.1.168'
    proxy_port = 19000

    client = redis.Redis(host=proxy_ip, port=proxy_port)
    cnt = 0
    result = dict()
    result["V"] = "0.1.0"
    for k, rs in item_dict.items():
        key = key_pattern.format(k)
        result["Results"] = rs
        if client.setex(key, json.dumps(result), 2592000) is not True:
            print("%s\t%s" % (key, result))
        if cnt % 100 == 0 and verbose is True:
            print('num of handled record: {0}'.format(cnt))
        cnt += 1
    return cnt


def show_item_info(items, projection, config="./etc/config.ini"):
    """
    items: dict, key is item id, value is weight
    projection: dict, key is property name,
            value be 1 if wanted else 0
    """
    collection = get_collection()
    for id_ in items:
        try:
            info = collection.find({"pp_tencent.tencentId": id_}, projection)[0]
            print(info["name"]),
        except Exception, e:
            print(id_, e)
        print(items[id_]),
        print(id_)
