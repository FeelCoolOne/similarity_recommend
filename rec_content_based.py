# encoding:utf-8
"""
==========================
calculation of similarity
==========================

use charactor  'tag', 'director', 'country',
    'actor', 'language', 'year', 'score'
To redis it is that result of calculation be saved

Created by:
    yonggang Huang
In:
    03-31-2017
"""
import logging
import redis
import json
import cPickle as pickle
import ConfigParser
from datetime import date, timedelta
from main.sim import Sim
import sys
import traceback
reload(sys)
sys.setdefaultencoding('utf-8')


def set_logger():
    format = '''[ %(levelname)s %(asctime)s @ %(process)d] (%(filename)s:%(lineno)d) - %(message)s'''
    curDate = date.today() - timedelta(days=0)
    log_name = './log/similar_{0}.log'.format(curDate)
    formatter = logging.Formatter(format)
    logger = logging.getLogger("vav_result")
    handler = logging.FileHandler(log_name, 'a')
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = set_logger()


def init_client(config_file_path):
    global logger
    cf = ConfigParser.ConfigParser()
    cf.read(config_file_path)
    address = cf.get('redis', 'address')
    port = int(cf.get('redis', 'port'))
    logger.info('connect redis: {0}:{1}'.format(address, port))
    try:
        client = redis.StrictRedis(address, port)
    except Exception, e:
        logger.error('connect redis fail: {0}'.format(e))
        sys.exit()
    else:
        logger.info('connection success')
        return client


def main(data_file_path, config_file):
    global logger
    weight = {'tag': 20, 'actor': 10,
              'director': 10000, 'language': 10,
              'country': 200}
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']

    logger.info('Start')
    con = init_client(config_file)
    key_pattern = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'

    for model in models:
        if model not in ['tv', 'movie']:
            continue
        data = dict()
        with open(data_file_path + r'/' + model + r'.dat', 'rb') as f:
            data = pickle.load(f)
        logger.info('load model {0} data success'.format(model))
        with open(data_file_path + r'/' + model + r'_douban.dat', 'rb') as f:
            train_dataset = pickle.load(f)
        logger.info('load model {0} data from douban success'.format(model))
        if len(data) != 7:
            logger.error('Error: read data of model {0}, model feature data be not matched{1}'.format(model, len(data.keys())))
            raise Exception('model feature data be not matched')
        if len(data['director'].index) < 500:
            logger.error('Error: read data of model {0}, num of record wrong'.format(model))
            raise Exception('model data be wrong')
        logger.info('start process data of model : {0}'.format(model))
        logger.info('data feature: {0}'.format(data.keys()))
        logger.info('start init sim handler')
        s = Sim(data)
        logger.info('sim handler success')
        logger.info('start search weight ...')
        weight, score = s.weight_search(train_dataset, patch_size=200, verbose=True)
        logger.info('weight search finish, socre: {0}, weight: {1}'.format(score, weight))

        # s.set_weight(weight)
        count = 0
        try:
            for cover_id, result in s.process():
                logger.debug('{0}  {1}'.format(cover_id, result))
                print cover_id, result
                # con.set(key_pattern + cover_id, json.dumps(result))
                # con.expire(key_pattern + cover_id, 1296000)
                count += 1
                if count == 10:
                    raise()
        except Exception, e:
            logger.error('catched error :{0}, processed num: {1}, model: {2}'.format(e, count, model))
            traceback.print_exc()
            raise Exception('Error')
        logger.info('model {0} has finished'.format(model))
    logger.info('Finished')


if __name__ == '__main__':
    data_file_path = r'./data'
    config_file = r'./etc/config.ini'
    main(data_file_path, config_file)
