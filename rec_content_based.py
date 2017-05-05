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


def value_fix(value, prefix_map):
    '''
        add prefix for every item's value

        Parameters:
        ----------
        value: {key1:value1, key2:value2,...}, dict
        prefix_map: {key:prefix,...}, dict\

        Returns:
        --------
        new_value:  {key1:value1, key2:value2,...}, dict
    '''
    result = value
    for key, value in result['Results'].iteritems():
        prefix = prefix_map.get(key, None)
        # filter out non-prefix record in recommend result
        if prefix is None:
            result['Results'].pop(key, None)
            continue
        result['Results'][key] = prefix + ':' + result['Results'][key]
    return result


def main(data_file_path, config_file, mode=False):
    global logger
    weight = {'movie': {'language': 0.8, 'country': 0.1,
                        'tag': 0.5, 'actor': 0.1, 'director': 0.8,
                        'score': 1, 'year': 0.4},
              'tv': {'language': 0.5, 'country': 0.8,
                     'tag': 0.9, 'actor': 0.1, 'director': 0.1,
                     'score': 0.4, 'year': 0.5}}
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    if mode == 'search':
        logger.info('config: Search weight from douban')
    else:
        logger.info('config: Use history weight')
        logger.info('History weight: {0}'.format(weight))
        if mode is 'prefix':
            logger.info('load prefix map from local file')
            with open(data_file_path + r'/' + 'prefix_map' + r'.dat', 'rb') as f:
                prefixs = pickle.load(f)

    logger.info('start')
    con = init_client(config_file)
    key_pattern = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'

    for model in models:
        if model not in ['tv', 'movie']:
            continue
        data = dict()
        with open(data_file_path + r'/' + model + r'.dat', 'rb') as f:
            data = pickle.load(f)
        logger.info('load model {0} data success'.format(model))
        if len(data) != 7:
            logger.error('Error: read data of model {0}, model feature data be not matched{1}'.format(model, len(data.keys())))
            raise Exception('model feature data be not matched')
        if len(data['director'].index) < 500:
            logger.error('Error: read data of model {0}, num of record wrong'.format(model))
            raise Exception('model data be wrong')
        logger.info('Start process data of model : {0}'.format(model))
        logger.info('data feature: {0}'.format(data.keys()))
        logger.info('Start init sim handler')
        s = Sim(data)
        logger.info('Success: initial sim handler ')
        if mode == 'search':
            with open(data_file_path + r'/' + model + r'_douban.dat', 'rb') as f:
                train_dataset = pickle.load(f)
            logger.info('Success: load model {0} data from douban '.format(model))
            logger.info('Start search weight ...')
            weight, score = s.weight_search(train_dataset, patch_size=200, verbose=True)
            logger.info('Finished: weight search, socre: {0}, weight: {1}'.format(score, weight))
            continue
        else:
            s.set_weight(weight[model])

        count = 0
        try:
            for cover_id, result in s.process():
                logger.debug('{0}  {1}'.format(cover_id, result))
                # print cover_id, result
                if mode == 'prefix':
                    con.set(key_pattern + cover_id, json.dumps(value_fix(result, prefixs)))
                    con.expire(key_pattern + cover_id, 1296000)
                    count += 1
                elif mode == 'work':
                    con.set(key_pattern + cover_id, json.dumps(result))
                    con.expire(key_pattern + cover_id, 1296000)
                    count += 1
        except Exception, e:
            logger.error('catched error :{0}, processed num: {1}, model: {2}'.format(e, count, model))
            traceback.print_exc()
            raise Exception('Error')
        logger.info('model {0} has finished'.format(model))
    logger.info('Finished')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: rec_content_based <train/work/prefix>")
        exit(-1)
    data_file_path = r'./data'
    config_file = r'./etc/config.ini'
    mode = sys.argv[1]
    if mode not in ['train', 'work', 'prefix']:
        raise ValueError('mode should be one of train/work/prefix')
    main(data_file_path, config_file, mode)
