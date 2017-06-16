# encoding:utf-8
"""
==========================
calculation of similarity
==========================

use charactor 'language', 'country', 'writer', 'director', 'tag', 'actor', 'year', 'score'

To redis it is that result of calculation be saved

Created by:
    yonggang Huang
In:
    05-25-2017
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
    features = ['language', 'country', 'writer', 'director', 'tag', 'actor', 'year', 'score']
    weight = {'movie': [0.8, 0.1, 0.4, 0.8, 0.5, 0.1, 0.4, 1],
              'tv': [0.5, 0.8, 0.5, 0.1, 0.9, 0.1, 0.5, 0.4]}

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
        if model not in ['tv', 'movie']: continue
        with open(data_file_path + r'/' + model + r'.dat', 'rb') as f:
            data = pickle.load(f)
        logger.info('load model {0} data success'.format(model))
        logger.info('Start process data of model : {0}'.format(model))
        if mode == 'search':
            with open(data_file_path + r'/' + model + r'_douban.dat', 'rb') as f:
                train_dataset = pickle.load(f)
            logger.info('load {0} from douban '.format(model))
            logger.info('Start search weight ...')
            hehe = map(lambda x: x[1], data['property'])
            s = Sim(index=data['values']['cover_id'], feat_properties=hehe, std_output=train_dataset)
            data['values'].drop('cover_id', axis=1, inplace=True)
            s.fit(data['values'].values)
            logger.info('result of weight search: {0}, score: {1}'.format(s.weight, s.score))
            print s.weight, s.score
            break
        hehe = map(lambda x: x[1], data['property'])
        s = Sim(weight=weight[model], index=data['values']['cover_id'], feat_properties=hehe)
        data['values'].drop('cover_id', axis=1, inplace=True)
        s.fit(data['values'].values)
        count = 0
        try:
            for cover_id, result in s.transform():
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
        logger.info('model {0} has finished, num of record: {1}'.format(model, count))
    logger.info('Finished')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: rec_content_based <train/work/prefix>")
        exit(-1)
    data_file_path = r'./data'
    config_file = r'./etc/config.ini'
    mode = sys.argv[1]
    if mode not in ['search', 'work', 'prefix']:
        raise ValueError('mode should be one of train/work/prefix')
    main(data_file_path, config_file, mode)
