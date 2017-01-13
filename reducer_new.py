# -*- coding: utf-8 -*-
import numpy as np
# from pandas import DataFrame
import sim
import cPickle as pickle
import logging
import redis
import json
import ConfigParser
from datetime import date, timedelta
import sys
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


weights_all = {}
weights_all['cartoon'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['doc'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['education'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['entertainment'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['movie'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['sports'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['tv'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['variety'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
model_handler = {}
model_handler['cartoon'] = sim.Cartoon_Sim
model_handler['doc'] = sim.Doc_Sim
model_handler['education'] = sim.Educatiion_Sim
model_handler['entertainment'] = sim.Educatiion_Sim
model_handler['movie'] = sim.Movie_Sim
model_handler['sports'] = sim.Sports_Sim
model_handler['tv'] = sim.TV_Sim
model_handler['variety'] = sim.Variety_Sim
'''
models = ['cartoon', 'doc', 'education', 'entertainment', 'movie', 'sports', 'tv', 'variety']
features = ['id', 'model', 'year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
'''


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


def main(file, config_file):
    global logger
    data_all = {}
    logger.info('Start')
    con = init_client(config_file)
    logger.info('start read data file: {0}'.format(file))
    with open(file, 'rb') as f:
        data_all = pickle.load(f)
    logger.info('finished read data file')
    if len(data_all) == 0:
        logger.error('no data')
        return False
    key_pattern = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'
    for model, data_frame in data_all.items():
        logger.info('start process data of model {}:'.format(model))
        logger.info('data record size is {0}'.format(data_frame.index.size))
        logger.info('data feature: {0}'.format(data_frame.columns.tolist()))
        logger.debug('data : {0}'.format(data_frame.values.tolist()))
        model_sim = model_handler[model](data_frame, weights_all[model])
        count = 0
        try:
            for cover_id, result in model_sim.process():
                logger.debug('{0}  {1}'.format(cover_id, result))
                # print cover_id, result
                con.set(key_pattern + cover_id, json.dumps(result))
                count += 1
        except Exception, e:
            logger.error('catched error :{0}, processed num: {1}, model: {2}'.format(e, count, model))
        logger.info('model {0} has finished'.format(model))
    logger.info('Finished')
    return True


if __name__ == '__main__':
    data_file = './data/all_video_info.dat'
    config_file = './etc/config.ini'
    main(data_file, config_file)
