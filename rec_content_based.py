import logging
import redis
import json
import cPickle as pickle
import ConfigParser
from datetime import date, timedelta
from threading import Thread, Lock
import sim
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


def work(fun, data, weight, index, mutex, con):
    key_pattern = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'
    while index < data.index.size:
        cover_id, result = fun(data, weight, data.index[index])
        mutex.acquire()
        con.set(key_pattern + cover_id, json.dumps(result))
        index += 1
        mutex.release()


def main(file, config_file):
    global logger
    weight = {'tag': 0, 'actor': 5,
              'director': 0, 'language': 0,
              'country': 0}
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
    # key_pattern = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'
    for model, data in data_all.items():
        if model != 'movie': continue
        logger.info('start process data of model : {}'.format(model))
        logger.info('data feature: {0}'.format(data.keys()))
        for key in data.keys():
            logger.debug('features of {0}: {1}'.format(key, data[key].columns))
            logger.debug('num of record in features {0}: {1}'.format(key, len(data[key].index)))
        count = 0
        print model
        s = sim.Sim(weight, data)
        print s.work(s.data, s.weight, '5c58griiqftvq00')
        '''
        threads = []
        mutex = Lock()
        for index in s.data.index:
            count += 1
            if count == 10:
                break
            args = 
            tmp = Thread(target=work, args=(s.work, s.data, s.weight, index, mutex, con))
            threads.append(tmp)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        '''

        try:
                # con.set(key_pattern + cover_id, json.dumps(result))

            '''
            for cover_id, result in s.process():
                logger.debug('{0}  {1}'.format(cover_id, result))
                # print cover_id, result
                con.set(key_pattern + cover_id, json.dumps(result))
                count += 1
            '''
        except Exception, e:
            logger.error('catched error :{0}, processed num: {1}, model: {2}'.format(e, count, model))
            raise Exception('Error')
        logger.info('model {0} has finished'.format(model))
    logger.info('Finished')



if __name__ == '__main__':
    data_file = r'./data/all_video_info.dat'
    config_file = r'./etc/config.ini'
    main(data_file, config_file)
