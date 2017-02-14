import logging
import redis
import json
import cPickle as pickle
import ConfigParser
from datetime import date, timedelta
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
    for model, dataframe in data_all.items():
        logger.info('start process data of model : {}'.format(model))
        logger.info('data record size : {0}'.format(dataframe.index.size))
        logger.info('data feature: {0}'.format(dataframe.columns.tolist()))
        logger.debug('data : {0}'.format(dataframe.values.tolist()))
        y = dataframe['grade_score'].values.astype(float)
        dataframe.drop('grade_score', inplace=True, axis=1)
        X = dataframe.values.astype(float)
        count = 0
        s = sim.Sim(X, y, dataframe.index)
        try:
            for cover_id, result in s.process():
                logger.debug('{0}  {1}'.format(cover_id, result))
                # print cover_id, result
                con.set(key_pattern + cover_id, json.dumps(result))
                count += 1
        except Exception, e:
            logger.error('catched error :{0}, processed num: {1}, model: {2}'.format(e, count, model))
            raise
        logger.info('model {0} has finished'.format(model))
    logger.info('Finished')


if __name__ == '__main__':
    data_file = './data/id.dat'
    config_file = './etc/config.ini'
    main(data_file, config_file)
