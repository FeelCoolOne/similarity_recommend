# -*- coding: utf-8 -*-

from pymongo import MongoClient
from pandas import DataFrame
import cPickle as pickle
import ConfigParser
from datetime import date, timedelta
import logging


def set_logger():
    format = '''[ %(levelname)s %(asctime)s @ %(process)d] (%(filename)s:%(lineno)d) - %(message)s'''
    curDate = date.today() - timedelta(days=0)
    log_name = './log/get_mongo_{0}.log'.format(curDate)
    formatter = logging.Formatter(format)
    logger = logging.getLogger("vav_result")
    handler = logging.FileHandler(log_name, 'a')
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = set_logger()

features_handler = {}
features_handler['cartoon'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['doc'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['education'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['entertainment'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['movie'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['sports'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['tv'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['variety'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']


def init_con(config_file_path):
    global logger
    cf = ConfigParser.ConfigParser()
    cf.read(config_file_path)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    logger.info('connect mongo: {0}:{1} by user {2}'.format(address, port, username))
    logger.info('database : {0}, collection : {1}'.format(database, collection))
    con = MongoClient(address, port)
    EPGInfo = con[database]
    EPGInfo.authenticate(username, password)
    logger.debug('authentication success')
    collection = EPGInfo[collection]
    logger.debug('success to collection')
    # document = collection.find_one({'model': 'movie'})
    return collection


def get_documents(collection, condition, num=10):
    documents = collection.find(condition).limit(num)
    # documents = collection.find(condition)
    return documents


def filter_id(document):
    '''filte other attribution but id
    -->-1: not exist, other be video id'''
    record = document
    id = '-1'
    if record['tencent'] == '1':
        id = record['pp_tencent']['tencentId']
    elif record['youpeng'] == '1':
        id = record['pp_youpeng']['yp_id']
    return id


def filter_pay_status(document):
    record = document
    vip = 0
    if record['tencent'] == '1' and record['pp_tencent']['pay_status'].strip() in ['用券', '会员点播', '会员免费']:
        vip = 1
    elif record['youpeng'] == '1':
        vip = 1
    else:
        vip = 0
    return vip


def filetr_year(document):
    year = document.get('year', 'None')
    if year == 'None':
        year = document.get('issue', 'None').split('-')[0]
    return year


def fileter_tag(document):
    tag = document.get('tag', 'None').split('/')
    return tag


def handle_all_attr(document):
    data = {}
    document = dict(document)
    data['cover_id'] = filter_id(document)
    data['model'] = document.get('model', 'empty')
    data['alias'] = document.get('alias', 'empty')
    data['duration'] = str(document.get('duration', -1))
    data['enname'] = document.get('enName', 'empty')
    data['language'] = document.get('language', 'empty')
    data['name'] = document.get('name', 'empty')
    # data['issue'] = document.get('issue', '0000-00-00')
    data['director'] = str(document.get('director', 'empty').split('/'))
    data['actor'] = str(document.get('cast', 'empty').split('/'))
    data['grade_score'] = str(document.get('grade_score', -1))
    data['tag'] = str(fileter_tag(document))
    data['country'] = str(document.get('country', 'empty').split('/'))
    # TODO data['country_group'] = document.get('country_group', '[]')
    data['episodes'] = str(document.get('episodes', -1))
    data['definitions'] = str(document.get('definitions', -1))
    data['writer'] = str(document.get('writer', 'empty').split('/'))
    data['year'] = str(filetr_year(document))
    # 上架
    data['enable'] = document.get('enable', '-1')
    # 碎视频
    data['isClip'] = document.get('isClip', '-1')
    data['categorys'] = document.get('categorys', '')
    data['pp_tencent'] = document.get('pp_tencent', '{}')
    data['pp_iqiyi'] = document.get('pp_iqiyi', '{}')
    data['pp_youpeng'] = document.get('pp_youpeng', '{}')
    data['tencent'] = document.get('tencent', '-1')
    data['iqiyi'] = document.get('iqiyi', '-1')
    data['youpeng'] = document.get('youpeng', '-1')
    data['focus'] = document.get('focus', 'empty')
    data['vip'] = str(filter_pay_status(document))
    return data


# TODO
# how to process None feature
def format_input_record(collection, model):
    global logger
    # TODO
    # tag diff category
    # duration equal 0
    data_stack = []
    id_stack = []
    logger.info('model {0} features: {1}'.format(model, features_handler[model]))
    for document in collection:
        data = handle_all_attr(document)
        if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
            continue
        record = []

        for feature in features_handler[model]:
            if data[feature].strip() == '':
                data[feature] = 'empty'
            record.append(data[feature])
        logger.debug('format data, cover_id : {0}, data : {0}'.format(data['cover_id'], record))
        id_stack.append(data['cover_id'])
        data_stack.append(record)
    return DataFrame(data_stack, index=id_stack, columns=features_handler[model])


def main(config_file, data_file):
    global logger
    logger.info('Start:')
    logger.info('')
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    logger.info('all models: {0}'.format(models))
    logger.info('init connection to mongodb')
    collection = init_con(config_file)
    logger.debug('success to connect to mongodb')
    data = {}
    for model in models:
        logger.info('get collections of model : {0}'.format(model))
        documents = get_documents(collection, {'model': model}, 5)
        logger.debug('success to get data of model {0}'.format(model))
        logger.debug('start format data of model : {0}'.format(model))
        data[model] = format_input_record(documents, model)
        logger.debug('success format data of model : {0}'.format(model))
    logger.debug('store data to file: {0}'.format(data_file))
    with open(data_file, 'wb') as f:
        pickle.dump(data, f, protocol=True)
    logger.info("stored data to file: {0}".format(data_file))
    return data


if __name__ == '__main__':
    config_file = './etc/config.ini'
    data_file = './data/all_video_info.dat'
    data = main(config_file, data_file)
    # print data['tv']

    # for key in data:
    #     print data[key]
