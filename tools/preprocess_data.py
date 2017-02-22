# encoding:utf-8
# get data id group by model
# and save to local file '../data/id.dat'
from pymongo import MongoClient
import logging
from datetime import date, timedelta
import ConfigParser
import cPickle as pickle
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
# import pandas as pd
import numpy as np


import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Video(object):

    def __init__(self):
        self.model_features = dict()
        self.features_trans = dict()
        self.logger = self.set_logger('../log')
        self.init_model_features()
        self.models = ['movie', 'tv',
                       'sports', 'entertainment', 'variety',
                       'education', 'doc', 'cartoon']

    def set_logger(self, path):
        format = '''[ %(levelname)s %(asctime)s @ %(process)d] (%(filename)s:%(lineno)d) - %(message)s'''
        curDate = date.today() - timedelta(days=0)
        log_name = '{0}/get_raw_data_{1}.log'.format(path, curDate)
        formatter = logging.Formatter(format)
        logger = logging.getLogger("vav_result")
        handler = logging.FileHandler(log_name, 'a')
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def connect_mongodb(
            self,
            host,
            port,
            username,
            password,
            database,
            collection):
        con = MongoClient(host, port)
        EPGInfo = con[database]
        EPGInfo.authenticate(username, password)
        collection = EPGInfo[collection]
        self.collection = collection

    def init_model_features(self):
        '''default features to be handled in process'''
        self.model_features['cartoon'] = ['tag', 'director', 'country',
                                          'actor', 'language']
        self.model_features['doc'] = ['tag', 'director', 'country',
                                      'actor', 'language']
        self.model_features['education'] = ['tag', 'director', 'country',
                                            'actor', 'language']
        self.model_features['entertainment'] = ['tag', 'director', 'country',
                                                'actor', 'language']
        self.model_features['movie'] = ['tag', 'director', 'country',
                                        'actor', 'language']
        self.model_features['sports'] = ['tag', 'director', 'country',
                                         'actor', 'language']
        self.model_features['tv'] = ['tag', 'director', 'country',
                                     'actor', 'language']
        self.model_features['variety'] = ['tag', 'director', 'country',
                                          'actor', 'language']

    def set_feature(self, model, features):
        self.model_features[model] = features

    def _filter_id(self, document):
        '''filte other attribution but id
        -->-1: not exist, other be video id'''
        record = document
        id = '-1'
        if record['tencent'] == '1':
            id = record['pp_tencent']['tencentId']
        elif record['youpeng'] == '1':
            id = record['pp_youpeng']['yp_id']
        return id

    def _filter_pay_status(self, document):
        record = document
        vip = 0
        if record['tencent'] == '1' and record['pp_tencent']['pay_status'].strip() in ['用券', '会员点播', '会员免费']:
            vip = 1
        elif record['youpeng'] == '1':
            vip = 1
        else:
            vip = 0
        return vip

    def _filter_year(self, document):
        year = document.get('year', 'None')
        if isinstance(year, str):
            year = year[:4]
        try:
            if year == 'None' or year == u'未知' or int(year) - 0 < 1500:
                year = int(document.get('issue', 'None').split('-')[0])
        except ValueError, e:
            print 'ValueError {0}'.format(e)
            year = None
        if year is not None and int(year) < 1500:
            year = None
        elif year is not None:
            year = int(year)
        return year

    def _filter_tag(self, document):
        tag = document.get('tag', '').replace(r' ', '')
        return tag

    def _filter_language(self, document):
        language = document.get('language', '').encode('utf-8').replace(r' ', '')
        return language

    def _filter_country(self, document):
        country = document.get('country', '').replace(r' ', '')
        return country

    def _filter_categorys(self, document):
        categorys = list()
        for item in document.get('categorys', []):
            categorys.append(item['name'])
        return categorys

    def _handle_all_attr(self, document):
        data = {}
        none_label = ''
        document = dict(document)
        data['cover_id'] = self._filter_id(document)
        data['model'] = document.get('model', none_label)
        data['alias'] = document.get('alias', none_label)
        # num
        data['duration'] = document.get('duration', -1)
        data['enname'] = document.get('enName', none_label)
        data['language'] = self._filter_language(document)

        data['name'] = document.get('name', none_label).strip()
        # data['issue'] = document.get('issue', '0000-00-00')
        data['director'] = str(document.get('director', '').strip())
        data['actor'] = str(document.get('cast', '')).replace(r' ', '')
        # num
        data['grade_score'] = document.get('grade_score', 0)
        data['tag'] = str(self._filter_tag(document))
        data['country'] = str(self._filter_country(document))
        # TODO data['country_group'] = document.get('country_group', '[]')
        # num
        data['episodes'] = document.get('episodes', -1)
        data['definition'] = str(document.get('definition', -1))
        data['writer'] = str(document.get('writer', none_label).strip())
        data['year'] = self._filter_year(document)
        # 上架
        data['enable'] = document.get('enable', '-1')
        # 碎视频
        data['isClip'] = document.get('isClip', '-1')
        data['categorys'] = self._filter_categorys(document)
        data['pp_tencent'] = document.get('pp_tencent', '{}')
        data['pp_iqiyi'] = document.get('pp_iqiyi', '{}')
        data['pp_youpeng'] = document.get('pp_youpeng', '{}')
        data['tencent'] = document.get('tencent', '-1')
        data['iqiyi'] = document.get('iqiyi', '-1')
        data['youpeng'] = document.get('youpeng', '-1')
        data['focus'] = document.get('focus', '').strip()
        data['vip'] = self._filter_pay_status(document)
        return data

    def process(self):
        '''return dict {model_name: [tag, actor, director]} '''
        data = dict()
        self.logger.info('start process data')
        for model in self.models:
            # if model in ['tv', 'movie']: continue
            self.logger.info('get data in database of model : {0}'.format(model))
            documents = self._get_documents(self.collection, {'model': model}, 10)
            self.logger.info('start handle data of model: {0}'.format(model))
            data[model] = self._process_documents(model, documents)
            self.logger.info('format data of model {0} finished'.format(model))
        return data

    def _get_documents(self, collection, condition, num=10):
        documents = collection.find(condition).limit(num)
        # documents = collection.find(condition)
        return documents

    def _process_documents(self, model, documents):
        tag_stack = list()
        actor_stack = list()
        id_stack = list()
        director_stack = list()
        for document in documents:
            data = self._handle_all_attr(document)
            self.logger.debug('record: {0}'.format(data))
            if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
                continue
            if data['cover_id'] in ['q5ni28gov0wrnr3', '0efab4wwezfsswp', 'e6dvr0t33mtckia', '0k7ue81txhpmozo', '0t301lolceby6hu', '3yi89pocfvj3vkg']:
                continue
            id_stack.append(data['cover_id'])
            tag_stack.append(self.handle_tag_categorys(data))
            actor_stack.append(self.handle_multi_label('actor', data))
            director_stack.append(self.handle_multi_label('director', data))
        v = DictVectorizer(sparse=False)
        tag_stack = v.fit_transform(tag_stack)
        tag_stack = DataFrame(tag_stack, index=id_stack, columns=v.feature_names_)
        self.logger.debug('tag features of model {0}: {1}'.format(model, v.feature_names_))
        actor_stack = v.fit_transform(actor_stack)
        actor_stack = DataFrame(actor_stack, index=id_stack, columns=v.feature_names_)
        self.logger.debug('actor features of model {0}: {1}'.format(model, v.feature_names_))
        director_stack = v.fit_transform(director_stack)
        director_stack = DataFrame(director_stack, index=id_stack, columns=v.feature_names_)
        self.logger.debug('director features of model {0}: {1}'.format(model, v.feature_names_))
        return {'tag':tag_stack, 'actor':actor_stack, 'director':director_stack}

    def handle_tag_categorys(self, record):
        tag = record['tag']
        categorys = record['categorys']
        tag = tag.split(r'/')
        data = dict()
        for index in categorys:
            data[index] = 1
        for index in set(tag) - set(categorys):
            data[index] = self._weight_tune(tag.index(index) + 1)
        return data

    def handle_multi_label(self, key, record):
        record = record[key].split(r'/')
        data = dict()
        for index in record:
            data[index] = self._weight_tune(record.index(index) + 1)
            # data[index] = record.index(index)
        return data

    def _weight_tune(self, index):
        if index <= 0:
            raise ValueError
        return np.exp(1 - np.sqrt(index))


def main(config_file_path, data_file):
    cf = ConfigParser.ConfigParser()
    cf.read(config_file_path)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    handler = Video()
    print 'connect mongo'
    handler.connect_mongodb(address, port, username, password, database, collection)
    print 'connect success'
    print 'start get and process data'
    data = handler.process()
    print 'process end'
    # print('store data to file: {0}'.format(data_file))
    with open(data_file, 'wb') as f:
        pickle.dump(data, f, protocol=True)
    print("stored data to file: {0}".format(data_file))
    '''
    print 'save data to excel'
    for model, values in data.items():
        print values.shape
        print model
        values.to_excel('../data/{0}_data.xlsx'.format(model))

    print 'Finished'
    '''
    return data


if __name__ == '__main__':
    config_file = '../etc/config.ini'
    data_file = '../data/all_video_info.dat'
    data = main(config_file, data_file)
    print data
    print 'Finished'
