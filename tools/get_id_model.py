# encoding:utf-8
# get data id group by model
# and save to local file '../data/id.dat'
from pymongo import MongoClient
import cPickle as pickle
import logging
from datetime import date, timedelta
import ConfigParser
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Video(object):

    def __init__(self):
        self.model_features = {}
        self.features_trans = {}
        self.logger = self.set_logger('../log')
        self.init_model_features()
        self.init_trans_features()
        self.models = ['movie', 'tv',
                       'sports', 'entertainment', 'variety',
                       'education', 'doc', 'cartoon']

    def set_logger(self, path):
        format = '''[ %(levelname)s %(asctime)s @ %(process)d] (%(filename)s:%(lineno)d) - %(message)s'''
        curDate = date.today() - timedelta(days=0)
        log_name = '{0}/get_mongo_{1}.log'.format(path, curDate)
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

    def init_trans_features(self):
        '''set features that need to be transformed from list to binary like dummy_feature in sklearn'''
        self.features_trans['cartoon'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['doc'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['education'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['entertainment'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['movie'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['sports'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}
        self.features_trans['tv'] = {'tag': 4, 'writer': 1, 'director': 1, 'actor': 4, 'country': 1}
        self.features_trans['variety'] = {'tag': 3, 'writer': 2, 'director': 2, 'actor': 4, 'country': 1}

    def init_model_features(self):
        '''default features to be handled in process'''
        self.model_features['cartoon'] = ['year',
                                          'tag', 'writer', 'director',
                                          'country', 'episodes', 'actor',
                                          'language', 'duration']
        self.model_features['doc'] = ['year', 'tag', 'writer',
                                      'director', 'country', 'episodes',
                                      'actor', 'language', 'duration']
        self.model_features['education'] = ['year', 'tag', 'writer',
                                            'director', 'country', 'episodes',
                                            'actor', 'language', 'duration']
        self.model_features['entertainment'] = ['year', 'tag', 'writer',
                                                'director', 'country', 'episodes',
                                                'actor', 'language', 'duration']
        self.model_features['movie'] = ['year', 'tag', 'writer',
                                        'director', 'country', 'episodes',
                                        'actor', 'language', 'duration']
        self.model_features['sports'] = ['year', 'tag', 'writer',
                                         'director', 'country', 'episodes',
                                         'actor', 'language', 'duration']
        self.model_features['tv'] = ['year', 'tag', 'writer',
                                     'director', 'country', 'episodes',
                                     'actor', 'language', 'duration']
        self.model_features['variety'] = ['year', 'tag', 'writer',
                                          'director', 'country', 'episodes',
                                          'actor', 'language', 'duration']

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

    def _filetr_year(self, document):
        year = document.get('issue', 'None').split('-')[0]
        if year == 'None':
            year = document.get('year', 'None')
        return year

    def _filter_tag(self, document):
        tag = document.get('tag', 'None').split('/')
        return tag

    def _filter_language(self, document):
        language = document.get('language', 'empty').strip()
        if language in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '不详']:
            language = 'empty'
        elif language == '美国':
            language = '英语'
        elif language in ['国语', '华语', '普通话']:
            language = '普通话'
        return language

    def _filter_country(self, document):
        country = document.get('country', 'empty').strip().split('/')
        for index in range(len(country)):
            if country[index].strip() == '内地':
                country[index] = '中国内地'
            else:
                country[index] = country[index].strip()
        return country

    def _handle_all_attr(self, document):
        data = {}
        document = dict(document)
        data['cover_id'] = self._filter_id(document)
        data['model'] = document.get('model', 'empty')
        data['alias'] = document.get('alias', 'empty')
        # num
        data['duration'] = document.get('duration', -1)
        data['enname'] = document.get('enName', 'empty')
        data['language'] = self._filter_language(document)

        data['name'] = document.get('name', 'empty').strip()
        # data['issue'] = document.get('issue', '0000-00-00')
        data['director'] = str(document.get('director', 'empty').strip().split('/'))
        data['actor'] = str(document.get('cast', 'empty').split('/'))
        # num
        data['grade_score'] = document.get('grade_score', -1)
        data['tag'] = str(self._filter_tag(document))
        data['country'] = str(self._filter_country(document))
        # TODO data['country_group'] = document.get('country_group', '[]')
        # num
        data['episodes'] = document.get('episodes', -1)
        data['definition'] = str(document.get('definition', -1))
        data['writer'] = str(document.get('writer', 'empty').strip().split('/'))
        data['year'] = self._filetr_year(document)
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
        data['focus'] = document.get('focus', 'empty').strip()
        data['vip'] = str(self._filter_pay_status(document))
        return data

    def process(self):
        data = {}
        self.logger.info('start process data')
        for model in self.models:
            self.logger.info('get data in database of model : {0}'.format(model))
            documents = self._get_documents(self.collection, {'model': model}, 2)
            self.logger.info('start handle data of model: {0}'.format(model))
            data[model] = self._process_documents(model, documents)
            self.logger.info('format data of model {0} finished'.format(model))
        # print len(data[model])
        return data

    def _get_documents(self, collection, condition, num=10):
        # documents = collection.find(condition).limit(num)
        documents = collection.find(condition)
        return documents

    def _process_documents(self, model, documents):
        data_stack = []
        id_stack = []
        for document in documents:
            data = self._handle_all_attr(document)
            self.logger.debug('record: {0}'.format(data))
            if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
                continue

            record = {}
            for feature in self.model_features[model]:
                # '' : string NULL  or value == 'unkown' or by default 'empty'
                if data[feature] == -1 or (isinstance(data[feature], str) is True and
                                           data[feature].strip() in ['', '未知', 'empty', 'None', '-1', '不详']):
                    data[feature] = np.nan
                elif data[feature] == -1:
                    data[feature] = np.nan
                if feature in self.features_trans[model]:
                    feats = self._feature_transform(model, feature, data[feature])
                    for k, v in feats.items():
                        record[k] = v
                else:
                    record[feature] = data[feature]
            self.logger.debug('formated data: {0}'.format(record))
            id_stack.append(data['cover_id'])
            data_stack.append(record.values())
        # columns = self._get_columns(model)
        columns = record.keys()
        self.logger.debug('columns of model {0}: {1}'.format(model, columns))
        return DataFrame(data_stack, index=id_stack, columns=columns)

    def _get_columns(self, model):
        '''transform feature in type-list to binary'''
        columns = []
        for feature in self.model_features[model]:
            if feature not in self.features_trans[model]:
                columns.append(feature)
            else:
                for index in range(self.features_trans[model][feature]):
                    columns.append(feature + str(index))
        return columns

    def _feature_transform(self, model, feature, value):
        '''transform value of feature in list-type to binary'''
        transformed = {}
        value = eval(value)
        for index in range(self.features_trans[model][feature]):
            if index < len(value) and value[index].strip() not in ['', 'empty', '未知', 'None', '不详']:
                transformed[feature + str(index)] = value[index].strip()
            else:
                transformed[feature + str(index)] = np.nan
        return transformed

    def dummy_process(self, dataframe, model='tv'):
        except_features = ['duration', 'grade_score', 'year', 'episodes']
        columns_dummy = self._get_columns(model)
        for feature in except_features:
            if feature in columns_dummy:
                columns_dummy.remove(feature)

        dummies = pd.get_dummies(data=dataframe, columns=columns_dummy)
        dataframe.drop(labels=columns_dummy, axis=1, inplace=True)
        dataframe = pd.concat([dataframe, dummies], axis=1)
        return dataframe


def main(config_file_path):
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
    data = handler.process()
    print 'Finished'
    return data


def analysis_data(data):
    matplotlib.rc('xtick', labelsize=16)
    for index in range(len(data['language'])):
        if data['language'][index] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '不详']:
            data['language'][index] = np.nan
    '''
    for index in range(len(data['writer0'])):
        if data['writer0'][index] == '':
            data['writer0'][index] = np.nan
    for index in range(len(data['country0'])):
        data['country0'][index] = data['country0'][index].strip()
    '''
    dataframe = data.fillna('不详')
    dataframe = data
    for index in dataframe.columns:
        plt.figure(index)
        plt.title('{0}统计分布'.format(index))
        plt.xlabel(index, fontsize=16)
        plt.ylabel('num')
        c = dataframe[index].value_counts()
        c.sort_values(ascending=False)[:50].plot('bar')
    plt.show()


if __name__ == '__main__':
    '''
    config_file = '../etc/config.ini'
    data = main(config_file)
    with open('../data/id.dat', 'wb') as f:
        pickle.dump(data, f, protocol=True)
    '''

    data = {}
    with open('../data/id.dat', 'rb') as f:
        data = pickle.load(f)
    if len(data) == 0:
        print 'data source error'
        exit()
    handler = Video()
    dataframe = data['movie']
    analysis_data(dataframe)
'''
    data = handler.dummy_process(dataframe)
    print data.columns
    with open('../data/tmp.txt', 'w') as f:
        pickle.dump(data.columns, f)
        pickle.dump(data[1:2], f)
'''