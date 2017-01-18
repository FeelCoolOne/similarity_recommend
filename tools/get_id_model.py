# encoding:utf-8
# get data id group by model
# and save to local file '../data/id.dat'
from pymongo import MongoClient
import cPickle as pickle
from pandas import DataFrame
import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


'''
def init_con(host='127.0.0.1', port=50000):

    con = MongoClient(host, port)
    EPGInfo = con.EPGInfo
    EPGInfo.authenticate(username, password, mechanism='SCRAM-SHA-1')
    collection = EPGInfo['chiq_video_converge']
    # document = collection.find_one({'model': 'movie'})
    return collection
'''

def filter_id(document, condition=''):
    ''' filter id from document
        tencent id is priority from yp_id
    '''
    'filte other attribution but id'
    record = dict(document)
    id = ''
    if record['tencent'] == '1':
        id = record['pp_tencent']['tencentId']
    elif record['youpeng'] == '1':
        id = record['pp_youpeng']['yp_id']
    else:
        print 'error'
        print record
    return id


class Video(object):

    def __init__(self):
        self.model_features = {}
        self.features_trans = {}
        self.init_model_features()
        self.init_trans_features()
        self.models = ['movie', 'tv',
                       'sports', 'entertainment', 'variety',
                       'education', 'doc', 'cartoon']

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
        year = document.get('year', 'None')
        if year == 'None':
            year = document.get('issue', 'None').split('-')[0]
        return year

    def _fileter_tag(self, document):
        tag = document.get('tag', 'None').split('/')
        return tag

    def _handle_all_attr(self, document):
        data = {}
        document = dict(document)
        data['cover_id'] = self._filter_id(document)
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
        data['tag'] = str(self._fileter_tag(document))
        data['country'] = str(document.get('country', 'empty').split('/'))
        # TODO data['country_group'] = document.get('country_group', '[]')
        data['episodes'] = str(document.get('episodes', -1))
        data['definitions'] = str(document.get('definitions', -1))
        data['writer'] = str(document.get('writer', 'empty').split('/'))
        data['year'] = str(self._filetr_year(document))
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
        data['vip'] = str(self._filter_pay_status(document))
        return data

    def process(self):
        data = {}
        # print 'get and process data'
        for model in self.models:
            documents = self._get_documents(self.collection, {'model': model}, 2)
            data[model] = self._process_documents(model, documents)
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
            if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
                continue

            record = {}
            for feature in self.model_features[model]:
                # '' : string NULL  or value == 'unkown' or by default 'empty'
                if data[feature].strip() in ['', '未知', 'empty', 'None']:
                    data[feature] = np.nan
                if feature in self.features_trans[model]:
                    feats = self._feature_transform(model, feature, data[feature])
                    for k, v in feats.items():
                        record[k] = v
                else:
                    record[feature] = data[feature]
            id_stack.append(data['cover_id'])
            data_stack.append(record.values())
        # columns = self._get_columns(model)
        columns = record.keys()
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
            if index < len(value) and value[index] not in ['empty', '未知', 'None']:
                transformed[feature + str(index)] = value[index]
            else:
                transformed[feature + str(index)] = np.nan
        return transformed


def main():
    host = '127.0.0.1'
    port = 50000
    username = 'EPGInfo'
    password = 'EPGInfo@20150603'
    database = 'EPGInfo'
    collection = 'chiq_video_converge'
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    handler = Video()
    print 'connect mongo'
    handler.connect_mongodb(host, port, username, password, database, collection)
    print 'connect success'
    data = handler.process()
    return data


if __name__ == '__main__':
    data = main()
    with open('../data/id.dat', 'w') as f:
        pickle.dump(data, f, protocol=True)
