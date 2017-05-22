# encoding:utf-8
"""
============================================================
Fetch and clean media data before calculation of similarity
============================================================

fetch data from mongodb.
use charactor  'tag', 'director', 'country', 'actor', 'language', 'year', 'score'

Script output:
    data/{model_name}.dat
        local clean data for calculation to reading directly from locals
        make local cache since frequence of similarity calculation
        be greater than of the scripts that fetch and clean media data

Created by:
    yonggang Huang
In:
    03-31-2017
"""

from pymongo import MongoClient
import logging
from datetime import datetime, date, timedelta
import ConfigParser
import cPickle as pickle
from pandas import DataFrame, Series
# import pandas as pd
import numpy as np
import re
import traceback

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Video(object):

    def __init__(self):
        self.model_features = dict()
        self.features_trans = dict()
        self.logger = self.set_logger('log')
        self.init_model_features()
        self.pat = re.compile(r'\s*[/\\|]\s*')
        self.models = ['movie', 'tv',
                       'sports', 'entertainment', 'variety',
                       'education', 'doc', 'cartoon']
        self.outliers_list = {'language': ['0', '1', '2', '3' '4', '5', '6', '7', '8', '9', '38'],
                              'country': [u'北京金英马影视文化有限责任公司', u'中華']}
        self.tran_alias_label_dict = {'language': {u'普通话': u'国语', u'普通话': '1', u'英语': '2', u'法语': '3', u'日语': '7'},
                                      'country': {u'中国大陆': u'内地', u'中国大陆': u'中国内地', u'中国台湾': u'台湾', u'中国香港': u'香港'}}

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
                                          'actor', 'language', 'year', 'score']
        self.model_features['doc'] = ['tag', 'director', 'country',
                                      'actor', 'language', 'year', 'score']
        self.model_features['education'] = ['tag', 'director', 'country',
                                            'actor', 'language', 'year', 'score']
        self.model_features['entertainment'] = ['tag', 'director', 'country',
                                                'actor', 'language', 'year', 'score']
        self.model_features['movie'] = ['tag', 'director', 'country',
                                        'actor', 'language', 'year', 'score']
        self.model_features['sports'] = ['tag', 'director', 'country',
                                         'actor', 'language', 'year', 'score']
        self.model_features['tv'] = ['tag', 'director', 'country',
                                     'actor', 'language', 'year', 'score']
        self.model_features['variety'] = ['tag', 'director', 'country',
                                          'actor', 'language', 'year', 'score']

    def set_feature(self, model, features):
        self.model_features[model] = features

    def _filter_id(self, document):
        record = document
        id = None
        if record['tencent'] == '1' and 'pp_tencent' in record:
            id = record['pp_tencent']['tencentId']
        elif record['youpeng'] == '1' and 'pp_youpeng' in record:
            id = record['pp_youpeng']['yp_id']
        '''
        elif record['iqiyi'] == '1':
            id = '666666'
        '''
        return id

    def _filter_pay_status(self, document):
        record = document
        vip = 0
        if record['tencent'] == '1' and record['pp_tencent']['pay_status'].strip() in [u'用券', u'会员点播', u'会员免费']:
            vip = 1
        elif record['youpeng'] == '1':
            vip = 1
        else:
            vip = 0
        return vip

    def _filter_year(self, document):
        try:
            if 'd_issue' in document and document.get('d_issue').strip() != '':
                d_issue = document.get('d_issue').strip()
                year = datetime.strptime(d_issue, "%Y-%m-%d").year
            elif 'issue' in document and document.get('issue').strip() != '':
                issue = document.get('issue').strip()
                year = datetime.strptime(issue, "%Y-%m-%d").year
            elif 'year' in document:
                year = str(document.get('year'))
                year = datetime.strptime(year, "%Y").year
            if year > date.today().year or year < 1900:
                raise Exception("year error : {0}".format(year))
        except Exception:
            year = None
        return year

    def _filter_tag(self, document):
        tag = document.get('d_type', '').strip()
        return self.pat.split(tag)

    def _filter_language(self, document):
        language = document.get('language', '').strip()
        return self.pat.split(language)

    def _filter_country(self, document):
        country = document.get('country', '').strip()
        return self.pat.split(country)

    def _filter_categorys(self, document):
        categorys = list()
        for item in document.get('categorys', []):
            categorys.append(item['name'])
        return categorys

    def _filter_score(self, document):
        if 'd_grade_score' in document and document.get('d_grade_score').strip() != '':
            score = float(document.get('d_grade_score'))
        elif 'grade_score' in document:
            score = float(document.get('grade_score'))
        else:
            score = None
        return score

    def _filter_duration(self, document):
        duration = 0
        if document['tencent'] == 1 and 'duration' in document['pp_tencent']:
            duration = document['pp_tencent']['duration']
        elif document['youpeng'] == 1 and 'duration' in document['pp_youpeng']:
            duration = document['pp_youpeng']['duration']
        elif document['iqiyi'] == 1 and 'duration' in document['pp_iqiyi']:
            duration = document['pp_iqiyi']['duration']
        else:
            duration = document['duration'] if 'duration' in document else None
        return duration

    def _handle_all_attr(self, document):
        data = dict()
        document = dict(document)
        data['cover_id'] = self._filter_id(document)
        # data['model'] = document.get('model', none_label)
        # data['alias'] = document.get('alias', none_label)
        # data['episodes'] = document.get('episodes', -1)
        # data['enname'] = document.get('enName', none_label)
        # data['name'] = document.get('name', none_label).strip()
        data['duration'] = self._filter_duration(document)
        data['director'] = self.pat.split(document.get('director').strip())  # list
        data['actor'] = self.pat.split(document.get('cast', '').strip())[:7]  # list
        data['writer'] = self.pat.split(document.get('writer', '').strip())  # list
        data['score'] = self._filter_score(document)  # float
        data['tag'] = self._filter_tag(document)  # list
        data['country'] = self._filter_country(document)  # list
        data['language'] = self._filter_language(document)  # list
        data['definition'] = document.get('definition', None)  # int
        data['year'] = self._filter_year(document)  # int
        data['vip'] = self._filter_pay_status(document)  # bool
        # 上架
        data['enable'] = document.get('enable', None)  # str
        # data['isClip'] = document.get('isClip', '-1')
        # data['categorys'] = self._filter_categorys(document)
        # data['pp_tencent'] = document.get('pp_tencent', '{}')
        # data['pp_iqiyi'] = document.get('pp_iqiyi', '{}')
        # data['pp_youpeng'] = document.get('pp_youpeng', '{}')
        # data['tencent'] = document.get('tencent', '-1')
        # data['iqiyi'] = document.get('iqiyi', '-1')
        # data['youpeng'] = document.get('youpeng', '-1')
        # data['focus'] = document.get('focus', '').strip()
        return data

    def process(self):
        '''return dict {model_name: [tag, actor, director]} '''
        data = dict()
        self.logger.info('start process data')
        for model in self.models:
            if model not in ['tv', 'movie']: continue
            self.logger.info('get data in database of model : {0}'.format(model))
            documents = self._get_documents(self.collection, {'model': model}, 1000)
            self.logger.info('start handling data of model: {0}'.format(model))
            data[model] = self._process_documents(model, documents)
            self.logger.info('finish formating data of model {0} '.format(model))
        return data

    def _get_documents(self, collection, condition, num=10):
        # documents = collection.find(condition).limit(num)
        documents = collection.find(condition)
        return documents

    def _process_documents(self, model, documents):
        tag_stack = list()
        actor_stack = list()
        id_stack = list()
        year_stack = list()
        director_stack = list()
        language_stack = list()
        country_stack = list()
        score_stack = list()
        for document in documents:
            try:
                data = self._handle_all_attr(document)
            except Exception:
                self.logger.error('handle all raw attr catch error')
                self.logger.error('{0}'.format(traceback.print_exc()))
                continue
                # raise Exception('handle attr catch error')
            self.logger.debug('record: {0}'.format(data))
            if data['enable'] in ['0', None] or data['cover_id'] is None:
                continue
            if data['cover_id'] in ['0k7ue81txhpmozo', '0t301lolceby6hu', '3yi89pocfvj3vkg']:
                continue
            id_stack.append(data['cover_id'])
            year_stack.append(data['year'])
            score_stack.append(data['score'])
            tag_stack.append(self.handle_multi_label(data['tag']))
            actor_stack.append(self.handle_multi_label(data['actor']))
            director_stack.append(self.handle_multi_label(data['director']))
            language_stack.append(self.handle_multi_label(data['language']))
            country_stack.append(self.handle_multi_label(data['country']))
        tag_stack = self._feature_format(tag_stack, id_stack)
        actor_stack = self._feature_format(actor_stack, id_stack)
        director_stack = self._feature_format(director_stack, id_stack)
        language_stack = self._feature_format(language_stack, id_stack)
        country_stack = self._feature_format(country_stack, id_stack)
        id_stack = Series(data=id_stack, index=id_stack)  # for remove duplicated ids
        year_stack = Series(data=year_stack, index=id_stack)
        score_stack = Series(data=score_stack, index=id_stack)
        self.logger.debug('tag features of model {0}: {1}'.format(model, tag_stack.columns))
        self.logger.debug('tag size of model {0}: {1}'.format(model, len(tag_stack.index)))
        self.logger.debug('actor features of model {0}: {1}'.format(model, actor_stack.columns))
        self.logger.debug('actor size of model {0}: {1}'.format(model, len(actor_stack.index)))
        self.logger.debug('director features of model {0}: {1}'.format(model, director_stack.columns))
        self.logger.debug('director size of model {0}: {1}'.format(model, len(director_stack.index)))
        self.logger.debug('language features of model {0}: {1}'.format(model, language_stack.columns))
        self.logger.debug('language size of model {0}: {1}'.format(model, len(language_stack.index)))
        self.logger.debug('country features of model {0}: {1}'.format(model, country_stack.columns))
        self.logger.debug('country size of model {0}: {1}'.format(model, len(country_stack.index)))
        try:
            self._column_alias_clean(language_stack, self.tran_alias_label_dict.get('language', dict()), self.outliers_list.get('language', list()))
            self._column_alias_clean(country_stack, self.tran_alias_label_dict.get('country', dict()), self.outliers_list.get('country', list()))
            # drop duplicated id
            duplicated_id_list = id_stack.index[id_stack.duplicated()]
            for stack in [tag_stack, actor_stack, director_stack, language_stack, country_stack, year_stack, score_stack]:
                stack.drop(labels=duplicated_id_list, axis=0, inplace=True)
        except Exception:
            self.logger.error('clean label get error')
            self.logger.error('{0}'.format(traceback.print_exc()))
        invalid_columns = [u'未知', u'不详']
        for data in [tag_stack, actor_stack, director_stack, language_stack, country_stack]:
            drop_label = set(invalid_columns).intersection(set(data.columns))
            data.drop(drop_label, inplace=True, axis=1)
            # print data.columns
        return {'tag': tag_stack, 'actor': actor_stack,
                'director': director_stack, 'language': language_stack,
                'country': country_stack, 'year': year_stack, 'score': score_stack}

    def _column_alias_clean(self, frame, tran_alias_dict, outliers_list):
        for stand_label, alias_label in tran_alias_dict.iteritems():
            self._frame_column_merge(frame, alias_label, stand_label)
        drop_label = set(tran_alias_dict.values() + outliers_list).intersection(frame.columns)
        frame.drop(drop_label, inplace=True, axis=1)

    def _frame_column_merge(self, frame, alias_label, standard_label):
        if standard_label not in frame.columns:
            raise Exception('{0} not in frame : {1}'.format(standard_label, frame.columns))
        if alias_label in frame.columns and standard_label in frame.columns:
            f = lambda row: row[alias_label] if row[standard_label] == 0 else row[standard_label]
            frame[standard_label] = frame.apply(f, axis=1)

    def _feature_format(self, data_stack, id_stack):
        '''
        data_statck: list
            [dict(), dict(), dict()]: dict => label:weight
        id_stack: list
            [id, id, id]: program id
        '''
        tmp = list()
        for record in data_stack:
            tmp.extend(record.keys())
        label_list = set(tmp)
        data_matrix = DataFrame(data=np.zeros((len(data_stack), len(label_list))), columns=list(label_list), index=id_stack)
        for index in range(len(data_stack)):
            data_matrix.ix[id_stack[index], data_stack[index].keys()] = data_stack[index].values()
        return data_matrix

    def handle_multi_label(self, label_list):
        data = dict()
        if label_list in [None, ['']]:
            return data
        for id, index in enumerate(label_list):
            data[index] = self._weight_tune(id + 1)
            # data[index] = record.index(index)
        return data

    def _weight_tune(self, index):
        return np.exp(1 - np.sqrt(index))


def main(config_file, data_file_path):
    cf = ConfigParser.ConfigParser()
    cf.read(config_file)
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
    print 'connecting mongo'
    handler.connect_mongodb(address, port, username, password, database, collection)
    print 'success to mongo'
    print 'start getting and processing data'
    data = handler.process()
    print 'finish processing data'
    # print('store data to file: {0}'.format(data_file_path))
    for model in models:
        for feat in data[model]:
            print('feature {0} of model {1} has {2} records '.format(feat, model, len(data[model][feat].index)))
        with open(data_file_path + r'/' + model + r'.dat', 'wb') as f:
            pickle.dump(data[model], f, protocol=True)
    print("stored data to path: {0}".format(data_file_path))
    '''
    print 'save data to excel'
    for (model, values) in data.iteritems():
        values.to_excel('../data/{0}_data.xlsx'.format(model))
    '''
    return data


if __name__ == '__main__':
    config_file = 'etc/config.ini'
    data_file_path = 'data'
    data = main(config_file, data_file_path)
    # print data['tv']
    print 'Finished'
