# encoding:utf-8
# get data id group by model
# and save to local file '../data/id.dat'
from pymongo import MongoClient
import logging
from datetime import date, timedelta
import ConfigParser
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import pullword
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
        self.features_trans['cartoon'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['doc'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['education'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['entertainment'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['movie'] = {'tag': 4, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['sports'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['tv'] = {'tag': 4, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}
        self.features_trans['variety'] = {'tag': 3, 'writer': 1, 'director': 1, 'actor': 2, 'country': 1, 'language': 1}

    def init_model_features(self):
        '''default features to be handled in process'''
        self.model_features['cartoon'] = ['grade_score', 'year', 'tag', 'writer',
                                          'director', 'country', 'episodes',
                                          'actor', 'language', 'duration', 'vip']
        self.model_features['doc'] = ['grade_score', 'year', 'tag', 'writer',
                                      'director', 'country', 'episodes',
                                      'actor', 'language', 'duration', 'vip']
        self.model_features['education'] = ['grade_score', 'year', 'tag', 'writer',
                                            'director', 'country', 'episodes',
                                            'actor', 'language', 'duration', 'vip']
        self.model_features['entertainment'] = ['grade_score', 'year', 'tag', 'writer',
                                                'director', 'country', 'episodes',
                                                'actor', 'language', 'duration', 'vip']
        self.model_features['movie'] = ['grade_score', 'year', 'tag', 'writer',
                                        'director', 'country', 'episodes',
                                        'actor', 'language', 'duration', 'vip']
        self.model_features['sports'] = ['grade_score', 'year', 'tag', 'writer',
                                         'director', 'country', 'episodes',
                                         'actor', 'language', 'duration', 'vip']
        self.model_features['tv'] = ['grade_score', 'year', 'tag', 'writer',
                                     'director', 'country', 'episodes',
                                     'actor', 'language', 'duration', 'vip']
        self.model_features['variety'] = ['grade_score', 'year', 'tag', 'writer',
                                          'director', 'country', 'episodes',
                                          'actor', 'language', 'duration', 'vip']

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
        data['director'] = str(document.get('director', none_label).strip())
        data['actor'] = str(document.get('cast', none_label)).replace(r' ', '')
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
        data['categorys'] = document.get('categorys', none_label)
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
        data = {}
        self.logger.info('start process data')
        for model in self.models:
            # if model in ['tv', 'movie']: continue
            self.logger.info('get data in database of model : {0}'.format(model))
            documents = self._get_documents(self.collection, {'model': model}, 100)
            self.logger.info('start handle data of model: {0}'.format(model))
            data[model] = self._process_documents(model, documents)
            self.logger.info('format data of model {0} finished'.format(model))
        return data

    def _get_documents(self, collection, condition, num=10):
        documents = collection.find(condition).limit(num)
        # documents = collection.find(condition)
        return documents

    def _process_documents(self, model, documents):
        data_stack = []
        id_stack = []
        for document in documents:
            data = self._handle_all_attr(document)
            self.logger.debug('record: {0}'.format(data))
            if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
                continue
            if data['cover_id'] in ['q5ni28gov0wrnr3', '0efab4wwezfsswp', 'e6dvr0t33mtckia', '0k7ue81txhpmozo', '0t301lolceby6hu', '3yi89pocfvj3vkg']:
                continue
            record = []
            for feature in self.model_features[model]:
                # check '0'
                if data[feature] == -1 or (isinstance(data[feature], (str, unicode)) and
                                           data[feature].strip() in ['未知', 'None', '不详']):
                    data[feature] = np.nan
                # if feature == 'language': print data[feature].strip()
                record.append(data[feature])
            id_stack.append(data['cover_id'])
            data_stack.append(record)

        columns = self.model_features[model]
        self.logger.debug('columns of model {0}: {1}'.format(model, columns))
        data = DataFrame(data_stack, index=id_stack, columns=columns)
        data = self.clean_data(data)
        data = self._expand_all_columns(data, model)
        data.fillna(data.median(), inplace=True)
        return data

    def clean_data(self, data):
        # if feature in features_trans, string null stand for missing value
        # else None will None
        data['country'] = data['country'].str.replace(r' ', '')
        data['country'] = data['country'].str.replace(r'不详', '')
        data['country'] = data['country'].str.replace(r'中国内地', '内地')
        data['country'] = data['country'].str.replace(r'中国大陆', '内地')
        data['country'] = data['country'].str.replace(r'中国香港', '香港')
        data['country'] = data['country'].str.replace(r'内地剧', '内地')
        data['language'] = data['language'].str.replace(r' ', '')
        data = self._value_fix(data, 'language', ['华语', '汉语普通话'], '普通话')
        data = self._value_fix(data, 'language', ['美国'], '英语')
        data = self._value_fix(data, 'language', [''], '')
        tmp = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '38', '不详']
        data = self._value_fix(data, 'language', tmp, '')
        data = self._value_fix(data, 'tag', ['0'], '')
        # data['actor'] = data['actor'].str.replace(r' ', '')
        data = self._value_fix(data, 'director', ['0'], '')
        return data

    def _value_fix(self, data, feature, fault_value, correct_value):
        mask = data.isin({feature: fault_value})
        data = data.where(~mask, other=correct_value)
        return data

    def _expand_all_columns(self, data, model):
        for column in self.features_trans[model]:
            size = self.features_trans[model][column]
            # data = self._expand_single_column(data, column, size)
            data = self._multilabel_single_column(data, column)
            # data = self._raw_string_column(data, column)
        return data

    def _raw_string_column(self, data, column):
        tmp = data[column].str.split(r'/').apply(str).to_frame(column)
        data.drop(labels=column, axis=1, inplace=True)
        data = pd.concat([data, tmp], axis=1)
        return data

    def _multilabel_single_column(self, data, column):
        mlb = MultiLabelBinarizer()
        tmp = data[column].fillna('')
        index = data[column].index
        tmp = tmp.str.split(r'/')
        temp = mlb.fit_transform(tmp)
        data.drop(labels=column, axis=1, inplace=True)
        column_name = [column + '_' + s for s in mlb.classes_.tolist()]
        # if column in ['tag', 'writer']: return data
        formated_columns = DataFrame(data=temp, columns=column_name, index=index)
        formated_columns = self.filter_multilabel(formated_columns)
        data = pd.concat([data, formated_columns], axis=1)
        return data

    def filter_multilabel(self, data):
        min_num_common_feature = 5
        tmp = data.sum(axis=0, skipna=True)
        invalid_feature_column = tmp.index[tmp < min_num_common_feature]
        data.drop(labels=invalid_feature_column, axis=1, inplace=True)
        return data

    def _expand_single_column(self, data, column_name, min_size):
        expand = data[column_name].str.split('/', expand=True, n=min_size).iloc[:, :min_size]
        expand.columns = self._expand_column_names(column_name, len(expand.columns))
        expand[expand == ''] = None
        data.drop(labels=column_name, axis=1, inplace=True)
        data = pd.concat([data, expand], axis=1)
        return data

    def _expand_column_names(self, column_name, length):
        columns = []
        for index in range(length):
            columns.append(column_name + str(index))
        return columns

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
        except_features = ['duration', 'grade_score', 'year', 'episodes', 'vip']
        columns_dummy = self._get_columns(model)
        for feature in except_features:
            if feature in columns_dummy:
                columns_dummy.remove(feature)

        dummies = pd.get_dummies(data=dataframe, columns=columns_dummy, dummy_na=True)
        # dataframe.drop(labels=columns_dummy, axis=1, inplace=True)
        # dataframe = pd.concat([dataframe, dummies], axis=1)
        return dummies

    def text_feature(self, data_series):
        data, feature_name = pullword.extract_text_feature(data_series)
        return pd.DataFrame(data.toarray(), feature_name)


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
    print 'start get and process data'
    data = handler.process()
    print 'process end'

    print 'save data to excel'
    for model, values in data.items():
        print values.shape
        print model
        values.to_excel('../data/{0}_data.xlsx'.format(model))

    print 'Finished'
    return data


if __name__ == '__main__':
    import cPickle as pickle

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
    dataframe = data['tv']
    data = handler.dummy_process(dataframe)
    print data.columns
    with open('../data/tmp.txt', 'w') as f:
        pickle.dump(data.columns, f)
        pickle.dump(data[1:2], f)
'''
