# encoding:utf-8
"""
============================================================
Fetch and clean media data before calculation of similarity
============================================================

fetch data from mongodb.
use charactor  'language', 'country', 'writer', 'director', 'tag', 'actor', 'year', 'score'

Script output:
    data/{model_name}.dat
        local clean data for calculation to reading directly from locals
        make local cache since frequence of similarity calculation
        be greater than of the scripts that fetch and clean media data

Created by:
    yonggang Huang
In:
    05-25-2017
"""

from pymongo import MongoClient
import ConfigParser
import cPickle as pickle
from pandas import DataFrame, Series
from datetime import datetime, date, timedelta
import re
import traceback


models = ['movie', 'tv',
          'sports', 'entertainment', 'variety',
          'education', 'doc', 'cartoon']

tran_alias_label_dict = {'language': {u'普通话': u'国语', u'普通话': '1', u'英语': '2', u'法语': '3', u'日语': '7'},
                         'country': {u'中国大陆': u'内地', u'中国大陆': u'中国内地', u'中国台湾': u'台湾', u'中国香港': u'香港'}}

outliers_list = {'language': ['0', '1', '2', '3' '4', '5', '6', '7', '8', '9', '38'],
                 'country': [u'北京金英马影视文化有限责任公司', u'中華']}


class Formatter:

    def __init__(self):
        self.pat = re.compile(r'\s*[/\\|]\s*')

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
            year = int(year)
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
        for label in data:
            if isinstance(data[label], list):
                data[label] = '|'.join(data[label])
        return data

    def process(self, documents):
        data_df = list()
        for doc in documents:
            try:
                data = self._handle_all_attr(doc)
            except Exception:
                print('clean label get error')
                raise Exception('{0}'.format(traceback.print_exc()))
            data_df.append(data)
        return DataFrame(data=data_df)


def connect_mongodb(
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
    return collection


def multi_label_format(data_df, sep=r'|', threshold=3):

    tmp = dict()
    for labels in data_df:
        for lab in labels.split(sep):
            tmp[lab] = tmp.get(lab, 0) + 1
    effective_labels = [k for k in tmp if tmp[k] >= threshold]

    data_df = data_df.apply(lambda x: Series(data={k: 1 for k in x.split(sep)}))
    data_df.drop(set(tmp.keys()).difference(set(effective_labels)), axis=1, inplace=True)
    data_df.fillna(0, inplace=True)
    return data_df


def vector_dissimilarity(x, dtype='categorical'):
    from numpy import max, min, zeros
    size = x.shape[0]
    distance = zeros((size, size))
    for i in range(size):
        if dtype == 'categorical':
            distance[i] = map(lambda m: 0 if m == x[i] else 1, x)
        elif dtype == 'numerical':
            distance[i] = map(lambda m: abs(m - i), x)
        else:
            raise TypeError('Not supported dtype')
    if dtype == 'categorical':
        return distance

    # scale [0 1] by maxmin scale for numerical data
    distance = ((distance - min(distance, axis=0)) /
                (max(distance, axis=0) - min(distance, axis=0)))
    return distance


def main(config_file, num=1000):
    import pandas as pd
    from kmedoids import cluster

    cf = ConfigParser.ConfigParser()
    cf.read(config_file)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    collection = connect_mongodb(address, port, username, password, database, collection)
    print 'connect monog success'
    handler = Formatter()
    for model in models:
        if model not in ['tv', 'movie']:  continue
        documents = collection.find({'model': model}).limit(num)
        # documents = collection.find(condition)
        all_model_data_df = handler.process(documents)
        data_df = all_model_data_df

        year_df = pd.qcut(data_df['year'], 10, labels=False)  # discretization
        score_df = pd.qcut(data_df['score'], 5, labels=False)  # discretization
        # split high categorical feature to onehot coding
        train_tmp = []
        for label in ['language', 'country', 'writer', 'director']:
            tmp = multi_label_format(data_df[label])
            train_tmp.append(tmp)

        tag_df = multi_label_format(data_df['tag'])
        actor_df = multi_label_format(data_df['actor'])
        actor_list = actor_df.columns

        actor_df.index = data_df['cover_id']
        year_df.index = data_df['cover_id']
        score_df.index = data_df['cover_id']
        tag_df.index = data_df['cover_id']

        num_program_per_actor = actor_df[actor_list].sum(axis=0)
        actor_attr_matrix = list()
        for actor in actor_list:
            vec = list()
            ids_se = actor_df[actor_df[actor] > 0.9].index
            vec.append(num_program_per_actor[actor])
            vec.append(tag_df.loc[ids_se].sum(axis=1).argmax())
            year_mode = year_df.loc[ids_se].mode()
            year_mode = year_mode[0] if len(year_mode) != 0 else year_df.loc[ids_se][0]
            vec.append(year_mode)
            score_mode = score_df.loc[ids_se].mode()
            score_mode = score_mode[0] if len(score_mode) != 0 else score_df.loc[ids_se][0]
            vec.append(score_mode)
            actor_attr_matrix.append(vec)

        columns_name = ['num_program', 'tag', 'year', 'score']
        actor_attrs = DataFrame(data=actor_attr_matrix, index=actor_list, columns=columns_name)
        distances = vector_dissimilarity(actor_attrs['tag'], dtype='categorical')
        distances += vector_dissimilarity(actor_attrs['year'], dtype='numerical')
        distances += vector_dissimilarity(actor_attrs['score'], dtype='numerical')
        distances += vector_dissimilarity(actor_attrs['num_program'], dtype='numerical')
        # dimension reducation for actors
        actors_mapped, _ = cluster(distances, k=10)
        trans_map = {k: v for k, v in zip(actor_list, actors_mapped)}
        tmp = all_model_data_df['actor'].apply(lambda x: '|'.join(map(lambda y: '%s' % trans_map.get(y, ''), x.split('|'))))
        actor_df = multi_label_format(tmp)
        train_tmp.extend([tag_df, actor_df, pd.get_dummies(year_df, prefix='year'), pd.get_dummies(score_df, prefix='score')])
        size_per_feat_dict = [(k, v.shape[-1]) for k, v in zip(['language', 'country', 'writer', 'director',
                                                                'tag', 'actor', 'year', 'score'], train_tmp)]
        train_tmp.append(data_df['cover_id'])
        train_df = pd.concat(train_tmp, axis=1)
        train_df.drop_duplicates(subset=['cover_id'], inplace=True)
        yield model, {'values': train_df, 'property': size_per_feat_dict}


if __name__ == '__main__':
    config_file = '../etc/config.ini'
    data_file_path = '../data'
    for model, block in main(config_file):
        with open(data_file_path + r'/' + model + r'.dat', 'wb') as f:
            pickle.dump(block, f, protocol=True)
    print("stored data to path: {0}".format(data_file_path))
