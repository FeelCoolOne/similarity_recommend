# encoding:utf-8
"""
============================================================
Fetch and clean media data before calculation of similarity
============================================================

fetch data from mongodb.
use charactor  'tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'

Script output:
    data/{model_name}.dat
        local clean data for calculation to reading directly from locals
        make local cache since frequence of similarity calculation
        be greater than of the scripts that fetch and clean media data

Created by:
    yonggang Huang

on:
    03-31-2017

modified:
    05-29-2018
"""

import logging
from datetime import date
import cPickle as pickle
from pandas import DataFrame, Series
import numpy as np
import re
import traceback
import argparse

from tools import get_collection
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

DEBUG = False

SRCS = ["pp_tencent", "pp_iqiyi", "pp_mgtv", "pp_youku", "pp_sohu", ]

MODEL_FEAT = {
    "movie": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "tv": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "variety": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "cartoon": ['tag', 'country', 'year', ],
    "doc": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "entertainment": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "sports": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
    "education": ['tag', 'director', 'country', 'cast', 'language', 'year', 'grade_score'],
}

ATTRS_MULTI_VALUE = ["tag", "cast", "director", "language", "country", ]
ATTRS_SINGLE_VALUE = ["year", "grade_score", ]

INVALID_VALUES = {'language': ['0', '1', '2', '3' '4', '5', '6', '7', '8', '9', '38', u'未知', u'不详', ],
                  'country': [u'北京金英马影视文化有限责任公司', u'中華', u'未知', u'不详', ],
                  "director": [u'未知', u'不详'],
                  "writer": [u'未知', u'不详'], }

ALIAS_DICT = {'language': {u'普通话': u'国语', u'普通话': '1', u'英语': '2', u'法语': '3', u'日语': '7'},
              'country': {u'中国大陆': u'内地', u'中国大陆': u'中国内地', u'中国台湾': u'台湾', u'中国香港': u'香港'}}


class Video(object):

    def __init__(self):
        self.logger = self.set_logger('log')
        self.collection = None
        self.pat = re.compile(r'\s*[/\\|]\s*')
        self.models = ['movie', 'tv',
                       'sports', 'entertainment', 'variety',
                       'education', 'doc', 'cartoon']

    def set_logger(self, path):
        format_ = ("[ %(levelname)s %(asctime)s @ %(process)d]"
                   " (%(filename)s:%(lineno)d) - %(message)s")
        curDate = date.today().strftime("%Y-%m-%d")
        log_name = '{0}/get_raw_data_{1}.log'.format(path, curDate)
        formatter = logging.Formatter(format_)
        logger = logging.getLogger("vav_result")
        handler = logging.FileHandler(log_name, 'a')
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def process(self):
        '''return dict {model_name: [tag, cast, director]} '''
        data = dict()
        self.logger.info('Start formatting')
        condition = dict()
        for model in self.models:
            if model not in ['tv', 'movie', "cartoon"]:
                continue
            self.logger.info('Fetch {} data from mongo'.format(model))
            condition["$or"] = list()
            for src in SRCS:
                condition["$or"].append({"{}.model".format(src): model})
            if DEBUG:
                docs = self.collection.find(condition).limit(200)
            else:
                docs = self.collection.find(condition)
            self.logger.info('Start format data of model: {}'.format(model))
            data[model] = self._handle(model, docs)
            self.logger.info('Finish formating data of {} '.format(model))
        return data

    def _handle(self, model, docs):

        blocks = {attr: list() for attr in MODEL_FEAT[model]}
        id_set, id_list = set(), list()

        for doc in docs:
            try:
                id_, item = self._handle_item(doc)
            except Exception:
                self.logger.error('Handle raw attr catch error')
                self.logger.error("item info: \n {}".format(doc))
                self.logger.error('{}'.format(traceback.print_exc()))
                print(doc)
                continue
                # raise Exception('handle attr catch error')
            self.logger.debug('record: {}'.format(item))
            if id_ is None or id_ == "" or id_ in id_set:
                continue
            id_set.add(id_)  # omit duplicated item
            id_list.append(id_)
            for attr in MODEL_FEAT[model]:
                blocks[attr].append(item[attr])
        if len(blocks) == 0:
            raise Warning("{} has no valid item".format(model))
            return dict()

        result = dict()
        result["id_"] = id_list
        for attr in MODEL_FEAT[model]:
            result[attr] = self._transform(blocks[attr], id_list, attr)

        for attr, feats in result.items():
            if attr in ATTRS_MULTI_VALUE:
                self.logger.debug('{}: {}'.format(attr, feats.columns))
        return self._clean_feature(result)

    def _handle_item(self, doc):

        def get_first_valid(key):
            for src in SRCS:
                record = doc.get(src, {}).get(key, 0)
                if record:
                    break
            return record

        def get_highest_freq(key):
            records = list()
            for src in SRCS:
                rd = doc.get(src, {}).get(key, None)
                if rd:
                    records.append(rd)
            return max(records, key=records.count)

        item = dict()
        doc = dict(doc)
        id_ = get_first_valid("videoId")

        for attr in ["director", "cast", "writer",
                     "country", "language", "tag", "grade_score", ]:
            t = get_first_valid(attr)
            if not isinstance(t, str):
                t = str(t)
            item[attr] = self.pat.sub("|", t).strip()

        for attr in ["grade_score", ]:
            item[attr] = get_first_valid(attr)

        for attr in ["year", ]:
            item[attr] = get_highest_freq(attr)
        return id_, item

    def _transform(self, stack, ids, attr):
        '''
        Transform value from sample space to feature space.

        Parameter:
            data_statck: list
                [dict(), dict(), dict()]: dict => label:weight
            ids: list
                [id, id, id]: program id
            attr: str
        Return:
            pandas.DataFrame with ids as index if attr belong to the multis
            pandas.Series with ids as index if attr belong to the singles
        '''
        if attr in ATTRS_MULTI_VALUE:
            upper_bound = 7
            label_set = set()
            for record in stack:
                rs = record.split("|")[:upper_bound]
                label_set.update(set(rs))
            label_list = list(label_set)
            shape = (len(stack), len(label_list))
            df = DataFrame(data=np.zeros(shape), columns=label_list, index=ids)
            for id_, v in zip(ids, stack):
                rs = v.split("|")[:upper_bound]
                t = self._handle_multi_label(rs)
                df.ix[id_, t.keys()] = t.values()
            return df
        elif attr in ATTRS_SINGLE_VALUE:
            series = Series(data=stack, index=ids)
            return series
        else:
            raise ValueError("attr is invalid")

    def _handle_multi_label(self, labels):

        def weight_tune(index):
            return np.exp(1 - np.sqrt(index))

        data = dict()
        if labels in [None, ['']]:
            return data
        for index, label in enumerate(labels):
            data[label] = weight_tune(index + 1)
        return data

    def _clean_feature(self, result):
        # transform alias to std value
        for attr, data in result.items():
            if attr not in ALIAS_DICT:
                continue
            try:
                result[attr] = self._column_alias_clean(result[attr], attr)
            except Exception:
                self.logger.error('clean label get error')
                self.logger.error('{0}'.format(traceback.print_exc()))
        # clean invalid value in multi
        for attr, data in result.items():
            if attr not in INVALID_VALUES:
                continue
            drop_label = set(INVALID_VALUES[attr]).intersection(set(data.columns))
            data.drop(drop_label, inplace=True, axis=1)
        return result

    def _column_alias_clean(self, frame, attr):

        for std, alias in ALIAS_DICT.get(attr, dict()).items():
            if alias not in frame.columns:
                continue
            if std not in frame.columns:
                frame.rename(index=str, columns={alias: std}, inplace=True)
                continue

            def f(row):
                return row[alias] if row[std] == 0 else row[std]
            frame[std] = frame.apply(f, axis=1)
        drop_label = set(ALIAS_DICT.get(attr, dict()).values()).intersection(frame.columns)
        frame.drop(drop_label, inplace=True, axis=1)
        return frame


def preprocess(config_file, data_file_path):

    models = ['movie', 'tv', 'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    handler = Video()
    print('Connect mongo')
    handler.collection = get_collection(config_file)
    print('Connected to mongo')
    print('Start fetching and formatting')
    data = handler.process()
    print('Finished formatting')

    for model in models:
        if model not in ["tv", "movie", "cartoon"]:
            continue
        for feat in data.get(model, {}):
            if feat == "id_":
                continue
            print('feature {0} of {1} has {2} records'
                  .format(feat, model, data[model][feat].shape[0]))
        for feat in data.get(model, {}):
            np.save(data_file_path + os.sep + model + "_" + feat, data[model][feat])
    print("Stored data to path: {0}".format(data_file_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="get source data from mongoDB to local")
    parser.add_argument("-config", type=str,
                        default="etc/config.ini", help="configure file")
    parser.add_argument("-cache", type=str, default="data", help="cache directory")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        raise Exception("Invalid configure file")
    if not os.path.isdir(args.cache):
        raise Exception("Invalid cache directory")
    if args.debug:
        DEBUG = True
    preprocess(args.config, args.cache)
    # print data['tv']
    print('Finished')
