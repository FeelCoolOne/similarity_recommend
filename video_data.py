# -*- coding: utf-8 -*-

from pymongo import MongoClient
from pandas import DataFrame
import cPickle as pickle
import ConfigParser


def init_con(config_file_path):
    cf = ConfigParser.ConfigParser()
    cf.read(config_file_path)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    con = MongoClient(address, port)
    EPGInfo = con[database]
    EPGInfo.authenticate(username, password)
    collection = EPGInfo[collection]
    # document = collection.find_one({'model': 'movie'})
    return collection


def get_documents(collection, condition, num=10):
    # documents = collection.find(condition).limit(num)
    documents = collection.find(condition)
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
    pass


def filetr_year(document):
    year = document.get('year', 'None')
    if year == 'None':
        year = document.get('issue', 'None').split('-')[0]
    return year


def fileter_tag(document):
    tag = document.get('tag', 'None').split('/')
    return tag


features_handler = {}
features_handler['cartoon'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['doc'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['education'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['entertainment'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['movie'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['sports'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['tv'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['variety'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']


# TODO
# how to process None feature
def format_input_record(collection, model):
    # TODO
    # tag diff category
    # duration equal 0
    data_stack = []
    id_stack = []
    for document in collection:
        data = {}
        document = dict(document)
        data['cover_id'] = filter_id(document)
        data['model'] = document.get('model', 'None')
        data['alias'] = document.get('alias', 'None')
        data['duration'] = str(document.get('duration', -1))
        data['enname'] = document.get('enName', 'None')
        data['language'] = document.get('language', 'None')
        data['name'] = document.get('name', 'None')
        # data['issue'] = document.get('issue', '0000-00-00')
        data['director'] = str(document.get('director', 'None').split('/'))
        data['actor'] = str(document.get('cast', 'None').split('/'))
        data['grade_score'] = str(document.get('grade_score', -1))
        data['tag'] = str(fileter_tag(document))
        data['country'] = str(document.get('country', 'None').split('/'))
        # TODO data['country_group'] = document.get('country_group', '[]')
        data['episodes'] = document.get('episodes', -1)
        data['definitions'] = str(document.get('definitions', -1))
        data['writer'] = str(document.get('writer', 'None').split('/'))
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
        data['focus'] = document.get('focus', 'None')
        data['pay_status'] = filter_pay_status(document)

        if data['enable'] in ['0', '-1'] or data['cover_id'] == '-1':
            continue
        record = []
        for feature in features_handler[model]:
            record.append(data[feature])
        id_stack.append(data['cover_id'])
        data_stack.append(record)
    return DataFrame(data_stack, index=id_stack, columns=features_handler[model])


def main():
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    config_file = './mongodb.ini'
    collection = init_con(config_file)
    data = {}
    for model in models:
        documents = get_documents(collection, {'model': model}, 3)
        data[model] = format_input_record(documents, model)
    return data


if __name__ == '__main__':
    data = main()
    # print data['tv']

    # for key in data:
    #     print data[key]

    with open('all_video_info.dat', 'wb') as f:
        pickle.dump(data, f, protocol=True)
