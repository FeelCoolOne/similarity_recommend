# encoding:utf-8

from pymongo import MongoClient
from pandas import DataFrame
import cPickle as pickle


def init_con():
    username = 'EPGInfo'
    password = 'EPGInfo@20150603'
    con = MongoClient("127.0.0.1", 50000)
    EPGInfo = con.EPGInfo
    EPGInfo.authenticate(username, password)
    collection = EPGInfo['chiq_video_converge']
    # document = collection.find_one({'model': 'movie'})
    return collection


def get_documents(collection, condition, num=10):
    documents = collection.find(condition).limit(num)
    #documents = collection.find(condition)
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
    pass


def fileter_tag(document):
    document.get('tag', '[]')
    pass


features_handler = {}
features_handler['cartoon'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['doc'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['education'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['entertainment'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['movie'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['sports'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['tv'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
features_handler['variety'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']


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
        data['issue'] = document.get('issue', '0000-00-00')
        data['director'] = document.get('director', 'None')
        data['actor'] = document.get('cast', 'None')
        data['grade_score'] = str(document.get('grade_score', -1))
        data['tag'] = fileter_tag(document)
        data['country'] = document.get('country', '[]')
        data['country_group'] = document.get('country_group', '[]')
        data['episodes'] = document.get('episodes', -1)
        data['definitions'] = str(document.get('definitions', -1))
        data['writer'] = document.get('writer', '[]')
        data['year'] = str(filetr_year(document))
        # 上架
        data['enable'] = document.get('enable', '-1')
        # 碎视频
        data['isClip'] = document.get('isClip', '-1')
        data['categorys'] = document.get('categorys', '[]')
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
        for key, value in data.items():
            if key not in features_handler[model]:
                continue
            record.append(value)
        id_stack.append(data['cover_id'])
        data_stack.append(record)
    return DataFrame(data_stack, index=id_stack, columns=features_handler[model])


def main():
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    collection = init_con()
    data = {}
    for model in models:
        data[model] = set()
        documents = get_documents(collection, {'model': model}, 3)
        for document in documents:
            id = filter_id(document)
            data[model].add(id)
    return data


if __name__ == '__main__':
    data = main()
    print data
    with open('video_info.dat', 'wb') as f:
        pickle.dump(data, f, protocol=True)
