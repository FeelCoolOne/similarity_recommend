# encoding:utf-8
# get data id group by model
# and save to local file '../data/id.dat'
from pymongo import MongoClient
import cPickle as pickle


def init_con(host='127.0.0.1', port=50000):
    con = MongoClient(host, port)
    EPGInfo = con.EPGInfo
    EPGInfo.authenticate(username, password, mechanism='SCRAM-SHA-1')
    collection = EPGInfo['chiq_video_converge']
    # document = collection.find_one({'model': 'movie'})
    return collection


def get_documents(collection, condition, num=10):
    # documents = collection.find(condition).limit(num)
    documents = collection.find(condition)
    return documents


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


class video:
    def __init__(self):
        self.features_handler = {}
        self.init_feature_handler()
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

    def init_feature_handler(self):
        self.features_handler['cartoon'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['doc'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['education'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['entertainment'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['movie'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['sports'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['tv'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
        self.features_handler['variety'] = ['year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']

    def set_feature(self, model, features):
        self.features_handler[model] = features

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

    def handle_all_attr(self, document):
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

    def process(self, collection):
        data = {}
        # print 'get and process data'
        for model in self.models:
            data[model] = set()
            documents = get_documents(collection, {'model': model}, 1)
            for document in documents:
                id = filter_id(document)
                data[model].add(id)
        # print len(data[model])
    return data


def main():
    HOST = '192.168.1.31'
    PORT = 50000
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    print 'connect mongo'
    collection = init_con(HOST, PORT)



if __name__ == '__main__':
    data = main()
    with open('../data/id.dat', 'wb') as f:
        pickle.dump(data, f, protocol=True)
