# encoding:utf-8
from pymongo import MongoClient
import ConfigParser
import cPickle as pickle
import traceback
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


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


def filter_id(document):
    record = document
    id = None
    if record['tencent'] == '1' and 'pp_tencent' in record:
        id = record['pp_tencent']['tencentId']
    elif record['youpeng'] == '1' and 'pp_youpeng' in record:
        id = record['pp_youpeng']['yp_id']
    return id


def get_documents(collection, model, num=10):
    projection = ['d_entityId', 'refereredDetail', 'tencent', 'youpeng', 'pp_tencent', 'pp_youpeng']
    # documents = collection.find(filter={'model': model}, projection=projection).limit(num)
    # documents = collection.find(filter={'model': model}).limit(num)
    documents = collection.find(filter={'model': model}, projection=projection)
    return documents


def main():
    config_file = './etc/config.ini'
    data_file_path = './data'
    cf = ConfigParser.ConfigParser()
    cf.read(config_file)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    coll = connect_mongodb(address, port, username,
                           password, database, collection)
    models = ['movie', 'tv',
              'sports', 'entertainment', 'variety',
              'education', 'doc', 'cartoon']
    for model in models:
        id_map = dict()
        d_result = dict()
        # if model not in ['tv', 'movie']: continue
        documents = get_documents(coll, model, 1000)
        for doc in documents:
            c_id = filter_id(doc)
            if c_id is not None and 'd_entityId' in doc:
                id_map[doc.get('d_entityId').strip()] = c_id
            if c_id is not None and 'refereredDetail' in doc:
                d_result[c_id] = doc.get('refereredDetail').strip()
        print model
        pat = re.compile(r'\s*[/\\|]\s*')
        standard_result = dict()
        for cid, rec in d_result.iteritems():
            tmp = list()
            for index in pat.split(rec):
                if index in id_map:
                    tmp.append(id_map[index])
            if len(tmp) > 0:
                standard_result[cid] = tmp
        with open(data_file_path + r'/' + model + '_douban' + r'.dat', 'wb') as f:
            pickle.dump(standard_result, f, protocol=True)
        print '{0} get result: {1}'.format(model, len(standard_result))


if __name__ == '__main__':
    main()
