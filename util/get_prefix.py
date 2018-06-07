# encoding=utf-8

from pymongo import MongoClient
import ConfigParser
import cPickle as pickle
# import traceback


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


def prefix_map(con):
    documents = con.find()
    item_prefix_map = dict()
    for document in documents:
        cid = document["item_id"]
        value = document["value"]
        if len(cid) < 5 or len(value) < 5:
            print 'invalid record item_id: {0}, value: {1}'.format(cid, value)
            continue
        item_prefix_map[cid] = value
    return item_prefix_map


def value_fix(value, prefix_map):
    '''
        add prefix for every item's value

        Parameters:
        ----------
        value: {key1:value1, key2:value2,...}, dict
        prefix_map: {key:prefix,...}, dict

        Returns:
        --------
        new_value:  {key1:value1, key2:value2,...}, dict
    '''
    result = value
    for key, value in result['Results'].iteritems():
        prefix = prefix_map.get(key, None)
        # filter out non-prefix record in recommend result
        if prefix is None:
            result['Results'].pop(key, None)
            continue
        result['Results'][key] = prefix + ':' + result['Results'][key]
    return result


def main():
    config_file = './etc/config.ini'
    data_file_path = './data'
    cf = ConfigParser.ConfigParser()
    cf.read(config_file)
    MONGO_ADDRESS = cf.get('prefix', 'address')
    MONGO_PORT = int(cf.get('prefix', 'port'))
    USERNAME = cf.get('prefix', 'username')
    PASSWORD = cf.get('prefix', 'password')
    DATABASE = cf.get('prefix', 'database')
    COLLECTION = cf.get('prefix', 'collection')
    con = connect_mongodb(MONGO_ADDRESS,
                          MONGO_PORT,
                          USERNAME,
                          PASSWORD,
                          DATABASE,
                          COLLECTION)
    prefixs = prefix_map(con)
    with open(data_file_path + r'/' + 'prefix_map' + r'.dat', 'wb') as f:
        pickle.dump(prefixs, f, protocol=True)


if __name__ == "__main__":
    main()
