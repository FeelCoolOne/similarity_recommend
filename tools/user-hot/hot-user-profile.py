# encoding:utf-8
"""
==========================
calculation of similarity
==========================


Created by:
    yonggang Huang
In:
    06-13-2017
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pymongo import MongoClient
import ConfigParser
from pandas import Series
import cPickle as pickle
# import traceback
from formatter import Formatter


models = dict()
models['movie'] = {'地区': ['华语', '美国', '欧洲', '韩国', '日本', '印度', '泰国', ],
                   '类型': ['爱情', '喜剧', '动作', '恐怖', '悬疑', '科幻', '经典', ],
                   '年代': ['更早', '80年代', '90年代', '2010-2000', '2013-2011', '2014', '2015', '2017-2016', ]
                   }
models['tv'] = {'地区': ['内地', '美剧', '韩剧', '日剧', '台剧', '港剧', '英剧', ],
                '类型': ['言情', '古装', '偶像', '青春', '宫廷', '武侠', '穿越', '家庭', '科幻', '军旅', '历史', '悬疑', ],
                '年代': ['更早', '80年代', '90年代', '2010-2000', '2013-2011', '2014', '2015', '2017-2016', ]
                }
models['entertainment'] = {'类型': ['明星资讯', '影视资讯', '音乐资讯', ]}
models['variety'] = {'类型': ['播报', '情感', '搞笑', '选秀', '游戏', '相亲', '职场', '脱口秀', '真人秀 ', ],
                     '地区': ['内地', '港台', '欧美', '日韩', ]
                     }
models['edcation'] = {'类型': ['幼儿', '小学', '初中', '高中', '外语学习 ', '职业教育 ', '管理培训 ', '实用教程 ', '公开课', ],
                      '年龄': [5, 12, 15, 18, ]
                      }
models['doc'] = {'类型': ['探索', '社会', '人物', '军事', '历史', '文化', '地理', '自然', '典藏', ],
                 '出品方': ['BBC', 'CNEX', 'Discovery', '美国历史频道', '凤凰视频', '央视', ]
                 }
models['cartoon'] = {'类型': ['少年热血', '武侠格斗', '科幻魔幻', '竞技体育', '爆笑喜剧', '侦探推理', '恐怖灵异', '同人', ],
                     '地区': ['日韩', '欧美', '港台', '大陆', ]
                     }
models['sports'] = {'类型': ['足球', '篮球', '网球', '高尔夫', '斯诺克', '赛车', '游泳', '羽毛球', '其他', ], }

alias = {'内地': ['中国大陆', '中国内地', '大陆', '内地', ],
         '港台': ['中国香港', '香港', '中国台湾', '台湾', ],
         '欧美': ['美国', '英国', '意大利', '法国', '新西兰', '德国', '加拿大', '比利时', '荷兰', '澳大利亚', '瑞典', '挪威', ],
         '日韩': ['日本', '韩国', '泰国', '新加坡', ],
         '欧洲': ['英国', '意大利', '法国', '新西兰', '德国', '加拿大', '比利时', '荷兰', '澳大利亚', '瑞典', '挪威', ],
         '华语': ['中国大陆', '中国内地', '大陆', '内地', ],
         '美剧': ['美国', ],
         '韩剧': ['韩国', ],
         '日剧': ['日本', ],
         '台剧': ['中国台湾', '台湾', ],
         '港剧': ['中国香港', '香港', ],
         '英剧': ['英国', '新西兰', '加拿大', '比利时', '澳大利亚', '挪威', ],
         '家庭': ['家庭', '都市', '生活', '农村'],
         '历史': ['历史', '抗日', '革命', '宫斗', '古装', '史诗', '谍战'],
         '宫廷': ['宫廷', '宫斗', ],
         '军旅': ['军旅', '战争', '抗日', '谍战'],
         '言情': ['言情', '爱情', '情感', ],
         '悬疑': ['悬疑', '警匪', '刑侦', ],
         }


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


def search_ids(data_df, character):
    global alias
    labels = alias.get(character, None)
    labels = [character] if labels is None else labels
    labels = list(set(labels) & set(data_df.columns))
    matched_data = data_df[labels]
    matched_data.dropna(axis=1, how='all')
    count_data = matched_data.sum(axis=1)
    result = count_data[count_data > 0].index.tolist()
    return result


def main(config_file):
    import pandas as pd

    cf = ConfigParser.ConfigParser()
    cf.read(config_file)
    address = cf.get('mongo', 'address')
    port = int(cf.get('mongo', 'port'))
    username = cf.get('mongo', 'username')
    password = cf.get('mongo', 'password')
    database = cf.get('mongo', 'database')
    collection = cf.get('mongo', 'collection')
    collection = connect_mongodb(address, port, username, password, database, collection)
    handler = Formatter()
    year_sep = [1900, 1979, 1989, 1999, 2010, 2013, 2014, 2015, 2017]
    year_labels = ['更早', '80年代', '90年代', '2010-2000', '2013-2011', '2014', '2015', '2017-2016']
    num = 100
    for model, characters in models.iteritems():
        if model not in ['tv', 'movie']:  continue
        print('{0} start ...'.format(model))
        documents = collection.find({'model': model}).limit(num)
        # documents = collection.find({'model': model})
        all_model_data_df = handler.process(documents)
        print('origin record get ready')
        data_df = all_model_data_df

        year_df = pd.cut(data_df['year'], year_sep, labels=year_labels)  # discretization
        year = pd.get_dummies(year_df, columns=year_labels)
        year.columns = year.columns.astype(str)
        country = multi_label_format(data_df['country'])
        category = multi_label_format(data_df['country'])

        train_vec = list()
        train_vec.append(year[~year.index.duplicated(keep='first')])
        train_vec.append(country[~country.index.duplicated(keep='first')])
        train_vec.append(category[~category.index.duplicated(keep='first')])
        train_vec.append(data_df['cover_id'].drop_duplicates(keep='first'))
        data = pd.concat(train_vec, axis=1)
        data.drop_duplicates(subset=['cover_id'], inplace=True)
        data.set_index('cover_id', drop=True, inplace=True)

        all_labels_ids = dict()
        for type_label, d in characters.iteritems():
            all_labels_ids[type_label] = dict()
            for label in d:
                all_labels_ids[type_label][label] = list()
                ids = search_ids(data, label)
                all_labels_ids[type_label][label].extend(ids)
        print 'Map ids for labels finished'

        print 'save ids for model {0}'.format(model)
        with open(('data/%s_label_ids.dat' % model), 'wb') as f:
            pickle.dump(all_labels_ids, f, protocol=True)


if __name__ == '__main__':
    main(r'E:/gitlab/etc/config.ini')
