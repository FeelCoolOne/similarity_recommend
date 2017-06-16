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
import pandas as pd
import itertools as it
import cPickle as pickle
from datetime import datetime

day_of_week = datetime.now().weekday()
MODELS = dict()
MODELS['movie'] = {'地区': ['华语', '美国', '欧洲', '韩国', '日本', '印度', '泰国', ],
                   '类型': ['爱情', '喜剧', '动作', '恐怖', '悬疑', '科幻', '经典', ],
                   '年代': ['更早', '80年代', '90年代', '2010-2000', '2013-2011', '2014', '2015', '2017-2016', ]
                   }
MODELS['tv'] = {'地区': ['内地', '美剧', '韩剧', '日剧', '台剧', '港剧', '英剧', ],
                '类型': ['言情', '古装', '偶像', '青春', '宫廷', '武侠', '穿越', '家庭', '科幻', '军旅', '历史', '悬疑', ],
                '年代': ['更早', '80年代', '90年代', '2010-2000', '2013-2011', '2014', '2015', '2017-2016', ]
                }
MODELS['entertainment'] = {'类型': ['明星资讯', '影视资讯', '音乐资讯', ]}
MODELS['variety'] = {'类型': ['播报', '情感', '搞笑', '选秀', '游戏', '相亲', '职场', '脱口秀', '真人秀 ', ],
                     '地区': ['内地', '港台', '欧美', '日韩', ]
                     }
MODELS['education'] = {'类型': ['幼儿', '小学', '初中 ', '高中', '外语学习 ', '职业教育 ', '管理培训 ', '实用教程 ', '公开课', ],
                       '年龄': [5, 12, 15, 18, ]
                       }
MODELS['doc'] = {'类型': ['探索', '社会', '人物', '军事', '历史', '文化', '地理', '自然', '典藏', ],
                 '出品方': ['BBC', 'CNEX', 'Discovery', '美国历史频道', '凤凰视频', '央视', ]
                 }
MODELS['cartoon'] = {'类型': ['少年热血', '武侠格斗', '科幻魔幻', '竞技体育', '爆笑喜剧', '侦探推理', '恐怖灵异', '同人', ],
                     '地区': ['日韩', '欧美', '港台', '大陆', ]
                     }
MODELS['sports'] = {'类型': ['足球', '篮球', '网球', '高尔夫', '斯诺克', '赛车', '游泳', '羽毛球', '其他', ], }

KEY_PATTERNS = 'AlgorithmsCommonBid_Cchiq3Test:UPHOT:ITI:'
LEVEL_ONE_LABEL = 'ZDY10018'
LEVEL_TWO_LABEL_MAP = dict()
LEVEL_TWO_LABEL_MAP['work_day'] = {'a': 'ZDY00002', 'b': 'ZDY00001', 'c': 'ZDY00003', 'd': 'ZDY00004', 'e': 'ZDY00005',
                                   'f': 'ZDY00006', 'g': 'ZDY00007', 'h': 'ZDY00008', 'i': 'ZDY00009', 'j': 'ZDY00010', }
LEVEL_TWO_LABEL_MAP['non_work_day'] = {'a': 'ZDY00011', 'b': 'ZDY00012', 'c': 'ZDY00013', 'd': 'ZDY00014', 'e': 'ZDY00015',
                                       'f': 'ZDY00016', 'g': 'ZDY00017', 'h': 'ZDY00018', 'i': 'ZDY00019', 'j': 'ZDY00020', }
LEVEL_THREE_LABEL_MAP = dict()
LEVEL_THREE_LABEL_MAP['movie'] = {'类型': 'film_type', '年代': 'film_years', '地区': 'film_region', }
LEVEL_THREE_LABEL_MAP['tv'] = {'类型': 'teleplay_type', '年代': 'teleplay_years', '地区': 'teleplay_region', }
LEVEL_THREE_LABEL_MAP['variety'] = {'类型': 'show_type', '地区': 'show_region', }
LEVEL_THREE_LABEL_MAP['cartoon'] = {'类型': 'comic_type', '地区': 'comic_region', }
LEVEL_THREE_LABEL_MAP['doc'] = {'类型': 'dmentary_type', '出品方': 'dmentary_producer', }
LEVEL_THREE_LABEL_MAP['entertainment'] = {'类型': 'amuse_type', }
LEVEL_THREE_LABEL_MAP['sports'] = {'类型': 'sports_type', }
LEVEL_THREE_LABEL_MAP['music'] = {'类型': 'music_type', '分类': 'music_region', }
LEVEL_THREE_LABEL_MAP['education'] = {'类型': 'edu_type', '年龄': 'edu_age', }
# LEVEL_THREE_MAP['others'] = {'类型': 'other_type', }
# LEVEL_THREE_MAP['tv'] = {'类型': 'funny_type', }
LEVEL_TWO_LABEL_MAP = LEVEL_TWO_LABEL_MAP['work_day'] if day_of_week not in [5, 6] else LEVEL_TWO_LABEL_MAP['non_work_day']


def merge_and_sort(data, label_ids, out_n=20):
    timeperid = data['timeperiod'].unique()
    for t in timeperid:
        tmp = data[data['timeperiod'] == t]
        label_ids = list(set(label_ids) & set(tmp.index))
        if len(label_ids) == 0:
            yield t, {}
        tmp = tmp.loc[label_ids]
        tmp.dropna(axis=0, how='any', inplace=True)
        result = tmp['pnt'].sort_values(ascending='False')[:out_n].to_json()
        yield t, result


def standardize_result(model, type_, label, timeperiod, result,):
    # level one, level two - time, level 3 - model and type_, addition label
    if all([isinstance(tmp, (tuple, list)) for tmp in (model, type_, label,)]):
        t = ':'.join([LEVEL_THREE_LABEL_MAP[m][t] for m, t in zip(model, type_)])
        lab = '|'.join(list(label))
        key = (KEY_PATTERNS + LEVEL_ONE_LABEL + ':' + LEVEL_TWO_LABEL_MAP[timeperiod] +
               ':' + t + ':' + lab)
        result = {'Results': result, 'V': '0.1.1'}
        return key, result
    elif all([isinstance(tmp, (str)) for tmp in (model, type_, label,)]):
        key = (KEY_PATTERNS + LEVEL_ONE_LABEL + ':' + LEVEL_TWO_LABEL_MAP[timeperiod] +
               ':' + LEVEL_THREE_LABEL_MAP[model][type_] + ':' + label)
        result = {'Results': result, 'V': '0.1.1'}
        return key, result
    else:
        raise TypeError('of correct type')


def label_hot(data_df, config, mode='single'):
    '''
    @ parameter mode:
        single: one label
        inbox_mix  : multi type label in same model
        mix: term from multi model
    '''
    global MODELS
    ALL = dict()
    for model, types in MODELS.iteritems():
        if model not in ['moive', 'tv']: continue
        with open('./data/%s_label_ids.dat' % model, 'rb') as f:
            model_ids = pickle.load(f)
        if 'inbox_mix' == mode:
            for labels in it.product(*types.itervalues()):
                label_ids = list()
                for i, j in zip(types, labels):
                    label_ids.extend(model_ids[i][j])

                for t, result in merge_and_sort(data_df, list(set(label_ids))):
                    print model, labels, t, result
        elif 'single' == mode:
            for type_, labels in types.iteritems():
                for label in labels:
                    label_ids = model_ids[type_][label]
                    if len(label_ids) == 0:
                        continue
                    for t, result in merge_and_sort(data_df, label_ids):
                        print model, type_, label, t, result
                        key, result = standardize_result(model, type_, label, t, result)
                        print key, result
        elif 'mix' == mode:
            ALL[model] = model_ids
        else:
            raise ValueError('mode must be one of [mix, single]')
    if 'mix' == mode:
        # for models each number of models can in the mix
        for i in range(len(MODELS)):
            # do combinations
            for models_tuple in it.combinations(MODELS.keys(), i):
                tmp1 = [MODELS[model].keys() for model in models_tuple]
                # do Cartesian product for category of each model
                for cates_tuple in it.product(*tmp1):
                    tmp2 = [MODELS[model][cate] for model, cate in zip(models_tuple, cates_tuple)]
                    # collect ids of labels in each categorys as label_ids
                    for labels_tuple in it.product(*tmp2):
                        label_ids = list()
                        for m, c, l in zip(models_tuple, cates_tuple, labels_tuple):
                            label_ids.extend(ALL[m][c][l])
                        for t, result in merge_and_sort(data_df, label_ids):
                            print models_tuple, cates_tuple, labels_tuple, t, result
                            key, result = standardize_result(models_tuple, cates_tuple, labels_tuple, t, result)
        pass


def main():
    # CHECK: if data of yesterday for Saturday and Monday should be used
    data1 = pd.read_csv(r'E:/sublime_workspace/data/valid_data.dat', sep='\s+', header=None)
    data1.columns = ['cover_id', 'timeperiod', 'pnt']
    data1.set_index('cover_id', drop=True, inplace=True)
    label_hot(data1, None)


if __name__ == '__main__':
    main()
