# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame
import sim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

weights_all = {}
weights_all['cartoon'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['doc'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['education'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['entertainment'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['movie'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['sports'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['tv'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['variety'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])

models = ['cartoon', 'doc', 'education', 'entertainment', 'movie', 'sports', 'tv', 'variety']
features = ['id', 'model', 'year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']


def read_data(filename):
    global models
    global features
    data_all = {}
    ids_all = {}
    for model in models:
        data_all[model] = []
        ids_all[model] = []
    for line in open(filename, 'r'):
        try:
            words = line.strip().split('\t')
            cover_id = words[0]
            model = words[1]
            data_all[model].append(words[2:])
            ids_all[model].append(cover_id)
        except:
            continue
    for model in models:
        data_all[model] = DataFrame(data_all[model], index=ids_all[model], columns=features[2:])
    return ids_all, data_all


if __name__ == '__main__':
    # filename = ''
    # _, data_all = read_data(filename)
    '''
    for model in models:
        samples = data_all[model]
        features_weight = weights_all['cartoon']
        model_sim = sim.Cartoon_Sim(model, samples, features_weight)
        model_sim.process()
    '''
    model = 'tv'
    # samples = data_all[model]
    features_weight = weights_all['tv']
    data = [['2006', '["ni","jiao","bu","lai","xx"]', '["chenglong"]', '["chenglong"]', '["China"]', '22', '["lixiang","xiangli"]', 'english', '46'],
            ['2007', '["jiao","bu","lai"]', '["成龙"]', '["chenglong"]', '["China"]', '24', '["lixiang","xiangli"]', 'english', '80'],
            ['2005', '["jiao","bu","lai"]', '["成龙"]', '["chenglong"]', '["China"]', '20', '["lixiang","xiangli"]', 'english', '50']]
    samples = DataFrame(data, index=['a', 'b', 'c'], columns=features[2:])
    model_sim = sim.TV_Sim(samples, features_weight)
    model_sim.process()
