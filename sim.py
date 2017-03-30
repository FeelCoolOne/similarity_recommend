# encoding:utf-8

from pandas import DataFrame
from numpy.linalg import norm
import warnings
import traceback

class Sim(object):
    def __init__(self, weight, data):
        self.data = data
        self.out_record_num = 20
        self.weight = weight
        self.min_num_common_feature = 2
        self.indexs = self.data['actor'].index
        self._features_sim = dict()
        self._init_data()
        # self._filter_label()

    def _init_data(self):
        from numpy import max, log10, absolute
        try:
            for key in self.data.keys():
                if key in ['tag', 'country', 'language', 'director']:
                    self._features_sim[key] = absolute(1 - self._calculate_cosine_similarity(self.data[key]))
                elif key in ['score', 'year', 'actor']:
                    sim = self._calculate_euclidean_distance(self.data[key])
                    # feature scaling [0, 1]
                    self._features_sim[key] = sim.where(sim <= 0, log10(sim + 1)) / log10(max(sim + 1))

                    # self._features_sim[key] = (sim - min(sim)) / (max(sim) - min(sim))
                else:
                    raise KeyError('error feature index')
        except Exception:
            traceback.print_exc()
            raise Exception("Catch error during feature similarity")

    def _calculate_euclidean_distance(self, data):
        '''
        calculation Euclidean similarity between samples in dataset
        parameter
        ---------
        data: array like

        output
        ---------
        pandas.DataFrame, distance between samples in data array
        '''
        from numpy import absolute
        if len(data.shape) == 1:
            index = data.index
            tmp = absolute(data - data.reshape(data.size, 1))
            return DataFrame(data=tmp, columns=index, index=index)
        elif len(data.shape) == 2:
            tmp = {cid: norm(data - data.ix[cid, :], axis=1) for cid in data.index}
            '''
            tmp = dict()
            for cid in data.index:
                tmp[cid] = norm(data - data.ix[cid, :], axis=1)
            '''
            return DataFrame(data=tmp, columns=data.index, index=data.index)

    def _calculate_cosine_similarity(self, dataset):
        '''
        calculation cosine similarity between samples in dataset
        Parameters:
            dataset: pandas.DataFrame, shape: (num_sample, num_feature)
        Return：
            array with shape (num_sample, num_sample)
        '''
        # data = dataset
        sample_norms = norm(dataset, axis=1)
        XX = dataset.dot(dataset.T)
        XX_norm = sample_norms.reshape(sample_norms.size, 1) * sample_norms
        sim_array = XX / XX_norm
        return sim_array

    def _filter_label(self):
        tmp = self.data.astype(bool).sum(axis=0)
        invalid_feature_column = tmp.index[tmp < self.min_num_common_feature]
        self.data.drop(labels=invalid_feature_column, axis=1, inplace=True)
        self.weight.drop(labels=invalid_feature_column, axis=1, inplace=True)

    def process(self):
        from numpy import zeros
        data = self._features_sim
        data_size = len(self.indexs)
        all_sim = DataFrame(data=zeros((data_size, data_size)), index=self.indexs, columns=self.indexs)
        for key in data.keys():
            all_sim += data[key] * self.weight[key]
            # print key
            # print data[key]

        for index in self.indexs:
            yield index, self._calculate_output(index, all_sim)

    def weight_search(self, features_sim, train_dataset, patch_size=20, iter_num=5):
        '''
        search weight for every feature one by one, by comparing the ratio of same id between
        the outputs and douban'recommends

        parameters
        ----------
        features_sim: dict
            key->feature name, value->feature similari matrix, pandas.DataFrame
        train_dataset:  dict
            key->cover_id, value->recommend list
        patch_size : int, default 20
            every calculation use samples of [patch_size]
        iter_num: int, default 5
            search in whole weight space for 'num' times

        outputs
        ----------
        weights_dict: dict
        best_score: float
        '''
        from numpy.random import permutation, randint

        if patch_size > len(train_dataset):
            warnings.warn('patch_size should less than train_dataset')
        weight_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_train_cids = train_dataset.keys()
        indexs = features_sim[features_sim.keys()[0]].index
        all_sim = DataFrame(data=np.zeros((len(indexs), len(indexs))), index=indexs, columns=indexs)
        # initial weight dict
        weights_dict = {key: weight_space[randint(len(weight_space))] for key in features_sim}
        # initial similar matrix between samples
        for key, frame in features_sim.iteritems():
            all_sim += weights_dict[key] * frame
        # circle all weights space
        best_score = 0
        print 'origin weight {0} '.format(weights_dict)
        for iter_index in range(iter_num):
            # search at single feature
            old_weights_dict = weights_dict.copy()
            print 'old weight {0}'.format(old_weights_dict)
            for feature, feat_sim in features_sim.iteritems():
                all_sim -= weights_dict[feature] * feat_sim
                # calculate best weight of maximum score
                for feature_weight in weight_space:
                    all_sim += feature_weight * feat_sim
                    idx_tmp = permutation(len(train_dataset))[:patch_size]
                    train_cids = map(lambda idx: all_train_cids[idx], idx_tmp)
                    tmp = {key: train_dataset[key] for key in train_cids}
                    score = self._calculate_metric(tmp, all_sim)
                    all_sim -= feature_weight * feat_sim
                    # update feature weight when better score
                    weights_dict[feature] = feature_weight if best_score < score else weights_dict[feature]
                    best_score = score if best_score < score else best_score

                print '''weight dict: {0} of feature {1},score:{2}
                      '''.format(weights_dict, feature, best_score)
                all_sim += weights_dict[feature] * feat_sim

            # if no change in one search, quit
            if old_weights_dict == weights_dict:
                print 'no change times {0}'.format(iter_index)
                break
        return weights_dict, best_score

    def _calculate_metric(self, stand_dict, similar_all):
        douban_std_rec_num = 10
        same_size = 0
        douban_filtered_total_size = 0
        for cid, std_rec in stand_dict.iteritems():
            try:
                output = self._calculate_output(cid, similar_all, debug=True)
            except Exception:
                # traceback.print_exc()
                # raise
                output = []
                std_rec = []
                print 'calculate output catch error'
            same_size += len(set(output).intersection(set(std_rec)))
            douban_filtered_total_size += len(std_rec)
        douban_filter_miss_num = douban_std_rec_num * len(stand_dict) - douban_filtered_total_size
        score = (same_size + douban_filter_miss_num / 2.0) / (douban_std_rec_num * len(stand_dict))
        return score

    def _calculate_output(self, cover_id, similar_frame, debug=False):
        result = similar_frame.sort_values(by=cover_id, ascending=True, axis=0)[cover_id]
        if debug is True:
            return result.index[1: self.out_record_num + 1]
        format_result = {'Results': result[1: self.out_record_num + 1].to_dict(), 'V': '4.0.0'}
        return format_result


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    weight = {'tag': 0.6, 'actor': 0.6, 'director': 0.8, 'country': 0.5, 'year': 0.3, 'language': 0.2, 'score': 0.2}
    tag = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['t1', 't2', 't3'])
    actor = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['a1', 'a2', 'a3'])
    director = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    country = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    year = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    language = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    score = pd.DataFrame(np.random.rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    douban_test = {'a': ['f', 'b', 'c', 'm'],
                   'b': ['f', 'b', 'c', 'm'],
                   'c': ['f', 'b', 'c', 'm'],
                   'd': ['f', 'b', 'c', 'm'],
                   'u': ['f', 'b', 'c', 'm'], }
    sim = Sim(weight, {'tag': tag, 'actor': actor, 'director': director, 'country': country, 'year': year, 'language': language, 'score': score})
    '''
    for index, result in sim.process():
        print index, result
    '''
    print sim.weight_search(sim._features_sim, douban_test)
