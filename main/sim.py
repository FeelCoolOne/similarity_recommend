# encoding:utf-8
from __future__ import division
from pandas import DataFrame
from numpy.linalg import norm
from numpy.random import permutation, randint
import numpy as np
from scipy import sparse as sp
import gc
import warnings


class Sim(object):

    def __init__(self, data, index, weight=None, static_=False):
        """
        Parameter:
            static_: bool
                case False: array_like objects in data.
                case True : file's names in storage system in data.
        """
        self.data = data
        self.out_num = 20
        self.weight = weight
        self.min_num_common_feature = 2
        self.douban_std_rec_num = 10
        self.indexs = index
        self._features_sim = dict()
        self._init_data(static_)
        # self._filter_label()

    def set_weight(self, weight):
        self.weight = weight

    def _init_data(self, static_):

        keys = self.data.keys()
        for key in keys:
            tmp = self.data.pop(key)
            if static_:
                tmp = np.load(tmp)
            if key in ['tag', 'country', 'language', ]:
                self._features_sim[key] = np.absolute(1 - self._calculate_cosine_similarity(tmp))
            elif key in ['grade_score', 'year', ]:
                s = self._calculate_euclidean_distance(tmp)
                self._features_sim[key] = np.true_divide(s, np.max(s + 1))  # feature scaling [0, 1]
            elif key in ["cast", 'director', ]:
                tmp = sp.csr_matrix(tmp)
                self._features_sim[key] = 1 - self._calculate_jaccard_similarity(tmp, True)
            else:
                raise KeyError('error feature index')
            del tmp
            gc.collect()

    def _calculate_euclidean_distance(self, data):
        '''
        calculation Euclidean similarity between samples in dataset
        parameter
        ---------
        data: numpy array

        output
        ------
        numpy array, distance between samples in data array
        '''
        if len(data.shape) == 1:
            tmp = np.absolute(data.reshape(data.size, 1) - data)
            return tmp
        elif len(data.shape) == 2:
            size = data.shape[0]
            tmp = np.zeros((size, size), dtype=float)
            for index in range(size):
                tmp[index, :] = norm(data - data[index, :], axis=1)
            return tmp
        else:
            raise TypeError('data should be 1 dimension or 2 dimension')

    def _calculate_cosine_similarity(self, dataset):
        '''
        calculation cosine similarity between samples in dataset
        -----------

        Parameters:
            dataset: numpy.array, shape: (num_sample, num_feature)

        Return：
            array with shape (num_sample, num_sample)
        '''
        # data = dataset
        sample_norms = norm(dataset, axis=1)
        XX = dataset.dot(dataset.T)
        XX_norm = sample_norms.reshape(sample_norms.size, 1) * sample_norms
        sim_array = np.true_divide(XX, XX_norm, out=np.zeros_like(XX), where=(XX_norm != 0))
        return sim_array

    def _calculate_jaccard_similarity(self, data, sparse=False):
        X = data.astype(bool).astype(int)
        intersect = X.dot(X.T)
        row_sums = intersect.diagonal()
        unions = row_sums[:, None] + row_sums - intersect
        if sparse:
            intersect = intersect.toarray()
        smy = np.divide(intersect, unions, out=np.zeros(intersect.shape, dtype=float), where=(unions != 0))
        return smy

    def _filter_label(self):
        tmp = self.data.astype(bool).sum(axis=0)
        invalid_feature_column = tmp.index[tmp < self.min_num_common_feature]
        self.data.drop(labels=invalid_feature_column, axis=1, inplace=True)
        self.weight.drop(labels=invalid_feature_column, axis=1, inplace=True)

    def process(self):

        n_sample = len(self.indexs)
        all_sim = DataFrame(data=np.zeros((n_sample, n_sample)), index=self.indexs, columns=self.indexs)
        keys = self._features_sim.keys()
        for key in keys:
            tmp = self._features_sim.pop(key)
            all_sim += tmp * self.weight[key]
            del tmp
            gc.collect()
        all_ = all_sim.round({c: 5 for c in all_sim.columns})
        for index in self.indexs:
            yield index, self._calculate_output(index, all_)
        del all_

    def weight_search(self, train_dataset, features_sim=None, patch_size=20,
                      iter_num=10, seed_times=20, verbose=False):
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
        iter_num: int, default 3
            search in whole weight space for 'num' times in a seed process
        seed_times: int, default 3
            search action will try by seed_times, return best weight of highest score
            since the search action might catch local optimization
        verbose: bool, default False
            print the search infomation for trace process track

        outputs
        ----------
        weight: dict
        score: float
        '''
        if features_sim is None:
            features_sim = self._features_sim
        if patch_size > len(train_dataset):
            warnings.warn('patch_size should less than train_dataset')
        weight_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_train_cids = train_dataset.keys()
        indexs = features_sim[features_sim.keys()[0]].index
        seed_result_keep = []
        # start seed
        for seed_index in range(seed_times):
            all_sim = DataFrame(data=np.zeros((len(indexs), len(indexs))), index=indexs, columns=indexs)
            # initial weight dict
            weights_dict = {key: weight_space[randint(len(weight_space))] for key in features_sim}
            # initial similar matrix between samples
            for key, frame in features_sim.iteritems():
                all_sim += weights_dict[key] * frame
            # circle all weights space
            best_score = 0
            if verbose is True:
                print('origin weight {0}'.format(weights_dict))
            for iter_index in range(iter_num):
                # search at single feature
                old_weights_dict = weights_dict.copy()
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

                    if verbose is True:
                        print("weight dict: {0} of feature {1}, score: {2}"
                              .format(weights_dict, feature, best_score))
                    all_sim += weights_dict[feature] * feat_sim

                # if no change in one search, quit
                if old_weights_dict == weights_dict:
                    if verbose is True:
                        print('no change times {0}'.format(iter_index))
                    break
            seed_result_keep.append([best_score, weights_dict])
        seed_result_keep.sort(key=lambda x: x[0], reverse=True)
        self.weight = seed_result_keep[0][1]
        self.score = seed_result_keep[0][0]
        return self.weight, self.score

    def _calculate_metric(self, stand_dict, similar_all):
        same_size = 0
        douban_filtered_total_size = 0
        for cid, std_rec in stand_dict.iteritems():
            try:
                output = self._calculate_output(cid, similar_all, debug=True)
            except Exception:
                # traceback.print_exc()
                # raise()
                output = []
                std_rec = []
            same_size += len(set(output).intersection(set(std_rec)))
            douban_filtered_total_size += len(std_rec)
        douban_filter_miss_num = self.douban_std_rec_num * len(stand_dict) - douban_filtered_total_size
        score = (same_size + douban_filter_miss_num / 2.000) / (self.douban_std_rec_num * len(stand_dict))
        return score

    def _calculate_output(self, cover_id, df, debug=False):
        sorted_index = np.argsort(df[cover_id])[1: self.out_num + 1]
        if debug is True:
            return df[cover_id][sorted_index]
        format_result = {'Results': df[cover_id][sorted_index].to_dict(), 'V': '5.0.1'}
        return format_result
