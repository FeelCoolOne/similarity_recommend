# encoding:utf-8
"""
==========================
calculation of similarity
==========================

charactor:
    'language', 'country', 'writer', 'director', 'tag', 'actor', 'year', 'score'

feature:
    weight for character can be search from douban standard recommends

Created by:
    yonggang Huang
In:
    05-25-2017
"""

from numpy.linalg import norm
from numpy.random import permutation, randint
from lazysorted import LazySorted
import warnings
import traceback


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    from numpy import isscalar, ndarray
    if isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class Sim(object):

    def __init__(self, weight=None, index=None, feat_properties=None, std_output=None):
        self._weight = weight
        self._indexs = index
        self._feat_properties = feat_properties
        self._std_output = std_output
        self._out_record_num = 20
        self.douban_std_rec_num = 10

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_weight):
        self._weight = new_weight

    @property
    def indexs(self):
        return self._indexs

    @indexs.setter
    def indexs(self, new_indexs):
        from numpy import array
        if isinstance(new_indexs, array):
            self._indexs = new_indexs
        elif isinstance(new_indexs, list):
            self._indexs = array(new_indexs)
        else:
            raise TypeError('type must be numpy.array')

    def _check_property(self, data, attr_name):
        if attr_name == 'data':
            from numpy import sum, array
            total = sum(array(self._feat_properties))
            n_col = data.shape[-1]
            if total != n_col:
                return False
            return True
        else:
            raise ValueError('attr_name not fit')

    def _calculate_euclidean_distance(self, data):
        '''
        calculation Euclidean similarity between samples in dataset
        parameter
        ---------
        data: array like

        output
        ------
        pandas.DataFrame, distance between samples in data array
        '''
        from numpy import shape, zeros
        m, _ = shape(data)
        distance = zeros((m, m))
        for i in range(m):
            distance[i, :] = norm(data - data[i, :], axis=1)
        return distance

    def _calculate_cosine_similarity(self, data):
        '''
        calculation cosine similarity between samples in dataset
        -----------

        Parameters:
            data: pandas.DataFrame, shape: (num_sample, num_feature)

        Return：
            array with shape (num_sample, num_sample)
        '''
        norms = norm(data, axis=1)
        XX = data.dot(data.T)
        XX_norm = norms.reshape(norms.size, 1) * norms
        XX_norm = _handle_zeros_in_scale(XX_norm)
        distance = XX / XX_norm
        return distance

    def fit(self, X, y=None):
        from numpy import ones, concatenate, arange, ndarray, array
        if not self._check_property(X, attr_name='data'):
            raise ValueError('X must match property')
        if self._indexs is None:
            warnings.warn('index is None, use origin numerical index default')
            self._indexs = arange(X.shape[0])
        if len(self._indexs) != X.shape[0]:
            raise ValueError('X must match indexs')
        if not isinstance(self._indexs, ndarray):
            self._indexs = array(self._indexs)
        if self._weight is None:
            warnings.warn('without given weight, start search')
            if not hasattr(self, '_std_output'):
                raise Exception('std_output must not to be None')
            self._weight, self.score = self.weight_search(X, self._std_output, verbose=False)
        elif len(self._weight) != len(self._feat_properties):
            raise Exception('weight must match feat properties')
        self._weight = array(self._weight, dtype='float64')
        weight = concatenate([tmp * w for tmp, w in zip(map(lambda x: ones(x), self._feat_properties), self._weight)])
        self.data = X * weight

    def transform(self, X=None):
        if not hasattr(self, 'data'):
            raise Exception('fit must be taken ahead of transform')
        self.distance = self._calculate_euclidean_distance(self.data)
        n_sample = self.data.shape[0]
        for i in range(n_sample):
            yield self._indexs[i], self._calculate_output(self._indexs, self.distance[i, :])

    def weight_search(self, X, std, patch_size=20, iter_num=10, seed_times=20, verbose=False):
        '''
        search weight for every feature one by one, by comparing the ratio of same id between
        the outputs and douban'recommends

        Parameters
        ----------
        features_sim: dict
            key->feature name, value->feature similari matrix, pandas.DataFrame
        std:  dict
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

        Returns
        ------------
        weight: list like
        score: float
        '''
        from numpy import ones, concatenate
        if len(std) < patch_size:
            warnings.warn('patch_size should less than std')
        weight_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_std_cids = std.keys()
        seed_result_cache = []
        # start seed
        n_feature = len(self._feat_properties)
        for _ in range(seed_times):
            # initial weight
            weight = [weight_space[randint(len(weight_space))]
                      for _ in range(n_feature)]
            # initial similar matrix between samples
            # circle all weights space
            best_score = 0
            if verbose is True:
                print '0 weight {0}'.format(weight)
            # todo: for to while
            for iter_index in range(iter_num):
                # search at single feature
                old_weight = list(weight)
                for feat_id in range(n_feature):
                    # calculate best weight of maximum score
                    for w in weight_space:
                        weight[feat_id] = w
                        weight_vec = concatenate([tmp * w for tmp, w in zip(map(lambda x: ones(x), self._feat_properties), weight)])
                        distance = self._calculate_euclidean_distance(X * weight_vec)
                        # select batch of std douban result for scoreing the weight
                        idx_tmp = permutation(len(std))[:patch_size]
                        train_cids = map(lambda idx: all_std_cids[idx], idx_tmp)
                        tmp = {key: std[key] for key in train_cids}
                        score = self._calculate_metric(tmp, distance, self._indexs)
                        # update feature weight when better score
                        weight[feat_id] = w if best_score < score else weight[feat_id]
                        best_score = score if best_score < score else best_score

                    if verbose is True:
                        print('weight dict: %d of feature %d, score: %d'
                              % (weight, feat_id, best_score))

                # if no change in one search, quit
                if old_weight == weight:
                    if verbose is True:
                        print 'no change times {0}'.format(iter_index)
                    break
            seed_result_cache.append([best_score, weight])
        seed_result_cache.sort(key=lambda x: x[0], reverse=True)
        w = seed_result_cache[0][1]
        s = seed_result_cache[0][0]
        return w, s

    def _calculate_metric(self, std_dict, dist_all, indexs):
        same_size = 0
        douban_filtered_total_size = 0
        for cid, std_rec in std_dict.iteritems():
            try:
                record_dist = dist_all[indexs == cid, :].flatten()
                if record_dist.size == 0:
                    raise ValueError('cid of std_dict not in indexs')
                output = self._calculate_output(indexs, record_dist, debug=True)
            except Exception:
                traceback.print_exc()
                # raise()
                output = []
                std_rec = []
            same_size += len(set(output).intersection(set(std_rec)))
            douban_filtered_total_size += len(std_rec)
        douban_filter_miss_num = self.douban_std_rec_num * len(std_dict) - douban_filtered_total_size
        score = (same_size + douban_filter_miss_num / 2.000) / (self.douban_std_rec_num * len(std_dict))
        return score

    def _calculate_output(self, ids, data, debug=False):
        from numpy import argsort
        if len(data.shape) != 1:
            raise TypeError('data must be 1-dimension, but {0} given'.format(len(data.shape)))
        sorted_ids = argsort(data, kind='quicksort')[:self._out_record_num + 1]
        if debug is True:
            return ids[sorted_ids]
        tmp = {ids[i]: data[i] for i in sorted_ids}
        format_result = {'Results': tmp, 'V': '5.0.0'}
        return format_result

    def _calculate_output_lazy(self, ids, data, debug=False):
        xs = [(t1, t2) for t1, t2 in zip(ids, data)]
        ls = LazySorted(xs, key=lambda x: x[1], reverse=True)
        if debug is True:
            return map(lambda x: x[0], ls[:self._out_record_num])
        tmp = {x[0]: x[1] for x in ls[:self._out_record_num]}
        format_result = {'Results': tmp, 'V': '5.0.0'}
        return format_result
