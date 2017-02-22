# encoding:utf-8


class Sim(object):
    def __init__(self, weight, data):
        self.data = data
        self.num = 20
        self.weight = weight
        self.min_num_common_feature = 2
        self._init_data()
        self._filter_label()

    def _init_data(self):
        from pandas import concat
        try:
            data = [self.data[key] * self.weight[key] for key in self.data.keys()]
        except:
            raise("data key and weight key is not matched")
        self.data = concat(data, axis=1)

    def _filter_label(self):
        tmp = self.data.astype(bool).sum(axis=0)
        invalid_feature_column = tmp.index[tmp < self.min_num_common_feature]
        self.data.drop(labels=invalid_feature_column, axis=1, inplace=True)

    def process(self):
        data = self.data
        for index in data.index:
            tmp = 2 * (data * data.loc[index]) / (data + data.loc[index])
            sim_record = tmp.sum(axis=1).drop(index)
            yield index, self.calculate_output(index, sim_record)

    def calculate_output(self, cover_id, sim_record):
        sim_record.sort_values(ascending=False, inplace=True)
        format_result = {'Results': sim_record[:self.num].to_dict(), 'V': '3.0.0'}
        return format_result


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    weight = {'tag': 1, 'actor': 1.2, 'director': 1.4}
    tag = pd.DataFrame(np.random.rand(4,3), index=['a','b','c','d'], columns=['t1','t2','t3'])
    actor = pd.DataFrame(np.random.rand(4,3), index=['a','b','c','d'], columns=['a1','a2','a3'])
    director = pd.DataFrame(np.random.rand(4,3), index=['a','b','c','d'], columns=['d1','d2','d3'])
    sim = Sim(weight, {'tag':tag, 'actor':actor, 'director':director})
    for index, result in sim.process():
        print index, result
