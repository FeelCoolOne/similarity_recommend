# encoding:utf-8

from pandas import DataFrame, Series
from numpy.random import rand
import sys
sys.path.append('..')
from main.sim import Sim


if __name__ == "__main__":

    weight = {'tag': 0.6, 'actor': 0.6, 'director': 0.8, 'country': 0.5, 'year': 0.3, 'language': 0.2, 'score': 0.2}
    tag = DataFrame(rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['t1', 't2', 't3'])
    actor = DataFrame(rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['a1', 'a2', 'a3'])
    director = DataFrame(rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    country = DataFrame(rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    language = DataFrame(rand(4, 3), index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3'])
    score = Series(rand(4), index=['a', 'b', 'c', 'd'])
    year = Series(rand(4), index=['a', 'b', 'c', 'd'])
    douban_test = {'a1': ['a3', 'a', 'a1', 'a2'],
                   'a2': ['a3', 'a', 'a1', 'a2'],
                   'a3': ['a3', 'a', 'a1', 'a2'],
                   'a4': ['a3', 'a', 'a1', 'a2'],
                   'a5': ['a3', 'a', 'a1', 'a2'],
                   'a6': ['a1', 'a2', 'a4', 'a3']}
    # work mode
    sim = Sim(weight=[.3, .4, .3, .6, .4], index=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], feat_properties=[3, 4, 5, 6, 4])
    sim.fit(rand(6, 22) * 10)
    for index, result in sim.transform():
        print index, result

    # weight search mode
    sim = Sim(index=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], feat_properties=[3, 4, 5, 6, 4], std_output=douban_test)
    sim.fit(rand(6, 22) * 10)
    print sim.weight, sim.score
