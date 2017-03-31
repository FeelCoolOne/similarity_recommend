# encoding:utf-8
"""
===================================
Test for calculation of similarity
===================================

use charactor  'tag', 'director', 'country',
    'actor', 'language', 'year', 'score'

Created by:
    yonggang Huang
In:
    03-31-2017
"""

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
    douban_test = {'a': ['f', 'b', 'c', 'm'],
                   'b': ['f', 'b', 'c', 'm'],
                   'c': ['f', 'b', 'c', 'm'],
                   'd': ['f', 'b', 'c', 'm'],
                   'u': ['f', 'b', 'c', 'm'], }
    sim = Sim({'tag': tag, 'actor': actor, 'director': director, 'country': country, 'year': year, 'language': language, 'score': score}, weight)
    '''
    for index, result in sim.process():
        print index, result
    '''
    print sim.weight_search(douban_test, verbose=True)
