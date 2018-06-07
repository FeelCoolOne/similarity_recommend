# -*- coding: utf-8 -*-

import cPickle as pickle

data_exposured = {}
data_click = {}
num_exposured = 0
num_click = 0
for line in open('../data/history.dat'):
    if len(line) < 80:
        continue
    [iids, click_id, p_log_date] = line.split('\t')
    if len(iids) < 10:
        continue
    for i in iids.strip()[1:-1].strip().split(' ; '):
        data_exposured[i] = data_exposured.get(i, 0) + 1
        num_exposured += 1
    if click_id != 'NULL':
        data_click[click_id] = data_click.get(click_id, 0) + 1
        num_click += 1
print 'exposured num {0}*10, click num : {1}'.format(num_exposured, num_click)

result = {}
for key, value in data_exposured.items():
    result[key] = data_click.get(key, 0) / value

print 'all num of id is {0}'.format(len(result))
with open('../data/fit.dat', 'wb') as f:
    pickle.dump(result, f)

print 'Finished'
# print result
