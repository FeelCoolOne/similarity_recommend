#!/usr/bin/lst/env python
import numpy as np
from numpy.linalg import norm
from pandas import DataFrame
from abc import ABCMeta, abstractmethod
import sys

weights_all = {}
weights_all['cartoorn'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['doc'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['education'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['entertainment'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['movie'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['sports'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['tv'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])
weights_all['variety'] = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6, 0.3])


models = ['cartoon', 'doc', 'education', 'entertainment', 'movie', 'sports', 'tv', 'variety']
features = ['id', 'model', 'year', 'tag', 'writer', 'director', 'country', 'episodes', 'actor', 'language', 'duration']
data_all = {}
ids_all = {}
for model in models:
    data_all[model] = []
    ids_all[model] = []
for line in sys.stdin:
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


def cosine_similarity(vector1, vector2):
    similarity = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2) + 0.00001)
    return similarity


def weight_func(index):
    if index <= 0:
        raise ValueError
    return np.exp(1 - np.sqrt(index))


def tune_similarity_list(base, cur):
    weights_cur = []
    for j in range(len(base)):
        tmp_weight = weight_func(cur.index(base[j]) + 1) if base[j] in cur else 0
        weights_cur.append(tmp_weight)
    weights_base = weight_func(range(1, 1 + len(base)))
    return cosine_similarity(np.array(weights_cur), weights_base)


def calculate_tag_similarity(tags_base, tags_cur):
    return tune_similarity_list(tags_base, tags_cur)


def calculate_writer_similarity(writer_base, writer_cur):
    return tune_similarity_list(writer_base, writer_cur)


def calculate_actor_similarity(actor_base, actor_cur):
    return tune_similarity_list(actor_base, actor_cur)


def calculate_country_similarity(country_base, country_cur):
    return tune_similarity_list(country_base, country_cur)


def calculate_director_similarity(director_base, director_cur):
    return tune_similarity_list(director_base, director_cur)


def calculate_year_similarity(year_base, year_cur):
    return np.exp((-np.abs(year_base - year_cur)**2 / 32))


def calculate_language_similarity(language_base, language_cur):
    similarity = 1 if language_base == language_cur else 0
    return similarity


def calculate_episodes_similarity(episodes_base, episodes_cur):
    if episodes_base < episodes_cur:
        episodes_base, episodes_cur = episodes_cur, episodes_base
    similarity = float(episodes_cur) / episodes_cur
    return similarity


def calculate_duration_similarity(duration_base, duration_cur):
    if duration_base < duration_cur:
        duration_base, duration_cur = duration_cur, duration_base
    similarity = duration_cur / duration_base * (1 - np.exp(-duration_cur / 8))
    return similarity


def calculate_output(cover_id, similar_frame):
    sorted_result = similar_frame.sort_index(by=cover_id, ascending=False)[cover_id]
    result = {}
    for index in range(len(sorted_result)):
        if sorted_result.index[index] == cover_id:
            continue
        if index == 21:
            break
        result[sorted_result.index[index]] = sorted_result[index]
    format_result = {'Results': result, 'V': '1.0.0'}
    return format_result


def format_input(base, cur, feature):
    handler = {}
    handler['year'] = [int(base), int(cur)]
    handler['tag'] = [eval(base), eval(cur)]
    handler['writer'] = [eval(base), eval(cur)]
    handler['director'] = [eval(base), eval(cur)]
    handler['country'] = [eval(base), eval(cur)]
    handler['episodes'] = [int(base), int(cur)]
    handler['actor'] = [eval(base), eval(cur)]
    handler['language'] = [base, cur]
    handler['duration'] = [int(base), int(cur)]
    return handler[feature]


class Sim(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, samples, weights):
        self.model = model
        self.samples = samples
        self.features = samples.columns
        self.ids = samples.index
        self.weights_feature = weights
        self.sample_size = samples.index.size
        self.similarity_samples = DataFrame(np.identity(samples.index.size), index=samples.index, columns=samples.index)
        self.handlers = {}
        self.output_num = 20
        self.init_handlers()

    def format_input(self, base, cur, feature):
        handler = {}
        handler['year'] = [int(base), int(cur)]
        handler['tag'] = [eval(base), eval(cur)]
        handler['writer'] = [eval(base), eval(cur)]
        handler['director'] = [eval(base), eval(cur)]
        handler['country'] = [eval(base), eval(cur)]
        handler['episodes'] = [int(base), int(cur)]
        handler['actor'] = [eval(base), eval(cur)]
        handler['language'] = [base, cur]
        handler['duration'] = [int(base), int(cur)]
        return handler[feature]

    def init_handlers(self):
        self.handlers['year'] = self.calculate_year_similarity
        self.handlers['tag'] = self.calculate_tag_similarity
        self.handlers['writer'] = self.calculate_writer_similarity
        self.handlers['director'] = self.calculate_director_similarity
        self.handlers['country'] = self.calculate_country_similarity
        self.handlers['episodes'] = self.calculate_episodes_similarity
        self.handlers['actor'] = self.calculate_actor_similarity
        self.handlers['language'] = self.calculate_language_similarity
        self.handlers['duration'] = self.calculate_duration_similarity

    def process(self):
        sim_matrix = self.similarity_samples
        for base_index in self.ids:
            record = self.samples[base_index]
            for cur_index in self.ids:
                try:
                    if sim_matrix[cur_index][base_index] or sim_matrix[base_index][cur_index]:
                        continue
                    cur_record = self.samples[cur_index]
                    listn = []
                    for feature, handler in self.handlers.items():
                        [base, cur] = self.format_input(record[feature], cur_record[feature], feature)
                        similarity = handlers[feature](base, cur)
                        listn.append(similarity)
                    # TODO
                    similar = norm(np.array(listn) * self.weights_feature)
                    sim_matrix[self.ids[cur_index]][self.ids[base_index]] = similar
                    sim_matrix[self.ids[base_index]][self.ids[cur_index]] = similar
                except:
                    continue
        for index in self.ids:
            format_result = self.calculate_output(index, sim_matrix)
            print cartoon_id, format_result

    def calculate_output(self, cover_id, similar_frame):
        sorted_result = similar_frame.sort_index(by=cover_id, ascending=False)[cover_id]
        result = {}
        for index in range(len(sorted_result)):
            if sorted_result.index[index] == cover_id:
                continue
            if index == self.output_num + 1:
                break
            result[sorted_result.index[index]] = sorted_result[index]
        format_result = {'Results': result, 'V': '1.0.0'}
        return format_result

    def cosine_similarity(self, vector1, vector2):
        similarity = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2) + 0.00001)
        return similarity

    def weight_func(self, index):
        if index <= 0:
            raise ValueError
        return np.exp(1 - np.sqrt(index))

    def tune_similarity_list(self, base, cur):
        weights_cur = []
        for j in range(len(base)):
            tmp_weight = weight_func(cur.index(base[j]) + 1) if base[j] in cur else 0
            weights_cur.append(tmp_weight)
        weights_base = weight_func(range(1, 1 + len(base)))
        return cosine_similarity(np.array(weights_cur), weights_base)

    @abstractmethod
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    @abstractmethod
    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    @abstractmethod
    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    @abstractmethod
    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Cartoon_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass




'''
计算卡通节目之间的相似度
'''
model = 'cartoon'
samples = data_all[model]
sample_size = samples.index.size
features_weight = weights_all['cartoon']
features = samples.columns
similar_cartoon = DataFrame(np.identity(sample_size), index=samples.index, columns=samples.index)
handlers = {}
handlers['year'] = calculate_year_similarity
handlers['tag'] = calculate_tag_similarity
handlers['writer'] = calculate_writer_similarity
handlers['director'] = calculate_director_similarity
handlers['country'] = calculate_country_similarity
handlers['episodes'] = calculate_episodes_similarity
handlers['actor'] = calculate_actor_similarity
handlers['language'] = calculate_language_similarity
handlers['duration'] = calculate_duration_similarity

for base_index in samples.index:
    record = samples[base_index]

# 对于每个卡通节目去计算其它卡通节目跟它3之间的相似度
    for cur_index in samples.index:
        try:
            if similar_cartoon[cur_index][base_index] or similar_cartoon[base_index][cur_index]:
                continue
            cur_record = samples[cur_index]
            listn = []
            for feature, handler in handlers.items():
                [base, cur] = format_input(record[feature], cur_record[feature], feature)
                similarity = handlers[feature](base, cur)
                listn.append(similarity)

            # TODO
            similar = norm(np.array(listn) * features_weight)
            similar_cartoon[samples.index[cur_index]][samples.index[base_index]] = similar
            similar_cartoon[samples.index[base_index]][samples.index[cur_index]] = similar
        except:
            continue
for cartoon_id in samples.index:
    format_result = calculate_output(cartoon_id, similar_cartoon)
    print cartoon_id, format_result


class Doc_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Educatiion_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Entertainment_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Movie_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Sports_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class TV_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass


class Variety_Sim(Sim):
    def calculate_year_similarity(self, year_base, year_cur):
        pass

    def calculate_tag_similarity(self, tags_base, tags_cur):
        pass

    def calculate_writer_similarity(self, writer_base, writer_cur):
        pass

    def calculate_director_similarity(self, writer_base, writer_cur):
        pass

    def calculate_country_similarity(self, writer_base, writer_cur):
        pass

    def calculate_episodes_similarity(self, writer_base, writer_cur):
        pass

    def calculate_actor_similarity(self, writer_base, writer_cur):
        pass

    def calculate_language_similarity(self, writer_base, writer_cur):
        pass

    def calculate_duration_similarity(self, writer_base, writer_cur):
        pass
