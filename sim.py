# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from pandas import DataFrame


class Sim(object):

    def __init__(self, samples, weights):
        self.samples = samples
        self.features = samples.columns
        self.ids = samples.index
        self.weights_feature = weights
        self.sample_size = samples.index.size
        self.similarity_samples = DataFrame(np.identity(samples.index.size), index=samples.index, columns=samples.index)
        self.handlers = {}
        self.format_handlers = {}
        self.output_num = 20
        self._init_feature_handlers()
        self._init_input_format()

    def _init_input_format(self):

        self.format_handlers['year'] = lambda x, y: [int(x), int(y)]
        self.format_handlers['tag'] = lambda x, y: [eval(x), eval(y)]
        self.format_handlers['writer'] = lambda x, y: [eval(x), eval(y)]
        self.format_handlers['director'] = lambda x, y: [eval(x), eval(y)]
        self.format_handlers['country'] = lambda x, y: [eval(x), eval(y)]
        self.format_handlers['episodes'] = lambda x, y: [int(x), int(y)]
        self.format_handlers['actor'] = lambda x, y: [eval(x), eval(y)]
        self.format_handlers['language'] = lambda x, y: [x, y]
        self.format_handlers['duration'] = lambda x, y: [int(x), int(y)]

    def _init_feature_handlers(self):

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
            record = self.samples.T[base_index]
            for cur_index in self.ids:
                if sim_matrix[cur_index][base_index] or sim_matrix[base_index][cur_index]:
                    continue
                try:
                    cur_record = self.samples.T[cur_index]
                    listn = []
                    for feature in self.features:
                        [base, cur] = self.format_handlers[feature](record[feature], cur_record[feature])
                        similarity = self.handlers[feature](base, cur)
                        listn.append(similarity)
                    # TODO
                    similar = norm(np.array(listn) * self.weights_feature)
                    sim_matrix[cur_index][base_index] = similar
                    sim_matrix[base_index][cur_index] = similar
                except NameError, e:
                    print e
                    continue
        for index in self.ids:
            format_result = self._calculate_output(index, sim_matrix)
            print index, format_result

    def _calculate_output(self, cover_id, similar_frame):
        sorted_result = similar_frame.sort_values(by=cover_id, ascending=False)[cover_id]
        result = {}
        for index in range(len(sorted_result)):
            if sorted_result.index[index] == cover_id:
                continue
            if index == self.output_num + 1:
                break
            result[sorted_result.index[index]] = sorted_result[index]
        format_result = {'Results': result, 'V': '1.0.0'}
        return format_result

    def _cosine_similarity(self, vector1, vector2):
        similarity = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2) + 0.00001)
        return similarity

    def _weight_func(self, index):
        if index <= 0:
            raise ValueError
        return np.exp(1 - np.sqrt(index))

    def _tune_similarity_list(self, base, cur):
        weights_cur = []
        for j in range(len(base)):
            tmp_weight = self._weight_func(cur.index(base[j]) + 1) if base[j] in cur else 0
            weights_cur.append(tmp_weight)
        weights_base = self._weight_func(range(1, 1 + len(base)))
        return self._cosine_similarity(np.array(weights_cur), weights_base)

    def calculate_year_similarity(self, year_base, year_cur):
        return np.exp((-np.abs(year_base - year_cur)**2 / 32))

    def calculate_tag_similarity(self, tags_base, tags_cur):
        return self._tune_similarity_list(tags_base, tags_cur)

    def calculate_writer_similarity(self, writer_base, writer_cur):
        return self._tune_similarity_list(writer_base, writer_cur)

    def calculate_director_similarity(self, director_base, director_cur):
        return self._tune_similarity_list(director_base, director_cur)

    def calculate_country_similarity(self, country_base, country_cur):
        return self._tune_similarity_list(country_base, country_cur)

    def calculate_episodes_similarity(self, episodes_base, episodes_cur):
        if episodes_base < episodes_cur:
            episodes_base, episodes_cur = episodes_cur, episodes_base
        similarity = float(episodes_cur) / episodes_cur
        return similarity

    def calculate_actor_similarity(self, actor_base, actor_cur):
        return self._tune_similarity_list(actor_base, actor_cur)

    def calculate_language_similarity(self, language_base, language_cur):
        similarity = 1 if language_base == language_cur else 0
        return similarity

    def calculate_duration_similarity(self, duration_base, duration_cur):
        if duration_base < duration_cur:
            duration_base, duration_cur = duration_cur, duration_base
        similarity = duration_cur / duration_base * (1 - np.exp(-duration_cur / 8))
        return similarity


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


class Educatiion_Sim(Sim):
    pass


class Doc_Sim(Sim):
    pass


class Entertainment_Sim(Sim):
    pass


class Movie_Sim(Sim):
    pass


class Sports_Sim(Sim):
    pass


class TV_Sim(Sim):
    pass


class Variety_Sim(Sim):
    pass
