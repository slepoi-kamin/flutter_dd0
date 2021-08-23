# -*- coding: utf-8 -*-
"""
Created on 09.02.2020.

@author: ulyanovas
"""

import numpy as np
import copy


class DD0:
    def __init__(self, file):
        self.dd0 = self.__file_read(file)
        self.__aggregates()
        self.__fill_data()
        self.__mirrow_aggregates()
        self.__calc_coords()
        self.__calc_sections()
        self.__calc_weights()

    def get_mic(self, point_coords: list):
        return self.__calc_global_mic(point_coords)

    def __file_read(self, fname):
        with open(fname, 'r') as tfile:
            text = tfile.read()
        dd0_list = text.split('!')
        dd0_list.pop(0)
        return dd0_list

    def __aggregates(self):
        self.aggregates = {}
        for agr in self.dd0:
            name = find_next(agr, 'short')
            self.aggregates[name] = {
                'name': name,
                'other': agr,
                'n_point_weights': int(find_next(agr, 'число_грузов')),
                'n_sections': int(find_next(agr, 'число_сечений')),
                'base_name': find_next(agr, 'имя_балки-основания'),
                'n_start_point': int(
                    find_next(agr, 'номер_точки_крепления')),
                'main_axis_len': float(find_next(agr, 'dlxi')),
                'angle_xi': float(find_next(agr, 'XIe')),
            }

    def __fill_data(self):
        for aggreggate in self.aggregates:
            agr = self.aggregates[aggreggate]
            if agr['n_point_weights'] > 0:
                weights = agr['other'].split('данные_гр')
                text_data = weights.pop(0)
            else:
                weights = None
                text_data = agr['other']

            agr['data'] = data_to_dict(text_data)

            if weights:

                agr['weights'] = {}
                for weight in weights:
                    name = find_next(weight, 'узa:')
                    agr['weights'][name] = {
                        'name': name,
                        'data': data_to_dict(weight),
                        'n_start_point': weight_start_point(weight),
                    }

    def __calc_coords(self):
        self.__calc_local_coords()
        self.__calc_global_coords()
        self.__create_coord_links()

    def __calc_local_coords(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            delta = float(agr['main_axis_len'] / (agr['n_sections'] - 1))
            # agr['data']['coords'] = {}
            local = [[0.0, 0.0, n * delta] for n in range(agr['n_sections'])]
            agr['data']['coords']['local'] = np.array(local)

    def __calc_global_coords(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            coords: np.ndarray = agr['data']['coords']
            _local: np.ndarray = coords['local']
            _cos: np.ndarray = coords['local_to_global']
            _start: np.ndarray = coords['start_global']
            coords['global'] = self.__transform_local_to_global(
                _local,
                _cos,
                start=_start
            )

    def __create_coord_links(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            agr['coords'] = agr['data']['coords']
            if 'weights' in agr.keys():
                for weight_name in agr['weights']:
                    weight = agr['weights'][weight_name]
                    weight['coords'] = weight['data']['coords']
                    weight['data']['coords']['cm_global'] = \
                        weight['data']['coords']['start_global']

    def __calc_sections(self):
        self.__add_sections()
        self.__calc_sectons_local_coords()
        self.__calc_sectons_global_coords()
        self.__calc_sections_mass()
        self.__calc_sections_cm()
        self.__calc_inertia()

    def __calc_sectons_local_coords(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            nodes_local: np.ndarray = agr['coords']['local']
            sec_local = []
            for i in range(len(nodes_local)-1):
                sec_local.append((nodes_local[i]+nodes_local[i+1])/2)
            agr['sections']['coords']['local'] = np.array(sec_local)

    def __add_sections(self):
        for agr_name in self.aggregates:
            _agr = self.aggregates[agr_name]
            # noinspection PyTypeChecker
            _agr['sections'] = dict(
                coords=dict(
                    local_to_global=_agr['coords']['local_to_global'],
                    start_global=_agr['coords']['start_global'],
                ),
                data={},
            )

    def __calc_sectons_global_coords(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            _local: np.ndarray = agr['sections']['coords']['local']
            _ltg: np.ndarray = agr['sections']['coords']['local_to_global']
            _start: np.ndarray = agr['sections']['coords']['start_global']
            _global = self.__transform_local_to_global(_local,
                                                       _ltg,
                                                       start=_start)
            agr['sections']['coords']['global'] = _global

    @staticmethod
    def __transform_local_to_global(local_coords, ltg_matrix, start=None):
        glob = []
        if start is None:
            start = np.array([0.0, 0.0, 0.0])
        for coords in local_coords:
            glob.append(ltg_matrix @ coords + start)
        return np.array(glob).reshape((-1,3))

    def __calc_sections_mass(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            section_length = agr['main_axis_len']/ (agr['n_sections'] - 1)
            g_: float = 9.80665
            bulk_weihts = agr['data']['m']
            masses = []
            for i in range(len(bulk_weihts) - 1):
                masses.append((bulk_weihts[i] + bulk_weihts[
                    i + 1]) * g_ * section_length / 2)
            agr['sections']['data']['mass'] = np.array(masses)

    def __calc_sections_cm(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            self.__calc_local_cm(agr)
            self.__calc_global_cm(agr)

    def __calc_inertia(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            self.__calc_i_local(agr)
            self.__calc_i330_local(agr)
            self.__calc_i330_global(agr)

    def __calc_i_local(self, aggregate):
        section_length = aggregate['main_axis_len'] / (aggregate['n_sections'] - 1)
        g_: float = 9.80665
        bulk_imoment = aggregate['data']['J33']
        imoment = []
        for i in range(len(bulk_imoment) - 1):
            imoment.append((bulk_imoment[i] + bulk_imoment[
                i + 1]) * g_ * section_length / 2)
        aggregate['sections']['data']['full_moment_local'] = np.array(imoment)

    def __calc_local_cm(self, aggregate):
        locals_: np.ndarray = aggregate['sections']['coords']['local']
        d_x1_nodes = aggregate['data']['s1']
        d_x2_nodes = aggregate['data']['s2']
        d_x1, d_x2 = [], []
        for i in range(len(d_x1_nodes) - 1):
            d_x1.append((d_x1_nodes[i] + d_x1_nodes[i + 1]) / 2)
            d_x2.append((d_x2_nodes[i] + d_x2_nodes[i + 1]) / 2)
        locals_cm = locals_.copy()
        locals_cm[:, 0] = np.array(d_x1)
        locals_cm[:, 1] = np.array(d_x2)
        aggregate['sections']['coords']['cm_local'] = locals_cm

    def __calc_global_cm(self, aggregate):
        locals_cm = aggregate['sections']['coords']['cm_local']
        ltg = aggregate['sections']['coords']['local_to_global']
        start = aggregate['sections']['coords']['start_global']
        globals_cm = self.__transform_local_to_global(locals_cm, ltg, start)
        aggregate['sections']['coords']['cm_global'] = globals_cm

    def __calc_i330_local(self, aggregate):
        i33_full = aggregate['sections']['data']['full_moment_local']
        mass = aggregate['sections']['data']['mass']
        locals_ = aggregate['sections']['coords']['local']
        cm_locals = aggregate['sections']['coords']['cm_local']
        sigmas = np.array([self.__calc_sigma(c1, c2) for c1, c2 in zip(locals_, cm_locals)])
        i330 = i33_full - mass * sigmas**2
        aggregate['sections']['data']['i330_local'] = i330

    @staticmethod
    def __calc_sigma(coord1:np.ndarray, coord2:np.ndarray):
        c1 = coord1.reshape(-1)
        c2 = coord2.reshape(-1)
        return ((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2 + (c2[2]-c1[2])**2)**0.5

    def __calc_i330_global(self, aggregate):
        ltg = aggregate['sections']['coords']['local_to_global']
        i330_loc = aggregate['sections']['data']['i330_local']
        i_tensors_global = []
        for i33 in i330_loc:
            i_tensor = np.zeros((3,3))
            i_tensor[2,2] = i33
            i_tensor_global = ltg.T @ i_tensor @ ltg
            i_tensors_global.append(i_tensor_global)
        aggregate['sections']['data']['i0_tensor_global'] = np.array(
            i_tensors_global)

    def __calc_weights(self):
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            if 'weights' in agr.keys():
                for weight_name in agr['weights']:
                    weight = agr['weights'][weight_name]
                    self.__calc_weight_mass(weight)
                    self.__calc_weight_inertia(weight)

    def __calc_weight_mass(self, weight):
        mass = weight['data']['M']
        g_: float = 9.80665
        if len(mass) == 1:
            weight['data']['mass'] = mass[0] * g_
        else:
            raise TypeError(f'Legth of var {mass.__name__} shoul be equal '
                            f'to 1 in function {__name__}')

    def __calc_weight_inertia(self, weight):
        self.calc_i0_tensor_local(weight)
        self.calc_i0_tensor_global(weight)

    def calc_i0_tensor_local(self, weight):
        tensor = np.zeros((3,3))
        g_: float = 9.80665
        tensor[0, 0] = weight['data']['I1'] * g_
        tensor[1, 1] = weight['data']['I2'] * g_
        tensor[2, 2] = weight['data']['I3'] * g_
        weight['data']['i0_tensor_local'] = tensor

    def calc_i0_tensor_global(self, weight):
        ltg = weight['coords']['local_to_global']
        tensor_local = weight['data']['i0_tensor_local']
        tensor_global = ltg.T @ tensor_local @ ltg
        weight['data']['i0_tensor_global'] = tensor_global

    @staticmethod
    def __calc_j(old_start: np.ndarray, new_start: np.ndarray):
        # new in old
        nio = new_start - old_start
        a_, b_, c_ = nio[0], nio[1], nio[2]
        j_ = np.array([[b_ ** 2 + c_ ** 2, -a_ * b_, -a_ * c_],
                      [-a_ * b_, a_ ** 2 + c_ ** 2, -b_ * c_],
                      [-a_ * c_, -b_ * c_, b_ ** 2 + a_ ** 2]])
        return j_

    def __calc_global_mic(self, point):
        mic = {}
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            zeros_point = np.array(point)
            mic.update(self.__calc_aggregate_global_mich(agr, zeros_point))
        mic = self.__calc_total_mic(mic)
        return mic

    def __calc_aggregate_global_mich(self, aggregate, zeros_point):
        i0_tens_glob = aggregate['sections']['data']['i0_tensor_global']
        m_ = aggregate['sections']['data']['mass']
        cm_coords = aggregate['sections']['coords']['cm_global']

        # aggregate characteristics
        i_agr = self.__calc_itenzor_parallel_trans(cm_coords,
                                                   i0_tens_glob,
                                                   m_, zeros_point)
        cm_agr, mass_agr = self.__calc_cm(m_, cm_coords)

        # weights characteristics
        weights_idat = self.__get_weights_inertia_data(aggregate)
        cm_weights, mass_weights = self.__calc_cm(weights_idat[2],
                                                  weights_idat[0])
        i_weights = self.__calc_itenzor_parallel_trans(*weights_idat,
                                                       zeros_point)

        # total characteristics
        mt = np.array([mass_agr, mass_weights])
        cmt = np.array([cm_agr, cm_weights])

        cm_total, mass_total = self.__calc_cm(mt, cmt)
        itotal = i_agr + i_weights

        ret = {
            aggregate['name']: {
                'aggregate_mic': {
                    'mass': mass_agr,
                    'cm': cm_agr,
                    'inertia_tensor': i_agr,
                },
                'weights_mic': {
                    'mass': mass_weights,
                    'cm': cm_weights,
                    'inertia_tensor': i_weights,
                },
                'total_mic': {
                    'mass': mass_total,
                    'cm': cm_total,
                    'inertia_tensor': itotal,
                },
            },
        }
        return ret

    @staticmethod
    def __calc_cm(masses: np.ndarray, cm_coords: np.ndarray):
        summ_mass = masses.sum()

        moments = []
        for i in range(len(masses)):
            moments.append(cm_coords[i] * masses[i])
        moments_xyz = np.array(moments).sum(0)
        if summ_mass == 0:
            cm_xyz = np.array([0.0, 0.0, 0.0])
        else:
            cm_xyz = moments_xyz / summ_mass
        return cm_xyz, summ_mass

    def __calc_itenzor_parallel_trans(self, cm_coords: np.ndarray,
                                      i0_tens_glob: np.ndarray,
                                      mass: np.ndarray,
                                      zeros_point: np.ndarray):
        j_ = []
        for cm_coord in cm_coords:
            j_.append(self.__calc_j(cm_coord, zeros_point))
        j_ = np.array(j_)
        ifull_tensors = []
        for i in range(len(i0_tens_glob)):
            ifull_tensors.append(i0_tens_glob[i] + j_[i] * mass[i])
        sum_ = np.array(ifull_tensors).sum(0)
        return sum_

    def __get_weights_inertia_data(self, aggregate):
        if 'weights' in aggregate.keys():
            cm_coord, i0_tensor, mass = [], [], []
            for weight_name in aggregate['weights']:
                weight = aggregate['weights'][weight_name]
                cm_coord.append(weight['coords']['cm_global'])
                i0_tensor.append(weight['data']['i0_tensor_global'])
                mass.append(weight['data']['mass'])
            cm_coord = np.array(cm_coord)
            i0_tensor = np.array(i0_tensor)
            mass = np.array(mass)
        else:
            cm_coord = np.zeros((1,3))
            i0_tensor = np.zeros((1,3,3))
            mass = np.zeros((1))
        return cm_coord, i0_tensor, mass

    def __calc_total_mic(self, mic):
        mass, itensor, cm = [], [], []
        for agr_name in mic:
            tm = mic[agr_name]['total_mic']
            mass.append(tm['mass'])
            cm.append(tm['cm'])
            itensor.append(tm['inertia_tensor'])
        mass, itensor, cm = np.array(mass), np.array(itensor), np.array(cm)
        itensor_total = np.array(itensor).sum(0)
        cm_total, m_total = self.__calc_cm(mass, cm)

        total = {
            'TOTAL': {
                'mass': m_total,
                'cm': cm_total,
                'inertia_tensor': itensor_total,
            },
        }

        total.update(mic)
        return total

    def __mirrow_aggregates(self):
        new_agrs = {}
        for agr_name in self.aggregates:
            agr = self.aggregates[agr_name]
            if self.__is_agr_mirrored(agr):
                new_agr = self.__start_mirror(agr)
                new_agrs[new_agr['name']] = new_agr
            elif 'weights' in agr.keys():
                new_weights = {}
                for weight_name in agr['weights']:
                    weight = agr['weights'][weight_name]
                    if weight['data']['coords']['start_global'][1] != 0.0:
                        new_weight = self.__start_mirror(weight)
                        new_weights[new_weight['name']] = new_weight
                agr['weights'].update(new_weights)
        self.aggregates.update(new_agrs)

    @staticmethod
    def __is_agr_mirrored(agr):
        ltg = agr['data']['coords']['local_to_global']
        l_ = agr['main_axis_len']
        coord = np.array([0.0, 0.0, l_])
        coord_ = ltg @ coord
        if coord_[1] == 0.0:
            return False
        return True

    def __start_mirror(self, agr):
        ltg_trans = np.array([[-1, 1, 1], [1, -1, -1], [1, 1, 1]])
        c_trans = np.array([1, -1, 1])
        new_agr = copy.deepcopy(agr)
        ltg = agr['data']['coords']['local_to_global']
        start = agr['data']['coords']['start_global']
        new_agr['data']['coords']['local_to_global'] = ltg_trans * ltg
        new_agr['data']['coords']['start_global'] = c_trans * start
        new_agr['name'] = agr['name'] + '_L'
        if 'weights' in agr.keys():
            new_weights ={}
            for weight_name in agr['weights']:
                weight = agr['weights'][weight_name]
                new_weight = self.__start_mirror(weight)
                new_weights[new_weight['name']] = new_weight
            new_agr['weights'] = new_weights
        return new_agr


def weight_start_point(text_data: str):
    text_data = text_data.replace('\n', '')
    return find_next(text_data, '__M_')


def data_to_dict(text_data: str):
    tables_data = text_data.split('a  _')
    tables_data.pop(0)
    data_dict = {}
    for table in tables_data:
        data_dict.update(create_data_tables(table))
    return data_dict


def create_data_tables(text_data):
    data_dict = {}

    if 'в_глобальной' in text_data:
        global_text = text_data[text_data.index('в_глобальной'):].strip()
        text_data = text_data[:text_data.index('в_глобальной')].strip()
        data_dict['coords'] = get_global_matrix(global_text)
    if 'g=\n' in text_data:
        text_data = text_data[:text_data.index('g=\n')].strip()
    list_data = text_data.split('\n')
    list_data = [x for x in list_data if x != '' and x != ' ']
    str_names = list_data.pop(0)
    col_names = data_from_str(str_names, start=1, format='str')
    data = np.array([data_from_str(x,start=1) for x in list_data]).T

    for i in range(len(col_names)):
        data_dict[col_names[i]] = data[i]
    return data_dict


def get_global_matrix(text: str):
    listed_text = text.split('\n')
    listed_text = [x for x in listed_text if x != '' and x != ' ']
    listed_text = listed_text[:4]
    data = [data_from_str(x, start=1) for x in listed_text]
    cos = np.array([x[0:3] for x in data[0:3]])
    # start_global = np.array([x[3:] for x in data[0:3]]).T
    start_global = np.array([x[3:] for x in data[0:3]]).reshape(-1)
    return {'local_to_global': cos, 'start_global': start_global}


def data_from_str(string_data: str, start=None, end=None, format=None):
    string_data = string_data.replace('_', '')
    list_data = string_data.split(' ')
    # if not format:
    #     list_data = [float(x) for x in list_data if x != '' and x != '_']
    # else:
    #     list_data = [x for x in list_data if x != '' and x != '_']

    list_data = [x for x in list_data if x != '' and x != '_']

    if start and end:
        returned_data = list_data[start: end]
    elif start:
        returned_data = list_data[start:]
    elif end:
        returned_data = list_data[:end]
    else:
        returned_data = list_data

    if not format:
        returned_data = [float(x) for x in returned_data]

    return returned_data


def find_next(init_str: str, search_str: str):
    start_0 = init_str.index(search_str)
    for i in range(len(init_str)):
        n_dig = i+start_0
        dig = init_str[n_dig]
        if dig == ' ' or dig == '=' or dig == ':':
            start_1 = n_dig+1
            break
    for i in range(len(init_str)):
        n_dig = i + start_1
        if not init_str[n_dig].isspace():
            start_2 = n_dig
            break
    for i in range(len(init_str)):
        n_dig = i + start_2
        if init_str[n_dig].isspace():
            start_3 = n_dig
            break
    return init_str[start_2: start_3]


if __name__ == '__main__':
    a = DD0(r'D:\Work\Python\Flutter_dd0\n8r112B.dd0')
    mic = a.get_mic([0.88334, 0.0, 8.11934])
    print(0)

