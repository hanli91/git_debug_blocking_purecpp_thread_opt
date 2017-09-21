from collections import namedtuple
import heapq as hq
import logging
import numpy
from operator import attrgetter
import pandas as pd
import time
from array import array
import os

import py_entitymatching as mg
import py_entitymatching.catalog.catalog_manager as cm

from debugblocker_cython import debugblocker_cython

logger = logging.getLogger(__name__)

SELECTED_FIELDS_UPPER_BOUND = 8


# Incorporate the reuse algorithm into this version.

def debug_blocker(ltable, rtable, candidate_set, activate_reusing_module, use_new_topk, use_parallel,
                  output_path, output_size=200, attr_corres=None, verbose=True):
    logging.info('\nstart blocking')
    total_start = time.time()
    print 'start time:', total_start
    # preprocessing_start = time.clock()
    # Basic checks.
    if len(ltable) == 0:
        raise StandardError('Error: ltable is empty!')
    if len(rtable) == 0:
        raise StandardError('Error: rtable is empty!')
    if output_size <= 0:
        raise StandardError('The input parameter: \'pred_list_size\''
                            ' is less than or equal to 0. Nothing needs'
                            ' to be done!')

    print 'cand set size:', len(candidate_set)

    # get metadata
    l_key, r_key = cm.get_keys_for_ltable_rtable(ltable, rtable, logger, verbose)

    # validate metadata
    cm._validate_metadata_for_table(ltable, l_key, 'ltable', logger, verbose)
    cm._validate_metadata_for_table(rtable, r_key, 'rtable', logger, verbose)

    # Check the user input field correst list (if exists) and get the raw
    # version of our internal correst list.
    check_input_field_correspondence_list(ltable, rtable, attr_corres)
    corres_list = get_field_correspondence_list(ltable, rtable,
                                                l_key, r_key, attr_corres)

    # Build the (col_name: col_index) dict to speed up locating a field in
    # the schema.
    ltable_col_dict = build_col_name_index_dict(ltable)
    rtable_col_dict = build_col_name_index_dict(rtable)

    # Filter correspondence list to remove numeric types. We only consider
    # string types for document concatenation.
    filter_corres_list(ltable, rtable, l_key, r_key,
                       ltable_col_dict, rtable_col_dict, corres_list)
    print corres_list

    # Get field filtered new table.
    ltable_filtered, rtable_filtered = get_filtered_table(
        ltable, rtable, corres_list)
    print ltable_filtered.columns
    print rtable_filtered.columns

    feature_list = select_features(ltable_filtered, rtable_filtered, l_key, r_key)
    feature_index_list = [feature_list[i][0] for i in range(len(feature_list))]
    print feature_index_list

    if len(feature_list) == 0:
        raise StandardError('\nError: the selected field list is empty,'
                            ' nothing could be done! Please check if all'
                            ' table fields are numeric types.')
    print 'selected_fields:', ltable_filtered.columns[feature_index_list]

    lrecord_id_to_index_map = build_id_to_index_map(ltable_filtered, l_key)
    rrecord_id_to_index_map = build_id_to_index_map(rtable_filtered, r_key)
    print 'finish building id to index map'

    lrecord_index_to_id_map = build_index_to_id_map(ltable_filtered, l_key)
    rrecord_index_to_id_map = build_index_to_id_map(rtable_filtered, r_key)
    print 'finish building index to id map'

    lrecord_list = get_tokenized_table(ltable_filtered, l_key, feature_index_list)
    rrecord_list = get_tokenized_table(rtable_filtered, r_key, feature_index_list)
    print 'finish tokenizing tables'

    order_dict, token_index_dict = build_global_token_order(
        lrecord_list, rrecord_list)
    print 'finish building global order'

    replace_token_with_numeric_index(lrecord_list, order_dict)
    replace_token_with_numeric_index(rrecord_list, order_dict)
    print 'finish replacing tokens with numeric indices'

    sort_record_tokens_by_global_order(lrecord_list)
    sort_record_tokens_by_global_order(rrecord_list)
    print 'finish sorting record tokens'

    lrecord_token_list, lrecord_index_list, lrecord_field_list =\
                            split_record_token_and_index(lrecord_list, len(feature_list))
    rrecord_token_list, rrecord_index_list, rrecord_field_list =\
                            split_record_token_and_index(rrecord_list, len(feature_list))
    print 'finish splitting record token and index'

    del lrecord_list
    del rrecord_list

    new_formatted_candidate_set = index_candidate_set(
        candidate_set, lrecord_id_to_index_map, rrecord_id_to_index_map, verbose)
    print 'finish reformating cand set'

    ltable_field_length_list = calc_table_field_length(lrecord_index_list, len(feature_list))
    rtable_field_length_list = calc_table_field_length(rrecord_index_list, len(feature_list))

    ltable_field_token_sum = calc_table_field_token_sum(ltable_field_length_list, len(feature_list))
    rtable_field_token_sum = calc_table_field_token_sum(rtable_field_length_list, len(feature_list))
    print ltable_field_token_sum
    print rtable_field_token_sum

    ltoken_sum = 0
    rtoken_sum = 0
    for i in range(len(ltable_field_token_sum)):
        ltoken_sum += ltable_field_token_sum[i]
    for i in range(len(rtable_field_token_sum)):
        rtoken_sum += rtable_field_token_sum[i]
    ltoken_ave = ltoken_sum * 1.0 / len(lrecord_token_list)
    rtoken_ave = rtoken_sum * 1.0 / len(rrecord_token_list)
    print ltoken_ave, rtoken_ave

    ltoken_ratio = []
    rtoken_ratio = []
    for i in range(len(ltable_field_token_sum)):
        ltoken_ratio.append(ltable_field_token_sum[i] * 1.0 / ltoken_sum)
    for i in range(len(rtable_field_token_sum)):
        rtoken_ratio.append(rtable_field_token_sum[i] * 1.0 / rtoken_sum)
    print ltoken_ratio
    print rtoken_ratio

    # tsum = 0
    # tlist = []
    # for i in xrange(len(ltable_field_token_sum)):
    #     value = (ltable_field_token_sum[i] + rtable_field_token_sum[i]) \
    #             * 1.0 / (len(lrecord_token_list) + len(rrecord_token_list))
    #     tsum += value
    #     tlist.append(value)
    # for i in xrange(len(tlist)):
    #     tlist[i] /= tsum
    # print tlist

    debugblocker_cython(lrecord_token_list, rrecord_token_list,
                        lrecord_index_list, rrecord_index_list,
                        lrecord_field_list, rrecord_field_list,
                        ltable_field_token_sum, rtable_field_token_sum,
                        new_formatted_candidate_set, len(feature_list), output_size, output_path,
                        activate_reusing_module, use_new_topk, use_parallel)

    total_end = time.time()
    total_time = total_end - total_start
    print 'total time:', total_time

    return total_time


def copy_record_list(record_list, removed_field):
    new_record_list = []
    for i in range(len(record_list)):
        new_record = []
        for j in range(len(record_list[i])):
            if record_list[i][j][1] != removed_field:
                new_record.append(record_list[i][j][1])
        new_record_list.append(new_record)


def output_topk_list(outfile, topk_list, lrecord_index_to_id_map, rrecord_index_to_id_map):
    out = open(outfile, 'w')
    temp_list = list(topk_list)
    hq.heapify(temp_list)
    while len(temp_list) > 0:
        pair = hq.heappop(temp_list)
        out.write(str(pair[0]) + ' ' + str(pair[1]) + ' ' + str(pair[2]) + ' ' +
                  str(lrecord_index_to_id_map[pair[1]]) + ' ' + str(rrecord_index_to_id_map[pair[2]]) + '\n')
    out.close()


def remove_tokens_of_removed_field(record_token_list, record_index_list, removed_field):
    for i in range(len(record_token_list)):
        token_list = record_token_list[i]
        index_list = record_index_list[i]
        for j in reversed(range(len(token_list))):
            if index_list[j] == removed_field:
                token_list.pop(j)
                index_list.pop(j)


def calc_table_field_length(record_index_list, num_field):
    table_field_length_list = []
    for i in range(len(record_index_list)):
        field_array = []
        for j in range(num_field):
            field_array.append(0)
        field_array = array('I', field_array)
        for j in range(len(record_index_list[i])):
            field_array[record_index_list[i][j]] += 1
        table_field_length_list.append(field_array)

    return table_field_length_list


def calc_record_length(ltable_field_length_list, remained_fields):
    record_length_list = []
    for i in range(len(ltable_field_length_list)):
        actual_length = 0
        for field in remained_fields:
            actual_length += ltable_field_length_list[i][field]
        record_length_list.append(actual_length)

    return record_length_list


def calc_table_field_token_sum(table_field_length_list, num_field):
    table_field_token_sum = []
    for i in range(num_field):
        table_field_token_sum.append(0)
    for i in range(len(table_field_length_list)):
        for j in range(len(table_field_length_list[i])):
            table_field_token_sum[j] += table_field_length_list[i][j]

    return table_field_token_sum


def print_topk_heap(topk_heap, lrecord_list, rrecord_list):
    hq.heapify(topk_heap)
    while len(topk_heap) > 0:
        pair = hq.heappop(topk_heap)
        print pair
        # print lrecord_list[pair[1]]
        # print rrecord_list[pair[2]]
        # lrec = lrecord_list[pair[1]]
        # rrec = rrecord_list[pair[2]]
        # ls = ''
        # rs = ''
        # print pair[0]
        # for token in lrec:
        #     if token in rrec:
        #         ls += '1 '
        #     else:
        #         ls += '0 '
        # for token in rrec:
        #     if token in lrec:
        #         rs += '1 '
        #     else:
        #         rs += '0 '
        # print ls
        # print rs
        # print '\n'


def assemble_topk_table(topk_heap, ltable, rtable):
    topk_heap.sort(key=lambda tup: tup[0], reverse=True)
    ret_data_col_name_list = ['similarity']
    ltable_col_names = list(ltable.columns)
    rtable_col_names = list(rtable.columns)

    for i in range(len(ltable_col_names)):
        ret_data_col_name_list.append('ltable.' + ltable_col_names[i])
    for i in range(len(rtable_col_names)):
        ret_data_col_name_list.append('rtable.' + rtable_col_names[i])

    ret_tuple_list = []
    for tuple in topk_heap:
        ret_tuple = [tuple[0]]
        lrecord = ltable.ix[tuple[1]]
        rrecord = rtable.ix[tuple[2]]
        for field in lrecord:
            ret_tuple.append(field)
        for field in rrecord:
            ret_tuple.append(field)
        ret_tuple_list.append(ret_tuple)

    data_frame = pd.DataFrame(ret_tuple_list)
    # When the ret data frame is empty, we cannot assign column names.
    if len(data_frame) == 0:
        return data_frame

    data_frame.columns = ret_data_col_name_list
    return data_frame


def gene_count_list(len_record_list, prefix_match_max_size):
    count_list = []
    for i in range(len_record_list):
        l = [0]
        for j in range(prefix_match_max_size):
            l.append(0)
        count_list.append(l)

    return count_list


def jaccard_sim(l_token_set, r_token_set):
    l_len = len(l_token_set)
    r_len = len(r_token_set)
    intersect_size = len(l_token_set & r_token_set)
    if l_len + r_len == 0:
        return 0.0

    return intersect_size * 1.0 / (l_len + r_len - intersect_size)


def print_jaccard_sim(l_token_list, r_token_list):
    l_len = len(l_token_list)
    r_len = len(r_token_list)
    for i in range(l_len):
        for j in range(r_len):
            if l_token_list[i] == r_token_list[j]:
                print i, j, l_token_list[i], r_token_list[j]


def list_jaccard_sim(l_token_list, r_token_list):
    l_len = len(l_token_list)
    r_len = len(r_token_list)
    sl = ''
    sr = ''
    for i in range(l_len):
        if l_token_list[i] in r_token_list:
            sl += '1 '
        else:
            sl += '0 '
    for i in range(r_len):
        if r_token_list[i] in l_token_list:
            sr += '1 '
        else:
            sr += '0 '
    print sl
    print sr
    print ''


def check_input_field_correspondence_list(ltable, rtable, field_corres_list):
    if field_corres_list is None:
        return
    true_ltable_fields = list(ltable.columns)
    true_rtable_fields = list(rtable.columns)
    for pair in field_corres_list:
        if type(pair) != tuple or len(pair) != 2:
            raise AssertionError('Error in checking user input field'
                                 ' correspondence: pair in not in the'
                                 'tuple format!' % (pair))

    given_ltable_fields = [field[0] for field in field_corres_list]
    given_rtable_fields = [field[1] for field in field_corres_list]
    for given_field in given_ltable_fields:
        if given_field not in true_ltable_fields:
            raise AssertionError('Error in checking user input field'
                                 ' correspondence: the field \'%s\' is'
                                 ' not in the ltable!' % (given_field))
    for given_field in given_rtable_fields:
        if given_field not in true_rtable_fields:
            raise AssertionError('Error in checking user input field'
                                 ' correspondence:'
                                 ' the field \'%s\' is not in the'
                                 ' rtable!' % (given_field))
    return


def get_field_correspondence_list(ltable, rtable, lkey, rkey, attr_corres):
    corres_list = []
    if attr_corres is None or len(attr_corres) == 0:
        corres_list = mg.get_attr_corres(ltable, rtable)['corres']
        if len(corres_list) == 0:
            raise AssertionError('Error: the field correspondence list'
                                 ' is empty. Please specify the field'
                                 ' correspondence!')
    else:
        for tu in attr_corres:
            corres_list.append(tu)

    key_pair = (lkey, rkey)
    if key_pair not in corres_list:
        corres_list.append(key_pair)

    return corres_list


def filter_corres_list(ltable, rtable, ltable_key, rtable_key,
                       ltable_col_dict, rtable_col_dict, corres_list):
    ltable_dtypes = list(ltable.dtypes)
    rtable_dtypes = list(rtable.dtypes)
    for i in reversed(range(len(corres_list))):
        lcol_name = corres_list[i][0]
        rcol_name = corres_list[i][1]
        # Filter the pair where both fields are numeric types.
        if ltable_dtypes[ltable_col_dict[lcol_name]] != numpy.dtype('O') \
                and rtable_dtypes[rtable_col_dict[rcol_name]] != numpy.dtype('O'):
            if lcol_name != ltable_key and rcol_name != rtable_key:
                corres_list.pop(i)

    if len(corres_list) == 0:
        raise StandardError('The field correspondence list is empty after'
                            ' filtering: nothing could be done!')


def get_filtered_table(ltable, rtable, corres_list):
    ltable_cols = [col_pair[0] for col_pair in corres_list]
    rtable_cols = [col_pair[1] for col_pair in corres_list]
    lfiltered_table = ltable[ltable_cols]
    rfiltered_table = rtable[rtable_cols]
    return lfiltered_table, rfiltered_table


def build_col_name_index_dict(table):
    col_dict = {}
    col_names = list(table.columns)
    for i in range(len(col_names)):
        col_dict[col_names[i]] = i
    return col_dict


def select_features(ltable, rtable, lkey, rkey):
    lcolumns = list(ltable.columns)
    rcolumns = list(rtable.columns)
    lkey_index = -1
    rkey_index = -1
    if len(lcolumns) != len(rcolumns):
        raise StandardError('Error: FILTERED ltable and FILTERED rtable'
                            ' have different number of fields!')
    for i in range(len(lcolumns)):
        if lkey == lcolumns[i]:
            lkey_index = i
    if lkey_index < 0:
        raise StandardError('Error: cannot find key in the FILTERED'
                            ' ltable schema!')
    for i in range(len(rcolumns)):
        if rkey == rcolumns[i]:
            rkey_index = i
    if rkey_index < 0:
        raise StandardError('Error: cannot find key in the FILTERED'
                            ' rtable schema!')

    lweight = get_feature_weight(ltable)
    # logging.info('\nFinish calculate ltable feature weights.')
    rweight = get_feature_weight(rtable)
    # logging.info('\nFinish calculate rtable feature weights.')
    if len(lweight) != len(rweight):
        raise StandardError('Error: ltable and rtable don\'t have the'
                            ' same schema')

    Rank = namedtuple('Rank', ['index', 'weight'])
    rank_list = []
    for i in range(len(lweight)):
        rank_list.append(Rank(i, lweight[i] * rweight[i]))
    if lkey_index == rkey_index:
        rank_list.pop(lkey_index)
    else:
        # Make sure we remove the index with larger value first!!!
        if lkey_index > rkey_index:
            rank_list.pop(lkey_index)
            rank_list.pop(rkey_index)
        else:
            rank_list.pop(rkey_index)
            rank_list.pop(lkey_index)

    rank_list = sorted(rank_list, key=attrgetter('weight'), reverse=True)
    rank_index_list = []
    num_selected_fields = 0
    # if len(rank_list) <= 3:
    #     num_selected_fields = len(rank_list)
    # elif len(rank_list) <= 5:
    #     num_selected_fields = 3
    # else:
    #     num_selected_fields = len(rank_list) / 2
    if len(rank_list) < SELECTED_FIELDS_UPPER_BOUND:
        num_selected_fields = len(rank_list)
    else:
        num_selected_fields = SELECTED_FIELDS_UPPER_BOUND

    for i in range(num_selected_fields):
        rank_index_list.append((rank_list[i].index, rank_list[i].weight))
    # return sorted(rank_index_list, key=lambda x: x[0])
    return rank_index_list


def get_feature_weight(table):
    num_records = len(table)
    if num_records == 0:
        raise StandardError('Error: empty table!')
    weight = []
    for col in table.columns:
        value_set = set()
        non_empty_count = 0
        col_values = table[col]
        for value in col_values:
            if not pd.isnull(value):
                value_set.add(value)
                non_empty_count += 1
        selectivity = 0.0
        if non_empty_count != 0:
            selectivity = len(value_set) * 1.0 / non_empty_count
        non_empty_ratio = non_empty_count * 1.0 / num_records
        weight.append(non_empty_ratio + selectivity)
    return weight


def build_id_to_index_map(table, table_key):
    record_id_to_index = {}
    id_col = list(table[table_key])
    for i in range(len(id_col)):
        # id_col[i] = str(id_col[i])
        if id_col[i] in record_id_to_index:
            raise Exception('record_id is already in record_id_to_index')
        record_id_to_index[id_col[i]] = i
    return record_id_to_index


def build_index_to_id_map(table, table_key):
    record_index_to_id_map = {}
    id_col = list(table[table_key])
    for i in range(len(id_col)):
        # id_col[i] = str(id_col[i])
        record_index_to_id_map[i] = id_col[i]
    return record_index_to_id_map


def get_tokenized_table(table, table_key, feature_list):
    record_list = []
    columns = table.columns[feature_list]
    tmp_table = []
    for col in columns:
        column_token_list = get_tokenized_column(table[col])
        tmp_table.append(column_token_list)

    num_records = len(table[table_key])
    for i in range(num_records):
        token_list = []
        index_map = {}

        for j in range(len(columns)):
            tmp_col_tokens = tmp_table[j][i]
            for token in tmp_col_tokens:
                if token != '':
                    if token in index_map:
                        token_list.append((token + '_' + str(index_map[token]), j))
                        index_map[token] += 1
                    else:
                        token_list.append((token, j))
                        index_map[token] = 1
        record_list.append(token_list)

    return record_list


def get_tokenized_column(column):
    column_token_list = []
    for value in list(column):
        tmp_value = replace_nan_to_empty(value)
        if tmp_value != '':
            tmp_list = list(tmp_value.lower().split(' '))
            column_token_list.append(tmp_list)
        else:
            column_token_list.append([''])
    return column_token_list


def replace_nan_to_empty(field):
    if pd.isnull(field):
        return ''
    elif type(field) in [float, numpy.float64, int, numpy.int64]:
        return str('{0:.0f}'.format(field))
    else:
        return str(field)
        # return field


def index_candidate_set(candidate_set, lrecord_id_to_index_map, rrecord_id_to_index_map, verbose):
    if len(candidate_set) == 0:
        return {}
    new_formatted_candidate_set = {}
    # # get metadata
    key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key = \
        cm.get_metadata_for_candset(candidate_set, logger, verbose)

    # # validate metadata
    # cm._validate_metadata_for_candset(candidate_set, key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key,
    #                                  logger, verbose)

    ltable_key_data = list(candidate_set[fk_ltable])
    rtable_key_data = list(candidate_set[fk_rtable])

    for i in range(len(ltable_key_data)):
        if ltable_key_data[i] in lrecord_id_to_index_map and \
                        rtable_key_data[i] in rrecord_id_to_index_map:
            # new_formatted_candidate_set.add((lrecord_id_to_index_map[ltable_key_data[i]],
            #                                 rrecord_id_to_index_map[rtable_key_data[i]]))
            l_key_data = lrecord_id_to_index_map[ltable_key_data[i]]
            r_key_data = rrecord_id_to_index_map[rtable_key_data[i]]
            if l_key_data in new_formatted_candidate_set:
                new_formatted_candidate_set[l_key_data].add(r_key_data)
            else:
                new_formatted_candidate_set[l_key_data] = {r_key_data}

    return new_formatted_candidate_set


def build_global_token_order(lrecord_list, rrecord_list):
    freq_order_dict = {}
    build_global_token_order_impl(lrecord_list, freq_order_dict)
    build_global_token_order_impl(rrecord_list, freq_order_dict)
    token_list = []
    for token in freq_order_dict:
        token_list.append(token)
    token_list = sorted(token_list, key=lambda x: (freq_order_dict[x], x))

    order_dict = {}
    token_index_dict = {}
    for i in range(len(token_list)):
        order_dict[token_list[i]] = i
        token_index_dict[i] = token_list[i]

    return order_dict, token_index_dict


def build_global_token_order_impl(record_list, order_dict):
    for record in record_list:
        for tup in record:
            token = tup[0]
            if token in order_dict:
                order_dict[token] += 1
            else:
                order_dict[token] = 1


def replace_token_with_numeric_index(record_list, order_dict):
    for i in range(len(record_list)):
        tmp_record = []
        for tup in record_list[i]:
            token = tup[0]
            index = tup[1]
            if token in order_dict:
                tmp_record.append((order_dict[token], index))
        record_list[i] = tmp_record


def sort_record_tokens_by_global_order(record_list):
    for i in range(len(record_list)):
        record_list[i] = sorted(record_list[i], key=lambda x: x[0])


def split_record_token_and_index(record_list, num_fields):
    record_token_list = []
    record_index_list = []
    record_field_list = []
    for i in range(len(record_list)):
        token_list = []
        index_list = []
        field_list = []
        for j in range(num_fields):
            field_list.append(0)
        for j in range(len(record_list[i])):
            token_list.append(record_list[i][j][0])
            index_list.append(record_list[i][j][1])
            field_list[record_list[i][j][1]] += 1
        record_token_list.append(array('I', token_list))
        record_index_list.append(array('I', index_list))
        record_field_list.append(array('I', field_list))

    return record_token_list, record_index_list, record_field_list


if __name__ == "__main__":
    # output_path = '../results/'
    # folder = 'citations'
    # lkey = 'ID'
    # rkey = 'ID'
    # ltable = mg.read_csv_metadata('../../debug_blocking/datasets/example_datasets/' + folder + '/A.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../../debug_   blocking/datasets/example_datasets/' + folder + '/B.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../../debug_blocking/datasets/example_datasets/' + folder + '/C.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, output_path + folder + '/', output_size)

#####################################################################################################
##### experiments
#####################################################################################################
    # TOPK_LISTS = [1, 10, 100, 200, 300, 400, 500, 600]
    # TOPK_LISTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    # TOPK_LISTS = [3100, 3200, 3300, 3400, 3500]
    TOPK_LISTS = [1000]
    EXP_TYPES = ['exp_reuse_topk', 'exp_noreuse_topk', 'exp_temp', 'exp_sensitivity_K',
                 'exp', 'final_topk_efficiency_cmp_exp', 'rm_long_field_cmp_exp',
                 'exp_long_field', 'exp_large_k', 'exp_hash_blocker', 'exp_thread_opt']
    # BLOCK_TYPES = {'rule-based':['title_token_jac<0.6_OR_absdiff_price>10']}
    # BLOCK_TYPES = {'overlap': ['authors_overlap<2']}
    # BLOCK_TYPES = ['overlap', 'hash-based', 'similarity-based', 'rule-based']

    # Amazon-GoogleProducts
    # BLOCK_TYPES = {'hash-based':['int_price', 'manuf', 'manuf_OR_int_price',
    #                              'manuf_OR_int_price_OR_trunc_title_1token',
    #                              'manuf_OR_price_OR_title',
    #                              'manuf_OR_trunc_price_OR_trunc_title_1token',
    #                              'price', 'title', 'trunc_price', 'trunc_title_1token']}
    # BLOCK_TYPES = {'hash-based':['manuf']}

    # # Walmart-Amazon
    # BLOCK_TYPES = {'hash-based':['brand', 'brand_OR_modelno_OR_category_OR_title_OR_price',
    #                              'brand_OR_modelno_OR_category_OR_trunc_title_OR_trunc_price',
    #                              'brand_OR_modelno_OR_category_OR_trunc_title_OR_price',
    #                              'category', 'modelno', 'price', 'title', 'trunc_price', 'trunc_title']}
    # BLOCK_TYPES = {'hash-based' : ['brand']}

    # ACM-DBLP
    # BLOCK_TYPES = {'hash-based':['authors', 'title']}
    #
    # # Fodors-Zagats
    # BLOCK_TYPES = {'hash-based':['addr', 'name', 'city', 'type', 'name_OR_addr_OR_city_OR_type']}
    # BLOCK_TYPES = {'hash-based':['addr']}

    #
    # # Song-Song
    # BLOCK_TYPES = {'hash-based':['artist_name', 'release', 'title']}

    # BLOCK_TYPES = ['overlap']
    # BLOCK_TYPES = {'hash-based':['attr_equal_artist_name'],
    #                'rule-based':['title_token_cos<0.7_OR_year_diff'],
    #                'overlap':['artist_name_overlap<2'],
    #                'similarity-based':['title_token_cos<0.5']}
    # BLOCK_TYPES = {'overlap':['title_overlap<3']}
    # BLOCK_TYPES = {'overlap':['title_overlap<3'],
    #                'hash-based':['attr_equal_brand'],
    #                'similarity-based':['title_token_cos<0.4'],
    #                'rule-based':['title_token_jac<0.5_OR_absdiff_price>20']}
    # BLOCK_TYPES = {'hash-based':['attr_equal_artist_name']}
    # BLOCK_TYPES = {'overlap': ['authors_overlap<2']}
    BLOCK_TYPES = {'rule-based': ['title_jac_token<0.2_AND_manuf_jac_3gram<0.4']}
    # BLOCK_TYPES = {'overlap':['title_overlap<3'],
    #                'hash-based':['attr_equal_manuf'],
    #                'similarity-based':['title_token_cos<0.4'],
    #                'rule-based':['title_jac_token<0.2_AND_manuf_jac_3gram<0.4']}
    TOPK_ALGO_TYPES = ['new_topk_nopara_noreuse', 'new_topk_para_noreuse', 'new_topk_para_reuse',
                      'original_topk_nopara_noreuse', 'original_topk_para_noreuse']
    KEEP_LONG_FIELD_TYPES = ['keep_long_field', 'rm_long_field']
    for topk in TOPK_LISTS:
        exp_type = EXP_TYPES[10]
        keep_long_field_type = KEEP_LONG_FIELD_TYPES[1]
        # output_dir = '../results/exp/top' + str(topk) + '/'
        topk_type = TOPK_ALGO_TYPES[2]
        output_dir = '../results/' + exp_type + '/top' + str(topk) + '/'
        activate_reusing_module = True
        use_new_topk = True
        use_parallel = True
        folder = "Amazon-GoogleProducts"
        # blocker_type = BLOCK_TYPES[3]
        # cand_name = 'title_overlap<3'
        for blocker_type in BLOCK_TYPES:
            lkey = 'id'
            rkey = 'id' 
            cand_list = os.listdir(output_dir + folder + '/' + blocker_type + '/')
            if '.DS_Store' in cand_list:
                cand_list.remove('.DS_Store')
            if 'combined_list' in cand_list:
                cand_list.remove('combined_list')
            cand_list = BLOCK_TYPES[blocker_type]
            print cand_list
            runtime_list = []
            for cand_name in cand_list:
                ltable = mg.read_csv_metadata('../../debug_blocking_cython/datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
                rtable = mg.read_csv_metadata('../../debug_blocking_cython/datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
                cand_set = mg.read_csv_metadata('../../debug_blocking_cython/datasets/exp_data/candidate_sets/' + folder + '/' +
                                                blocker_type + '/' + cand_name + '.csv',
                                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
                                                fk_rtable='rtable_' + rkey, key='_id')
                output_path = output_dir + folder + '/' + blocker_type + '/' + cand_name + '/'
                runtime_list.append(debug_blocker(ltable, rtable, cand_set, activate_reusing_module,
                                                  use_new_topk, use_parallel, output_path, topk))

            for cand_name in cand_list:
                print cand_name
            for t in runtime_list:
                print t

    # output_path = '../results/exp/'
    # use_plain = False
    # folder = 'Walmart-Amazon'
    # cand_name = 'cand_title<0.3'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/candidate_sets/' + cand_name + '.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/' + cand_name + '/', output_size)

    # output_dir = '../results/exp/new/'
    # use_plain = False
    # folder = 'Amazon-GoogleProducts'
    # blocker_type = 'similarity-based'
    # cand_name = 'title_token_cos<0.8'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/exp_data/candidate_sets/' + folder + '/' + blocker_type + '/' + cand_name + '.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # output_path = output_dir + folder + '/' + blocker_type + '/' + cand_name + '/'
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path, output_size)


    # output_path = '../results/exp/'
    # use_plain = False
    # folder = 'DBLP-GoogleScholar'
    # cand_name = 'cand_rule8'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/candidate_sets/' + cand_name + '.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/' + cand_name + '/', output_size)


    # output_path = '../results/exp/'
    # use_plain = False
    # folder = 'ACM-DBLP'
    # cand_name = 'cand_title_token_jac<0.8_AND_venue_3gram_jac<0.5'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/candidate_sets/' + cand_name + '.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/' + cand_name + '/', output_size)

    # output_path = '../results/exp/'
    # use_plain = False
    # folder = 'Fodors-Zagats'
    # cand_name = 'cand_name_token_jac<0.5_AND_type_token_jac<0.5'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/exp_data/cleaned/' + folder + '/candidate_sets/' + cand_name + '.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/' + cand_name + '/', output_size)
#####################################################################################################
    # output_path = '../results/new_allconfig_reuse_openmp/'
    # output_path = '../results/temp/'
    # folder = 'M_ganz'
    # use_plain = False
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/' + folder + '/tableC_new.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/', output_size)

    # output_path = '../results/new_allconfig_noreuse_openmp/'
    # output_path = '../results/temp/'
    # use_plain = False
    # folder = 'L_bsarkar'
    # lkey = 'Id'
    # rkey = 'Id'
    # ltable = mg.read_csv_metadata('../datasets/' + folder + '/tableA.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/' + folder + '/tableB.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/' + folder + '/tableC_new.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/', output_size)

    # output_path = '../results/new_allconfig_noreuse_openmp/'
    # use_plain = True
    # folder = 'TREC'
    # lkey = 'id'
    # rkey = 'id'
    # ltable = mg.read_csv_metadata('../datasets/' + folder + '/ohsumed_87.csv', key=lkey)
    # rtable = mg.read_csv_metadata('../datasets/' + folder + '/ohsumed_87.csv', key=rkey)
    # cand_set = mg.read_csv_metadata('../datasets/' + folder + '/tableC.csv',
    #                                ltable=ltable, rtable=rtable, fk_ltable='ltable_' + lkey,
    #                                fk_rtable='rtable_' + rkey, key='_id')
    # output_size = 200
    # debug_blocker(ltable, rtable, cand_set, use_plain, output_path + folder + '/', output_size)

    # folders = ['anime', 'beer', 'books', 'citations', 'cosmetics', 'ebooks', 'electronics']
    # key_pairs = [('ID', 'ID'), ('Label', 'Label'), ('ID', 'ID'), ('ID', 'ID'),
    #              ('Product_id', 'ID'), ('record_id', 'record_id'), ('ID', 'ID')]
    # # folders = ['ebooks', 'electronics']
    # # key_pairs = [('record_id', 'record_id'), ('ID', 'ID')]
    # # folders = ['electronics']
    # # key_pairs = [('ID', 'ID')]
    # use_plain = False
    # run_times = []
    # for i in range(len(folders)):
    #     # FOLDER = '../results/new_allconfig_reuse_openmp/' + folders[i] + '/'
    #     FOLDER = '../results/temp/' + folders[i] + '/'
    #
    #     #folder = 'ebooks'
    #     folder = folders[i]
    #     path_for_A = os.sep.join(['../datasets/', folder, 'A.csv'])
    #     path_for_B = os.sep.join(['../datasets/', folder, 'B.csv'])
    #     path_for_C = os.sep.join(['../datasets/', folder, 'C.csv'])
    #     #lkey = 'ID'
    #     #rkey = 'ID'
    #     lkey = key_pairs[i][0]
    #     rkey = key_pairs[i][1]
    #     A = mg.read_csv_metadata(path_for_A)
    #     mg.set_key(A, lkey)
    #     B = mg.read_csv_metadata(path_for_B)
    #     mg.set_key(B, rkey)
    #     C = mg.read_csv_metadata(path_for_C)
    #     cm.set_candset_properties(C, '_id', 'ltable_' + lkey, 'rtable_' + rkey, A, B)
    #     cm.set_candset_properties(C, '_id', 'ltable_' + lkey, 'rtable_' + rkey, A, B)
    #     #attr_corres = [('ID', 'ID'), ('Brand', 'Brand'), ('Name', 'Name'),
    #     #                        ('Amazon_Price', 'Price'), ('Features', 'Features')]
    #     output_size = 200
    #     print 'start debugging'
    #     # ret_table = debug_blocker(A, B, C, 1, output_size)
    #     # ret_table = debug_blocker(A, B, C, 1, 2, output_size)
    #     total_time = debug_blocker(A, B, C, use_plain, FOLDER, output_size)
    #     run_times.append(total_time)
    #
    # print run_times
    # for i in range(len(folders)):
    #     print folders[i], run_times[i]
