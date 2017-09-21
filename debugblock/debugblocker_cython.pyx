from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.queue cimport priority_queue as heap
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf
from libc.stdint cimport uint32_t as uint, uint64_t
from cython.parallel import prange, parallel
import time

# include "new_topk_sim_join.pyx"
# include "original_topk_sim_join.pyx"

cdef extern from "TopPair.h" nogil:
    cdef cppclass TopPair nogil:
        TopPair()
        TopPair(double, int, int)
        double sim
        uint l_rec
        uint r_rec
        bool operator>(const TopPair& other)
        bool operator<(const TopPair& other)

# cdef extern from "PrefixEvent.h" nogil:
#     cdef cppclass PrefixEvent:
#         PrefixEvent()
#         PrefixEvent(double, int, int, int)
#         double threshold
#         int table_indicator
#         int rec_idx
#         int tok_idx
#         bool operator>(const PrefixEvent& other)
#         bool operator<(const PrefixEvent& other)

cdef extern from "ReuseInfoArray.h" nogil:
    cdef cppclass ReuseInfoArray:
        ReuseInfoArray()
        ReuseInfoArray(int)
        int overlap
        int info[8][8]


cdef extern from "Signal.h" nogil:
    cdef cppclass Signal:
        int value;


# cdef extern from "TopkListGenerator.h" nogil:
#     cdef cppclass TopkListGenerator:
#         TopkListGenerator()
#         void generate_topklist_for_config(const vector[vector[int]]& ltoken_vector, const vector[vector[int]]& rtoken_vector,
#                                           const vector[vector[int]]& lindex_vector, const vector[vector[int]]& rindex_vector,
#                                           const vector[vector[int]]& lfield_vector, const vector[vector[int]]& rfield_vector,
#                                           const vector[int]& ltoken_sum_vector, const vector[int]& rtoken_sum_vector,
#                                           const vector[int]& field_list, const int max_field_num, const int removed_field,
#                                           const bool first_run, Signal& signal,
#                                           cmap[int, cset[int]]& cand_set, cmap[int, cmap[int, ReuseInfoArray]]& reuse_set,
#                                           vector[heap[TopPair]]& heap_vector, const heap[TopPair]& init_topk_heap,
#                                           const int prefix_match_max_size, const int rec_ave_length, const int output_size,
#                                           const int use_plain, const int topk_type, const string& output_path)


cdef extern from "GenerateRecomLists.h" nogil:
    cdef cppclass GenerateRecomLists:
        GenerateRecomLists()
        void generate_recom_lists(vector[vector[int]]& ltoken_vector, vector[vector[int]]& rtoken_vector,
                              vector[vector[int]]& lindex_vector, vector[vector[int]]& rindex_vector,
                              vector[vector[int]]& lfield_vector, vector[vector[int]]& rfield_vector,
                              vector[int]& ltoken_sum_vector, vector[int]& rtoken_sum_vector, vector[int]& field_list,
                              cmap[int, cset[int]]& cand_set, uint prefix_match_max_size, const uint rec_ave_len_thres,
                              uint offset_of_field_num, uint max_field_num,
                              uint minimal_num_fields, double field_remove_ratio,
                              uint output_size, string output_path,
                              const bool activate_reusing_module, const bool use_new_topk, const bool use_parallel);

PREFIX_MATCH_MAX_SIZE = 5
REC_AVE_LEN_THRES = 20
OFFSET_OF_FIELD_NUM = 10
MINIMAL_NUM_FIELDS = 0
FIELD_REMOVE_RATIO = 0.1

def debugblocker_cython(lrecord_token_list, rrecord_token_list,
                        lrecord_index_list, rrecord_index_list,
                        lrecord_field_list, rrecord_field_list,
                        ltable_field_token_sum, rtable_field_token_sum, py_cand_set,
                        py_num_fields, py_output_size, py_output_path, py_use_plain,
                        py_use_new_topk, py_use_parallel):
    cdef string output_path = py_output_path
    cdef bool use_plain = py_use_plain
    cdef bool use_new_topk = py_use_new_topk
    cdef bool use_parallel = py_use_parallel

    ### Convert py objs to c++ objs
    cdef vector[vector[int]] ltoken_vector, rtoken_vector
    convert_table_to_vector(lrecord_token_list, ltoken_vector)
    convert_table_to_vector(rrecord_token_list, rtoken_vector)

    cdef vector[vector[int]] lindex_vector, rindex_vector
    convert_table_to_vector(lrecord_index_list, lindex_vector)
    convert_table_to_vector(rrecord_index_list, rindex_vector)

    cdef vector[vector[int]] lfield_vector, rfield_vector
    convert_table_to_vector(lrecord_field_list, lfield_vector)
    convert_table_to_vector(rrecord_field_list, rfield_vector)

    cdef vector[int] ltoken_sum, rtoken_sum
    convert_py_list_to_vector(ltable_field_token_sum, ltoken_sum)
    convert_py_list_to_vector(rtable_field_token_sum, rtoken_sum)

    cdef cmap[int, cset[int]] cand_set
    convert_candidate_set_to_c_map(py_cand_set, cand_set)

    cdef vector[int] field_list
    for i in range(py_num_fields):
        field_list.push_back(i)

    cdef uint output_size = py_output_size
    cdef uint prefix_match_max_size = PREFIX_MATCH_MAX_SIZE
    cdef uint rec_ave_len_thres = REC_AVE_LEN_THRES
    cdef uint offset_of_field_num = OFFSET_OF_FIELD_NUM
    cdef uint minimal_num_fields = MINIMAL_NUM_FIELDS
    cdef double field_remove_ratio = FIELD_REMOVE_RATIO

    del lrecord_token_list, rrecord_token_list
    del lrecord_index_list, rrecord_index_list
    del lrecord_field_list, rrecord_field_list
    del py_cand_set

    ### Generate recommendation topk lists
    # cdef vector[vector[TopPair]] topk_lists
    cdef heap[TopPair] init_topk_heap

    cdef int max_field_num = field_list.size()

    # generate_recom_lists(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
    #                      lfield_vector, rfield_vector,
    #                      ltoken_sum, rtoken_sum, field_list, topk_lists,
    #                      True, cand_set, reuse_set, prefix_match_max_size,
    #                      rec_ave_len_thres, offset_of_field_num, max_field_num, minimal_num_fields,
    #                      field_remove_ratio, output_size, output_path, init_topk_heap,
    #                      use_plain, use_new_topk, use_parallel)
    cdef GenerateRecomLists generator
    generator.generate_recom_lists(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                   lfield_vector, rfield_vector, ltoken_sum, rtoken_sum, field_list,
                                   cand_set, prefix_match_max_size, rec_ave_len_thres,
                                   offset_of_field_num, max_field_num, minimal_num_fields, field_remove_ratio,
                                   output_size, output_path, use_plain, use_new_topk, use_parallel)


# cdef void generate_recom_lists(vector[vector[int]]& ltoken_vector, vector[vector[int]]& rtoken_vector,
#                                vector[vector[int]]& lindex_vector, vector[vector[int]]& rindex_vector,
#                                vector[vector[int]]& lfield_vector, vector[vector[int]]& rfield_vector,
#                                const vector[int]& ltoken_sum_vector, const vector[int]& rtoken_sum_vector,
#                                vector[int]& field_list, vector[vector[TopPair]]& topk_lists,
#                                const bool first_run, cmap[int, cset[int]]& cand_set, cmap[int, cmap[int, ReuseInfoArray]] reuse_set,
#                                uint prefix_match_max_size, const uint rec_ave_len_thres,
#                                const uint offset_of_field_num, const int max_field_num, const uint minimal_num_fields,
#                                const double field_remove_ratio, const uint output_size,
#                                const string output_path, heap[TopPair] init_topk_heap,
#                                const bool activate_reusing_module, const bool use_new_topk, const bool use_parallel):
#     if field_list.size() <= minimal_num_fields:
#         print 'too few lists:', field_list
#         print 'end time:', time.time()
#         return
#
#     cdef TopkListGenerator generator
#
#     start = time.time()
#     cdef heap[TopPair] resulted_topk_heap
#
#     cdef uint i
#     cdef int p
#     cdef vector[heap[TopPair]] heap_vector
#     for i in xrange(field_list.size()):
#         heap_vector.push_back(heap[TopPair]())
#
#     cdef Signal signal
#     cdef Signal temp_signal
#
#     cdef vector[cmap[int, cmap[int, ReuseInfoArray]]] para_reuse_set
#     cdef vector[vector[heap[TopPair]]] para_heap_vector
#     cdef vector[int] para_prefix_vector
#     cdef cmap[int, int] para_prefix_map
#
#     if use_new_topk and first_run:
#         para_prefix_vector.push_back(5)
#         para_prefix_vector.push_back(4)
#         para_prefix_vector.push_back(3)
#         para_prefix_vector.push_back(0)
#         para_prefix_map[0] = 3
#         para_prefix_map[3] = 2
#         para_prefix_map[4] = 1
#         para_prefix_map[5] = 0
#         for i in xrange(4):
#             para_reuse_set.push_back(cmap[int, cmap[int, ReuseInfoArray]]())
#             para_heap_vector.push_back(create_heap_vector(field_list.size()))
#
#         with nogil, parallel(num_threads=4):
#             for p in prange(4):
#                 generator.generate_topklist_for_config(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
#                                                        lfield_vector, rfield_vector, ltoken_sum_vector, rtoken_sum_vector,
#                                                        field_list, max_field_num, 0, first_run, signal, cand_set,
#                                                        para_reuse_set[p], para_heap_vector[p],
#                                                        init_topk_heap, para_prefix_vector[p], rec_ave_len_thres, output_size,
#                                                        activate_reusing_module, 0, output_path)
#         prefix_match_max_size = signal.value
#         heap_vector = para_heap_vector[para_prefix_map[signal.value]]
#         reuse_set = para_reuse_set[para_prefix_map[signal.value]]
#     else:
#         generator.generate_topklist_for_config(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
#                                                lfield_vector, rfield_vector, ltoken_sum_vector, rtoken_sum_vector,
#                                                field_list, max_field_num, 0, False, signal, cand_set, reuse_set, heap_vector,
#                                                init_topk_heap, prefix_match_max_size, rec_ave_len_thres, output_size,
#                                                activate_reusing_module, 0, output_path)
#     printf("reuse size: %d\n", reuse_set.size())
#
#     end = time.time()
#     print 'join time:', end - start
#
#     cdef double max_ratio = 0.0
#     cdef uint ltoken_total_sum = 0, rtoken_total_sum = 0
#     cdef int removed_field_index = -1
#     cdef bool has_long_field = False
#
#     for i in range(field_list.size()):
#         ltoken_total_sum += ltoken_sum_vector[field_list[i]]
#         rtoken_total_sum += rtoken_sum_vector[field_list[i]]
#
#     cdef double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size()
#     cdef double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size()
#     cdef double ratio = 1 - (field_list.size() - 1) * field_remove_ratio / (1.0 + field_remove_ratio) *\
#                  double_max(lrec_ave_len, rrec_ave_len) / (lrec_ave_len + rrec_ave_len)
#     # cdef double ratio = 0.5
#
#     for i in range(field_list.size()):
#         max_ratio = double_max(max_ratio, double_max(ltoken_sum_vector[field_list[i]] * 1.0 / ltoken_total_sum,
#                                                      rtoken_sum_vector[field_list[i]] * 1.0 / rtoken_total_sum))
#         if ltoken_sum_vector[field_list[i]] > ltoken_total_sum * ratio or\
#                 rtoken_sum_vector[field_list[i]] > rtoken_total_sum * ratio:
#             removed_field_index = i
#             has_long_field = True
#             break
#
#     if removed_field_index < 0:
#         removed_field_index = field_list.size() - 1
#     # removed_field_index = field_list.size() - 1
#     print 'required remove-field ratio:', ratio
#     print 'actual max ratio:', max_ratio
#
#
#     cdef vector[heap[TopPair]] heap_vector_parallel
#     cdef vector[vector[vector[int]]] ltoken_vector_parallel
#     cdef vector[vector[vector[int]]] rtoken_vector_parallel
#     cdef vector[vector[int]] field_list_parallel
#     cdef vector[int] field_parallel
#     cdef vector[int] temp
#     if True:
#     # if not has_long_field:
#         for i in range(field_list.size()):
#             if i != removed_field_index:
#                 temp = vector[int](field_list)
#                 temp.erase(temp.begin() + i)
#                 if temp.size() > minimal_num_fields:
#                     field_list_parallel.push_back(temp)
#                     field_parallel.push_back(field_list[i])
#                     heap_vector_parallel.push_back(heap_vector[i])
#                     ltoken_vector_parallel.push_back(vector[vector[int]]())
#                     copy_table_and_remove_field(ltoken_vector, lindex_vector,
#                                                 ltoken_vector_parallel[ltoken_vector_parallel.size() - 1],
#                                                 field_list[i])
#                     rtoken_vector_parallel.push_back(vector[vector[int]]())
#                     copy_table_and_remove_field(rtoken_vector, rindex_vector,
#                                                 rtoken_vector_parallel[rtoken_vector_parallel.size() - 1],
#                                                 field_list[i])
#
#         if use_parallel:
#             with nogil, parallel(num_threads=field_parallel.size()):
#                 for p in prange(field_parallel.size()):
#                     generator.generate_topklist_for_config(ltoken_vector_parallel[p], rtoken_vector_parallel[p],
#                                                            lindex_vector, rindex_vector, lfield_vector, rfield_vector,
#                                                            ltoken_sum_vector, rtoken_sum_vector,
#                                                            field_list_parallel[p], max_field_num, field_list[p], False, temp_signal, cand_set, reuse_set,
#                                                            heap_vector, heap_vector_parallel[p], prefix_match_max_size, rec_ave_len_thres,
#                                                            output_size, activate_reusing_module, 1, output_path)
#         else:
#             for p in range(field_parallel.size()):
#                 generator.generate_topklist_for_config(ltoken_vector_parallel[p], rtoken_vector_parallel[p],
#                                                        lindex_vector, rindex_vector, lfield_vector, rfield_vector,
#                                                        ltoken_sum_vector, rtoken_sum_vector,
#                                                        field_list_parallel[p], max_field_num, field_list[p], False, temp_signal, cand_set, reuse_set,
#                                                        heap_vector, heap_vector_parallel[p], prefix_match_max_size, rec_ave_len_thres,
#                                                        output_size, activate_reusing_module, 1, output_path)
#
#         # for p in range(field_parallel.size()):
#         #     generate_recom_list_for_config(ltoken_vector_parallel[p], rtoken_vector_parallel[p],
#         #                                    lindex_vector, rindex_vector,
#         #                                    ltoken_sum_vector, rtoken_sum_vector,
#         #                                    field_list_parallel[p], cand_set, reuse_set,
#         #                                    prefix_match_max_size, prefix_multiply_factor,
#         #                                    offset_of_field_num, output_size, use_plain, 1, output_path)
#
#     print 'remove', field_list[removed_field_index]
#     remove_field(ltoken_vector, lindex_vector, field_list[removed_field_index])
#     remove_field(rtoken_vector, rindex_vector, field_list[removed_field_index])
#     field_list.erase(field_list.begin() + removed_field_index)
#
#     generate_recom_lists(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector, lfield_vector, rfield_vector,
#                          ltoken_sum_vector, rtoken_sum_vector, field_list, topk_lists, False, cand_set, reuse_set,
#                          prefix_match_max_size, rec_ave_len_thres, offset_of_field_num, max_field_num,
#                          minimal_num_fields, field_remove_ratio, output_size, output_path,
#                          heap_vector[removed_field_index], activate_reusing_module, use_new_topk, use_parallel)
#
#     return


cdef vector[heap[TopPair]]create_heap_vector(const int size):
    cdef vector[heap[TopPair]] heap_vector
    for i in xrange(size):
        heap_vector.push_back(heap[TopPair]())

    return heap_vector


# cdef heap[TopPair] generate_recom_list_for_config(const vector[vector[int]]& ltoken_vector, const vector[vector[int]]& rtoken_vector,
#                                          const vector[vector[int]]& lindex_vector, const vector[vector[int]]& rindex_vector,
#                                          const vector[int]& ltoken_sum_vector, const vector[int]& rtoken_sum_vector,
#                                          const vector[int]& field_list, cmap[int, cset[int]]& cand_set,
#                                          cmap[int, cmap[int, ReuseInfo]]& reuse_set, const heap[TopPair]& init_topk_heap,
#                                          const uint prefix_match_max_size, const uint prefix_multiply_factor,
#                                          const uint offset_of_field_num, const uint output_size,
#                                          const bool use_plain, const uint type, const string& output_path) nogil:
#     cdef uint i, j
#     cdef char buf[10]
#
#     cdef string info = string(<char *>'current configuration: [')
#     for i in xrange(field_list.size()):
#         sprintf(buf, "%d", field_list[i])
#         if i != 0:
#             info.append(<char *>', ')
#         info.append(buf)
#     info += <char *>"]  "
#     # printf("current configuration: [")
#     # for i in xrange(field_list.size()):
#     #     if i == 0:
#     #         printf("%d", field_list[i])
#     #     else:
#     #         printf(" %d", field_list[i])
#     # printf("]\n")
#
#     cdef int ltoken_total_sum = 0, rtoken_total_sum = 0
#     for i in xrange(field_list.size()):
#         ltoken_total_sum += ltoken_sum_vector[field_list[i]]
#         rtoken_total_sum += rtoken_sum_vector[field_list[i]]
#
#     cdef double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size()
#     cdef double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size()
#     cdef double len_threshold = prefix_match_max_size * 1.0 * prefix_multiply_factor
#
#     cdef cset[int] remained_fields
#     for i in xrange(field_list.size()):
#         remained_fields.insert(field_list[i])
#
#     cdef heap[TopPair] topk_heap
#     cdef heap[TopPair] copy_init_topk_heap = init_topk_heap
#     cdef vector[TopPair] updated_init_topk_list
#     init_topk_heap_cython(ltoken_vector, rtoken_vector, copy_init_topk_heap, updated_init_topk_list)
#
#     if use_plain:
#         if lrec_ave_len >= len_threshold and rrec_ave_len >= len_threshold:
#             # topk_heap = new_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set,
#             #                                     prefix_match_max_size, output_size)
#             topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size)
#         else:
#             # topk_heap = new_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set,
#             #                                     1, output_size)
#             topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size)
#     else:
#         if lrec_ave_len >= len_threshold and rrec_ave_len >= len_threshold:
#             info += <char *>'new topk'
#             printf("%s\n", info.c_str())
#             if type == 0:
#                 topk_heap = new_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector, updated_init_topk_list,
#                                                      cand_set, reuse_set, offset_of_field_num, prefix_match_max_size, output_size)
#             elif type == 1:
#                 topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector, updated_init_topk_list,
#                                                     remained_fields, cand_set, reuse_set, offset_of_field_num, prefix_match_max_size, output_size)
#         else:
#             info += <char *>'original topk'
#             printf("%s\n", info.c_str())
#             # if type == 0:
#             #     topk_heap = original_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
#             #                                   cand_set, reuse_set, offset_of_field_num, output_size)
#             # elif type == 1:
#             #     topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector, remained_fields,
#             #                                  cand_set, reuse_set, offset_of_field_num, output_size)
#             if type == 0:
#                 topk_heap = new_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector, updated_init_topk_list,
#                                                      cand_set, reuse_set, offset_of_field_num, 1, output_size)
#             elif type == 1:
#                 topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector, updated_init_topk_list,
#                                                     remained_fields, cand_set, reuse_set, offset_of_field_num, 1, output_size)
#         # else:
#         #     if type == 0:
#         #         topk_heap = original_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
#         #                                       cand_set, reuse_set, offset_of_field_num, output_size)
#         #     elif type == 1:
#         #         topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector, remained_fields,
#         #                                      cand_set, reuse_set, offset_of_field_num, output_size)
#
#     save_topk_list_to_file(field_list, output_path, topk_heap)
#
#     return topk_heap


cdef void save_topk_list_to_file(const vector[int]& field_list, const string& output_path,
                                 heap[TopPair] topk_heap) nogil:
    cdef string path = output_path + <char *>'topk_'
    cdef char buf[10]
    cdef int i
    for i in xrange(field_list.size()):
        sprintf(buf, "%d", field_list[i])
        if i != 0:
            path.append(<char *>'_')
        path.append(buf)
    path += <char *>'.txt'
    printf("%s\n", path.c_str())

    cdef TopPair pair
    cdef FILE* fp = fopen(path.c_str(), "w+")
    while topk_heap.size() > 0:
        pair = topk_heap.top()
        topk_heap.pop()
        fprintf(fp, "%.16f %d %d\n", pair.sim, pair.l_rec, pair.r_rec)
    fclose(fp)

    return


cdef void copy_table_and_remove_field(const vector[vector[int]]& table_vector,
                                      const vector[vector[int]]& index_vector,
                                      vector[vector[int]]& new_table_vector, int rm_field):
    cdef uint i, j
    for i in xrange(table_vector.size()):
        new_table_vector.push_back(vector[int]())
        for j in xrange(table_vector[i].size()):
            if index_vector[i][j] != rm_field:
                new_table_vector[i].push_back(table_vector[i][j])


cdef void remove_field(vector[vector[int]]& table_vector,
                       vector[vector[int]]& index_vector, int rm_field):
    cdef uint i, j
    for i in xrange(table_vector.size()):
        for j in reversed(range(table_vector[i].size())):
            if index_vector[i][j] == rm_field:
                index_vector[i].erase(index_vector[i].begin() + j)
                table_vector[i].erase(table_vector[i].begin() + j)


cdef void convert_table_to_vector(table_list, vector[vector[int]]& table_vector):
    cdef int i, j
    for i in range(len(table_list)):
        table_vector.push_back(vector[int]())
        for j in range(len(table_list[i])):
            table_vector[i].push_back(table_list[i][j])


cdef void convert_candidate_set_to_c_map(cand_set, cmap[int, cset[int]]& new_set):
    cdef int key, value
    for key in cand_set:
        if not new_set.count(key):
            new_set[key] = cset[int]()

        l = cand_set[key]
        for value in l:
            new_set[key].insert(value)


cdef int convert_py_list_to_vector(py_list, vector[int]& vector):
    for value in py_list:
        vector.push_back(value)


cdef double double_max(const double a, double b):
    if a > b:
        return a
    return b


cdef void init_topk_heap_cython(const vector[vector[int]]& ltoken_vector, const vector[vector[int]]& rtoken_vector,
                                heap[TopPair]& old_heap, vector[TopPair]& new_heap) nogil:
    cdef int a
    cdef int overlap = 0
    cdef uint i
    cdef double new_sim
    cdef TopPair pair
    while old_heap.size() > 0:
        pair = old_heap.top()
        old_heap.pop()

        new_sim = init_topk_heap_calc_sim(ltoken_vector[pair.l_rec], rtoken_vector[pair.r_rec])
        if new_sim > 0:
            new_heap.push_back(TopPair(new_sim, pair.l_rec, pair.r_rec))

    return


cdef double init_topk_heap_calc_sim(const vector[int]& ltoken_list, const vector[int]& rtoken_list) nogil:
    cdef int overlap = 0
    cdef cset[int] rset
    cdef int denom = 0
    for i in xrange(rtoken_list.size()):
        rset.insert(rtoken_list[i])

    for i in xrange(ltoken_list.size()):
        if rset.count(ltoken_list[i]):
            overlap += 1


    denom = ltoken_list.size() + rtoken_list.size() - overlap
    if denom == 0:
        return 0.0
    return overlap * 1.0 / denom
