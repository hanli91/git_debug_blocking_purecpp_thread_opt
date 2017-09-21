from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set as uset
from libcpp.unordered_map cimport unordered_map as umap
from libcpp.set cimport set as oset
from libcpp.map cimport map as omap
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue as heap
from libcpp cimport bool
from libc.stdio cimport printf
from libc.stdint cimport uint32_t as uint, uint64_t
from libc.stdlib cimport malloc, free

cdef extern from "TopPair.h" nogil:
    cdef cppclass TopPair nogil:
        TopPair()
        TopPair(double, uint, uint)
        double sim
        uint l_rec
        uint r_rec
        bool operator>(const TopPair& other)
        bool operator<(const TopPair& other)

cdef extern from "PrefixEvent.h" nogil:
    cdef cppclass PrefixEvent:
        PrefixEvent()
        PrefixEvent(double, int, int, int)
        double threshold
        int table_indicator
        int rec_idx
        int tok_idx
        bool operator>(const PrefixEvent& other)
        bool operator<(const PrefixEvent& other)

cdef extern from "ReuseInfo.h" nogil:
    cdef cppclass ReuseInfo:
        ReuseInfo()
        ReuseInfo(int)
        int overlap
        umap[int, int] map


####################################################################################################
####################################################################################################
# For new topk sim join. The simplest version. Don't reuse or recording.
cdef heap[TopPair] new_topk_sim_join_plain(const vector[vector[int]]& ltoken_vector,
                                           const vector[vector[int]]& rtoken_vector,
                                           umap[int, uset[int]]& cand_set,
                                           const int prefix_match_max_size,
                                           const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    new_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    cdef heap[TopPair] topk_heap
    new_topk_sim_join_plain_impl(ltoken_vector, rtoken_vector,
                                 cand_set, prefix_events, topk_heap,
                                 prefix_match_max_size, output_size)

    return topk_heap


cdef void new_topk_sim_join_plain_impl(const vector[vector[int]]& ltoken_vector,
                                       const vector[vector[int]]& rtoken_vector,
                                       umap[int, uset[int]]& cand_set, heap[PrefixEvent]& prefix_events,
                                       heap[TopPair]& topk_heap, const int prefix_match_max_size,
                                       const int output_size) nogil:
    printf("in new topk\n")

    cdef uint64_t total_compared_pairs = 0
    cdef uset[uint64_t] total_compared_pairs_set
    cdef umap[int, uset[int]] compared_set
    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef umap[int, umap[int, short]] active_dict
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len, token, overlap
    cdef double sim, threshold
    cdef uint64_t value

    # printf("checkpoint2\n")

    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and \
                (topk_heap.top().sim >= prefix_events.top().threshold or
                 absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        table_indicator = event.table_indicator
        # printf("%.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        if table_indicator == 0:
            l_rec_idx = event.rec_idx
            l_tok_idx = event.tok_idx
            token = ltoken_vector[l_rec_idx][l_tok_idx]
            l_len = ltoken_vector[l_rec_idx].size()
            if r_inverted_index.count(token):
                r_records = r_inverted_index[token]
                for r_rec_tuple in r_records:
                    r_rec_idx = r_rec_tuple.first
                    r_tok_idx = r_rec_tuple.second
                    r_len = rtoken_vector[r_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        # printf("left1\n")
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            overlap += active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)

                        sim = overlap * 1.0 / (l_len + r_len - overlap)

                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     # printf("left2\n")
                    #     overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])
                    #
                    #     if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #         #overlap += active_dict[l_rec_idx][r_rec_idx]
                    #         active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #     sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #     if topk_heap.size() == output_size:
                    #         if topk_heap.top().sim < sim:
                    #             topk_heap.pop()
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #     else:
                    #         topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #     if compared_set.count(l_rec_idx):
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #     else:
                    #         compared_set[l_rec_idx] = uset[int]()
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     total_compared_pairs += 1
                    else:
                        # printf("left3\n")
                        if active_dict.count(l_rec_idx):
                            # printf("left3.1\n")
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                # printf("left3.1.1\n")
                                value = active_dict[l_rec_idx][r_rec_idx]
                                if value == prefix_match_max_size:
                                    # printf("left3.1.1.1\n")
                                    #overlap = value
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])
                                    active_dict[l_rec_idx].erase(r_rec_idx)

                                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                                    if topk_heap.size() == output_size:
                                        if topk_heap.top().sim < sim:
                                            topk_heap.pop()
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                    else:
                                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                    if compared_set.count(l_rec_idx):
                                        compared_set[l_rec_idx].insert(r_rec_idx)
                                    else:
                                        compared_set[l_rec_idx] = uset[int]()
                                        compared_set[l_rec_idx].insert(r_rec_idx)

                                    total_compared_pairs += 1
                                else:
                                    active_dict[l_rec_idx][r_rec_idx] += 1
                            else:
                                # printf("left3.1.2\n")
                                active_dict[l_rec_idx][r_rec_idx] = 1
                        else:
                            active_dict[l_rec_idx] = umap[int, short]()
                            active_dict[l_rec_idx][r_rec_idx] = 1
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)

            if l_tok_idx + 1 < l_len:
                threshold = min(1 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, l_rec_idx, l_tok_idx + 1))

            if not l_inverted_index.count(token):
                l_inverted_index[token] = oset[pair[int, int]]()
            l_inverted_index[token].insert(pair[int, int](l_rec_idx, l_tok_idx))
        else:
            r_rec_idx = event.rec_idx
            r_tok_idx = event.tok_idx
            token = rtoken_vector[r_rec_idx][r_tok_idx]
            r_len = rtoken_vector[r_rec_idx].size()
            if l_inverted_index.count(token):
                l_records = l_inverted_index[token]
                for l_rec_tuple in l_records:
                    l_rec_idx = l_rec_tuple.first
                    l_tok_idx = l_rec_tuple.second
                    l_len = ltoken_vector[l_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        # printf("right1\n")
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            overlap += active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)

                        sim = overlap * 1.0 / (l_len + r_len - overlap)

                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     # printf("right2\n")
                    #     overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])
                    #
                    #     if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #         #overlap += active_dict[l_rec_idx][r_rec_idx]
                    #         active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #     sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #     if topk_heap.size() == output_size:
                    #         if topk_heap.top().sim < sim:
                    #             topk_heap.pop()
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #     else:
                    #         topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #     if compared_set.count(l_rec_idx):
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #     else:
                    #         compared_set[l_rec_idx] = uset[int]()
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     total_compared_pairs += 1
                    else:
                        # printf("right3\n")
                        if active_dict.count(l_rec_idx):
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                value = active_dict[l_rec_idx][r_rec_idx]
                                if value == prefix_match_max_size:
                                    #overlap = value
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])
                                    active_dict[l_rec_idx].erase(r_rec_idx)

                                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                                    if topk_heap.size() == output_size:
                                        if topk_heap.top().sim < sim:
                                            topk_heap.pop()
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                    else:
                                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                    if compared_set.count(l_rec_idx):
                                        compared_set[l_rec_idx].insert(r_rec_idx)
                                    else:
                                        compared_set[l_rec_idx] = uset[int]()
                                        compared_set[l_rec_idx].insert(r_rec_idx)

                                    total_compared_pairs += 1
                                else:
                                    active_dict[l_rec_idx][r_rec_idx] += 1
                            else:
                                active_dict[l_rec_idx][r_rec_idx] = 1
                        else:
                            active_dict[l_rec_idx] = umap[int, short]()
                            active_dict[l_rec_idx][r_rec_idx] = 1
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)

            if r_tok_idx + 1 < r_len:
                threshold = min(1 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, r_rec_idx, r_tok_idx + 1))

            if not r_inverted_index.count(token):
                r_inverted_index[token] = oset[pair[int, int]]()
            r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))
        # printf("finish\n")

    # printf("checkpoint3\n")

    cdef double bound = 1e-6
    if prefix_events.size() > 0:
        bound = prefix_events.top().threshold

    cdef pair[int, umap[int, short]] p1
    cdef pair[int, short] p2
    for p1 in active_dict:
        l_rec_idx = p1.first
        for p2 in p1.second:
            if ltoken_vector[l_rec_idx].size() < (prefix_match_max_size + 1) / bound and\
                    rtoken_vector[p2.first].size() < (prefix_match_max_size + 1) / bound:
                value = p2.second
                sim = value * 1.0 / (ltoken_vector[l_rec_idx].size() + rtoken_vector[p2.first].size() - value)
                if topk_heap.size() == output_size:
                    if topk_heap.top().sim < sim:
                        topk_heap.pop()
                        topk_heap.push(TopPair(sim, l_rec_idx, p2.first))
                else:
                    topk_heap.push(TopPair(sim, l_rec_idx, p2.first))


    printf("number of compared pairs: %ld\n", total_compared_pairs)
    # printf("checkpoint4\n")

    return



cdef int new_plain_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list) nogil:
    cdef int overlap = 0
    cdef uint i

    cdef uset[int] rset
    for i in xrange(rtoken_list.size()):
        rset.insert(rtoken_list[i])

    for i in xrange(ltoken_list.size()):
        if rset.count(ltoken_list[i]):
            overlap += 1

    return overlap


####################################################################################################
####################################################################################################
# For new topk sim join. Only record pre-calculated info but don't reuse.
cdef heap[TopPair] new_topk_sim_join_record(const vector[vector[int]]& ltoken_vector,
                                            const vector[vector[int]]& rtoken_vector,
                                            const vector[vector[int]]& lindex_vector,
                                            const vector[vector[int]]& rindex_vector,
                                            vector[TopPair]& init_topk_list,
                                            umap[int, uset[int]]& cand_set,
                                            umap[int, umap[int, ReuseInfo]]& reuse_set,
                                            const int offset_of_field_num, const int prefix_match_max_size,
                                            const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    new_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    cdef heap[TopPair] topk_heap

    new_topk_sim_join_record_impl(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                  cand_set, reuse_set, prefix_events, topk_heap, init_topk_list,
                                  offset_of_field_num, prefix_match_max_size, output_size)
    return topk_heap


cdef void new_topk_sim_join_record_impl(const vector[vector[int]]& ltoken_vector,
                                        const vector[vector[int]]& rtoken_vector,
                                        const vector[vector[int]]& lindex_vector,
                                        const vector[vector[int]]& rindex_vector,
                                        umap[int, uset[int]]& cand_set,
                                        umap[int, umap[int, ReuseInfo]]& reuse_set,
                                        heap[PrefixEvent]& prefix_events, heap[TopPair]& topk_heap,
                                        const vector[TopPair]& init_topk_list,
                                        const int offset_of_field_num, const int prefix_match_max_size,
                                        const int output_size) nogil:
    # printf("checkpoint1\n")

    cdef uint64_t total_compared_pairs = 0
    cdef uset[uint64_t] total_compared_pairs_set
    cdef umap[int, uset[int]] compared_set
    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef umap[int, umap[int, uint64_t]] active_dict
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len, token, overlap
    cdef ReuseInfo reuse_info
    cdef double sim, threshold, index_thres
    cdef uint64_t value
    cdef int v, p

    cdef double topk_heap_index

    cdef uint64_t COUNT = 0x000000000000000F
    cdef uint64_t *SHIFT_ARRAY = <uint64_t *>malloc(prefix_match_max_size * sizeof(uint64_t))
    cdef int COUNT_BITS = 4
    cdef int FIELD_BITS = (64 - COUNT_BITS) / prefix_match_max_size
    init_shift_array(prefix_match_max_size, FIELD_BITS, COUNT_BITS, SHIFT_ARRAY)
    cdef int INC = 1

    cdef uint64_t bits = 0
    cdef uint64_t bit_results = 0
    cdef uint64_t field_pair = 0

    cdef pair[int, int] temp

    # cdef int cmps[3]
    # cmps[0] = 0
    # cmps[1] = 0
    # cmps[2] = 0

    # printf("checkpoint2\n")

    cdef uint i
    if init_topk_list.size() > 0:
        for i in xrange(init_topk_list.size()):
            topk_heap.push(init_topk_list[i])
            if compared_set.count(init_topk_list[i].l_rec):
                compared_set[init_topk_list[i].l_rec].insert(init_topk_list[i].r_rec)
            else:
                compared_set[init_topk_list[i].l_rec] = uset[int]()
                compared_set[init_topk_list[i].l_rec].insert(init_topk_list[i].r_rec)

    printf("topk heap size: %d\n", topk_heap.size())


    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and\
            (topk_heap.top().sim >= prefix_events.top().threshold or
                 absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        table_indicator = event.table_indicator
        # printf("%.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        if table_indicator == 0:
            l_rec_idx = event.rec_idx
            l_tok_idx = event.tok_idx
            token = ltoken_vector[l_rec_idx][l_tok_idx]
            l_len = ltoken_vector[l_rec_idx].size()
            if r_inverted_index.count(token):
                r_records = r_inverted_index[token]
                for r_rec_tuple in r_records:
                    r_rec_idx = r_rec_tuple.first
                    r_tok_idx = r_rec_tuple.second
                    r_len = rtoken_vector[r_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        # cmps[0] += 1
                        # printf("left1\n")
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            bit_results = active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)
                            overlap += bit_results & COUNT

                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     cmps[1] += 1
                    #     # printf("left2\n")
                    #     reuse_info = ReuseInfo(0)
                    #     # new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #     #                       lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                    #     #                       l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)
                    #     #
                    #     # if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #     #     bit_results = active_dict[l_rec_idx][r_rec_idx]
                    #     #     active_dict[l_rec_idx].erase(r_rec_idx)
                    #     #     value = bit_results & COUNT
                    #     #     reuse_info.overlap += value
                    #     #     for v in xrange(value):
                    #     #         p = (bit_results & SHIFT_ARRAY[v]) >> (COUNT_BITS + FIELD_BITS * v)
                    #     #         if reuse_info.map.count(p):
                    #     #             reuse_info.map[p] += 1
                    #     #         else:
                    #     #             reuse_info.map[p] = 1
                    #
                    #     new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #                           lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                    #                           0, 0, reuse_info, offset_of_field_num)
                    #
                    #     if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #         active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #     overlap = reuse_info.overlap
                    #     sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #     if topk_heap.size() == output_size:
                    #         if topk_heap.top().sim < sim:
                    #             topk_heap.pop()
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #     else:
                    #         topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #     if compared_set.count(l_rec_idx):
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #     else:
                    #         compared_set[l_rec_idx] = uset[int]()
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     if reuse_set.count(l_rec_idx):
                    #         reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                    #     else:
                    #         reuse_set[l_rec_idx] = umap[int, ReuseInfo]()
                    #         reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                    #
                    #     total_compared_pairs += 1
                    else:
                        # printf("left3\n")
                        if active_dict.count(l_rec_idx):
                            # printf("left3.1\n")
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                # printf("left3.1.1\n")
                                value = active_dict[l_rec_idx][r_rec_idx] & COUNT
                                if value == prefix_match_max_size:
                                    # cmps[2] += 1
                                    # printf("left3.1.1.1\n")
                                    reuse_info = ReuseInfo(0)
                                    # new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                    #                       lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                    #                       l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)
                                    # reuse_info.overlap += value
                                    #
                                    # bit_results = active_dict[l_rec_idx][r_rec_idx]
                                    # active_dict[l_rec_idx].erase(r_rec_idx)
                                    # for v in xrange(value):
                                    #     p = (bit_results & SHIFT_ARRAY[v]) >> (COUNT_BITS + FIELD_BITS * v)
                                    #     if reuse_info.map.count(p):
                                    #         reuse_info.map[p] += 1
                                    #     else:
                                    #         reuse_info.map[p] = 1

                                    new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                          lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                                          0, 0, reuse_info, offset_of_field_num)
                                    active_dict[l_rec_idx].erase(r_rec_idx)

                                    overlap = reuse_info.overlap
                                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                                    # if l_rec_idx == 82 and r_rec_idx == 7743:
                                    #     printf("in left\n")
                                    #     printf("overlap %d\n", overlap)
                                    #     printf("sim %.6f\n", sim)

                                    if topk_heap.size() == output_size:
                                        if topk_heap.top().sim < sim:
                                            topk_heap.pop()
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                    else:
                                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                    if compared_set.count(l_rec_idx):
                                        compared_set[l_rec_idx].insert(r_rec_idx)
                                    else:
                                        compared_set[l_rec_idx] = uset[int]()
                                        compared_set[l_rec_idx].insert(r_rec_idx)

                                    if reuse_set.count(l_rec_idx):
                                        reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                                    else:
                                        reuse_set[l_rec_idx] = umap[int, ReuseInfo]()
                                        reuse_set[l_rec_idx][r_rec_idx] = reuse_info

                                    total_compared_pairs += 1
                                else:
                                    bits = active_dict[l_rec_idx][r_rec_idx]
                                    field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                                 rindex_vector[r_rec_idx][r_tok_idx]
                                    bits |= (field_pair << ((COUNT & bits) * FIELD_BITS + COUNT_BITS))
                                    active_dict[l_rec_idx][r_rec_idx] = bits + INC
                            else:
                                # printf("left3.1.2\n")
                                field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                             rindex_vector[r_rec_idx][r_tok_idx]
                                bits = (field_pair << COUNT_BITS) + INC
                                active_dict[l_rec_idx][r_rec_idx] = bits
                        else:
                            # printf("left3.2.1\n")
                            field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                         rindex_vector[r_rec_idx][r_tok_idx]
                            # printf("left3.2.2\n")
                            bits = (field_pair << COUNT_BITS) + INC
                            # printf("left3.2.3\n")
                            active_dict[l_rec_idx] = umap[int, uint64_t]()
                            # printf("left3.2.4\n")
                            active_dict[l_rec_idx][r_rec_idx] = bits
                            # printf("left3.2.5\n")
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)
                            # printf("%d %d %d\n", cmps[0], cmps[1], cmps[2])

            if l_tok_idx + 1 < l_len:
                threshold = min(1 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, l_rec_idx, l_tok_idx + 1))

            # if not l_inverted_index.count(token):
            #     l_inverted_index[token] = oset[pair[int, int]]()
            # l_inverted_index[token].insert(pair[int, int](l_rec_idx, l_tok_idx))

            topk_heap_sim_index = 0.0
            if topk_heap.size() > 0:
                topk_heap_sim_index = topk_heap.top().sim
            index_thres = 0.0
            if l_len + l_tok_idx - prefix_match_max_size <= 0:
                index_thres = 1.0
            else:
                index_thres = (l_len - l_tok_idx + prefix_match_max_size) * 1.0 /\
                              (l_len + l_tok_idx - prefix_match_max_size)
            if index_thres >= topk_heap_sim_index:
                if not l_inverted_index.count(token):
                    l_inverted_index[token] = oset[pair[int, int]]()
                l_inverted_index[token].insert(pair[int, int](l_rec_idx, l_tok_idx))
        else:
            r_rec_idx = event.rec_idx
            r_tok_idx = event.tok_idx
            token = rtoken_vector[r_rec_idx][r_tok_idx]
            r_len = rtoken_vector[r_rec_idx].size()
            if l_inverted_index.count(token):
                l_records = l_inverted_index[token]
                for l_rec_tuple in l_records:
                    l_rec_idx = l_rec_tuple.first
                    l_tok_idx = l_rec_tuple.second
                    l_len = ltoken_vector[l_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        # cmps[0] += 1
                        # printf("right1\n")
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            bit_results = active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)
                            overlap += bit_results & COUNT

                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     cmps[1] += 1
                    #     # printf("right2\n")
                    #     reuse_info = ReuseInfo(0)
                    #     # new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #     #                       lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                    #     #                       l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)
                    #     #
                    #     # if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #     #     bit_results = active_dict[l_rec_idx][r_rec_idx]
                    #     #     active_dict[l_rec_idx].erase(r_rec_idx)
                    #     #     value = bit_results & COUNT
                    #     #     reuse_info.overlap += value
                    #     #     for v in xrange(value):
                    #     #         p = (bit_results & SHIFT_ARRAY[v]) >> (COUNT_BITS + FIELD_BITS * v)
                    #     #         if reuse_info.map.count(p):
                    #     #             reuse_info.map[p] += 1
                    #     #         else:
                    #     #             reuse_info.map[p] = 1
                    #
                    #     new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #                           lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                    #                           0, 0, reuse_info, offset_of_field_num)
                    #
                    #     if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #         active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #     overlap = reuse_info.overlap
                    #     sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #     if topk_heap.size() == output_size:
                    #         if topk_heap.top().sim < sim:
                    #             topk_heap.pop()
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #     else:
                    #         topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #     if compared_set.count(l_rec_idx):
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #     else:
                    #         compared_set[l_rec_idx] = uset[int]()
                    #         compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     if reuse_set.count(l_rec_idx):
                    #         reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                    #     else:
                    #         reuse_set[l_rec_idx] = umap[int, ReuseInfo]()
                    #         reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                    #
                    #     total_compared_pairs += 1
                    else:
                        # printf("right3\n")
                        if active_dict.count(l_rec_idx):
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                value = active_dict[l_rec_idx][r_rec_idx] & COUNT
                                if value == prefix_match_max_size:
                                    # cmps[2] += 1
                                    overlap = value
                                    reuse_info = ReuseInfo(0)
                                    # new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                    #                       lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                    #                       l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)
                                    # reuse_info.overlap += value
                                    #
                                    # bit_results = active_dict[l_rec_idx][r_rec_idx]
                                    # active_dict[l_rec_idx].erase(r_rec_idx)
                                    # for v in xrange(value):
                                    #     p = (bit_results & SHIFT_ARRAY[v]) >> (COUNT_BITS + FIELD_BITS * v)
                                    #     if reuse_info.map.count(p):
                                    #         reuse_info.map[p] += 1
                                    #     else:
                                    #         reuse_info.map[p] = 1

                                    new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                          lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                                          0, 0, reuse_info, offset_of_field_num)
                                    active_dict[l_rec_idx].erase(r_rec_idx)

                                    overlap = reuse_info.overlap
                                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                                    # if l_rec_idx == 82 and r_rec_idx == 7743:
                                    #     printf("in right\n")
                                    #     printf("llen: %d  rlen: %d\n", l_len, r_len)
                                    #     printf("overlap %d\n", overlap)
                                    #     printf("sim %.6f\n", sim)
                                    #     reusetest_map = reuse_info.map
                                    #     for ppp in reusetest_map:
                                    #         printf("%ld %d\n", ppp.first, ppp.second)

                                    if topk_heap.size() == output_size:
                                        if topk_heap.top().sim < sim:
                                            topk_heap.pop()
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                    else:
                                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                    if compared_set.count(l_rec_idx):
                                        compared_set[l_rec_idx].insert(r_rec_idx)
                                    else:
                                        compared_set[l_rec_idx] = uset[int]()
                                        compared_set[l_rec_idx].insert(r_rec_idx)

                                    if reuse_set.count(l_rec_idx):
                                        reuse_set[l_rec_idx][r_rec_idx] = reuse_info
                                    else:
                                        reuse_set[l_rec_idx] = umap[int, ReuseInfo]()
                                        reuse_set[l_rec_idx][r_rec_idx] = reuse_info

                                    total_compared_pairs += 1
                                else:
                                    bits = active_dict[l_rec_idx][r_rec_idx]
                                    field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                                 rindex_vector[r_rec_idx][r_tok_idx]
                                    bits |= (field_pair << ((COUNT & bits) * FIELD_BITS + COUNT_BITS))
                                    active_dict[l_rec_idx][r_rec_idx] = bits + INC
                            else:
                                field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                             rindex_vector[r_rec_idx][r_tok_idx]
                                bits = (field_pair << COUNT_BITS) + INC
                                active_dict[l_rec_idx][r_rec_idx] = bits
                        else:
                            field_pair = lindex_vector[l_rec_idx][l_tok_idx] * offset_of_field_num + \
                                         rindex_vector[r_rec_idx][r_tok_idx]
                            bits = (field_pair << COUNT_BITS) + INC
                            active_dict[l_rec_idx] = umap[int, uint64_t]()
                            active_dict[l_rec_idx][r_rec_idx] = bits
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)
                            # printf("%d %d %d\n", cmps[0], cmps[1],cmps[2])

            if r_tok_idx + 1 < r_len:
                threshold = min(1 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, r_rec_idx, r_tok_idx + 1))

            # if not r_inverted_index.count(token):
            #     r_inverted_index[token] = oset[pair[int, int]]()
            # r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

            topk_heap_sim_index = 0.0
            if topk_heap.size() > 0:
                topk_heap_sim_index = topk_heap.top().sim
            index_thres = 0.0
            if r_len + r_tok_idx - prefix_match_max_size <= 0:
                index_thres = 1.0
            else:
                index_thres = (r_len - r_tok_idx + prefix_match_max_size) * 1.0 /\
                              (r_len + r_tok_idx - prefix_match_max_size)
            if index_thres >= topk_heap_sim_index:
                if not r_inverted_index.count(token):
                    r_inverted_index[token] = oset[pair[int, int]]()
                r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))
        # printf("finish\n")

    # printf("checkpoint3\n")

    cdef double bound = 1e-6
    if prefix_events.size() > 0:
        bound = prefix_events.top().threshold

    cdef pair[int, umap[int, uint64_t]] p1
    cdef pair[int, uint64_t] p2
    for p1 in active_dict:
        l_rec_idx = p1.first
        for p2 in p1.second:
            if ltoken_vector[l_rec_idx].size() < (prefix_match_max_size + 1) / bound and\
                    rtoken_vector[p2.first].size() < (prefix_match_max_size + 1) / bound:
                value = p2.second & COUNT
                sim = value * 1.0 / (ltoken_vector[l_rec_idx].size() + rtoken_vector[p2.first].size() - value)
                if topk_heap.size() == output_size:
                    if topk_heap.top().sim < sim:
                        topk_heap.pop()
                        topk_heap.push(TopPair(sim, l_rec_idx, p2.first))
                else:
                    topk_heap.push(TopPair(sim, l_rec_idx, p2.first))


    printf("number of compared pairs: %ld\n", total_compared_pairs)
    # printf("checkpoint4\n")

    return


cdef void new_reuse_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list,
                                const vector[int]& lindex_list, const vector[int]& rindex_list,
                                const int l_tok_idx, const int r_tok_idx,
                                ReuseInfo& reuse_info, const int offset_of_field_num) nogil:
    cdef int value
    cdef uint i

    cdef umap[int, int] rmap
    for i in xrange(rtoken_list.size() - r_tok_idx):
        rmap[rtoken_list[i + r_tok_idx]] = rindex_list[i + r_tok_idx]

    for i in xrange(ltoken_list.size() - l_tok_idx):
        if rmap.count(ltoken_list[i + l_tok_idx]):
            reuse_info.overlap += 1
            value = lindex_list[i + l_tok_idx] * offset_of_field_num + rmap[ltoken_list[i + l_tok_idx]]
            if reuse_info.map.count(value):
                reuse_info.map[value] += 1
            else:
                reuse_info.map[value] = 1

    return


cdef void init_shift_array(const int num, const int field_bits, const int count_bits,
                           uint64_t* shift_array) nogil:
    cdef int i, j
    cdef uint64_t base = 0
    for i in xrange(field_bits):
        base = base * 2 + 1
    for i in xrange(count_bits):
        base <<= 1

    for i in xrange(num):
        shift_array[i] = base
        for j in xrange(field_bits):
            base <<= 1
    return


####################################################################################################
####################################################################################################
# For new topk sim join. Only reuse pre-calculated info but don't record.
cdef heap[TopPair] new_topk_sim_join_reuse(const vector[vector[int]]& ltoken_vector,
                                           const vector[vector[int]]& rtoken_vector,
                                           const vector[TopPair]& init_topk_list,
                                           uset[int]& remained_fields, umap[int, uset[int]]& cand_set,
                                           umap[int, umap[int, ReuseInfo]]& reuse_set,
                                           const int offset_of_field_num, const int prefix_match_max_size,
                                           const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    new_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    # cdef heap[TopPair] topk_heap
    cdef heap[TopPair] topk_heap

    new_topk_sim_join_reuse_impl(ltoken_vector, rtoken_vector, remained_fields,
                                 cand_set, reuse_set, prefix_events, topk_heap, init_topk_list,
                                 offset_of_field_num, prefix_match_max_size, output_size)

    return topk_heap


cdef void new_topk_sim_join_reuse_impl(const vector[vector[int]]& ltoken_vector,
                                       const vector[vector[int]]& rtoken_vector,
                                       uset[int]& remained_fields, umap[int, uset[int]]& cand_set,
                                       umap[int, umap[int, ReuseInfo]]& reuse_set,
                                       heap[PrefixEvent]& prefix_events, heap[TopPair]& topk_heap,
                                       const vector[TopPair]& init_topk_list,
                                       const int offset_of_field_num, const int prefix_match_max_size,
                                       const int output_size) nogil:
    # printf("checkpoint1\n")
    cdef uint64_t total_compared_pairs = 0
    cdef uset[uint64_t] total_compared_pairs_set
    cdef umap[int, uset[int]] compared_set
    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef umap[int, umap[int, short]] active_dict
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len, token, overlap
    cdef ReuseInfo reuse_info
    cdef double sim, threshold, index_thres
    cdef uint64_t value

    cdef int denom, lfield, rfield
    cdef pair[int, int] field_pair

    cdef int reuse_count = 0
    cdef uint i

    if init_topk_list.size() > 0:
        for i in xrange(init_topk_list.size()):
            topk_heap.push(init_topk_list[i])
            if compared_set.count(init_topk_list[i].l_rec):
                compared_set[init_topk_list[i].l_rec].insert(init_topk_list[i].r_rec)
            else:
                compared_set[init_topk_list[i].l_rec] = uset[int]()
                compared_set[init_topk_list[i].l_rec].insert(init_topk_list[i].r_rec)

    printf("topk heap size: %d\n", topk_heap.size())

    # printf("checkpoint2\n")

    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and\
            (topk_heap.top().sim >= prefix_events.top().threshold or
                 absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        table_indicator = event.table_indicator
        # printf("%.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        if table_indicator == 0:
            l_rec_idx = event.rec_idx
            l_tok_idx = event.tok_idx
            token = ltoken_vector[l_rec_idx][l_tok_idx]
            l_len = ltoken_vector[l_rec_idx].size()
            if r_inverted_index.count(token):
                r_records = r_inverted_index[token]
                for r_rec_tuple in r_records:
                    r_rec_idx = r_rec_tuple.first
                    r_tok_idx = r_rec_tuple.second
                    r_len = rtoken_vector[r_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            overlap += active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)

                        # if l_rec_idx == 3482 and r_rec_idx == 4047:
                        #         printf("left1\n")
                        #         printf("%d %d %d\n", overlap, l_tok_idx, r_tok_idx)

                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                    #         reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                    #         overlap = reuse_info.overlap
                    #         denom = l_len + r_len - overlap
                    #         # if l_rec_idx == 3482 and r_rec_idx == 4047:
                    #         #     printf("left2.1\n")
                    #         #     printf("%d %d %d %d\n", reuse_info.overlap, l_len, r_len, denom)
                    #         #     for field_pair in reuse_info.map:
                    #         #         printf("%d %d ", field_pair.first, field_pair.second)
                    #         #     printf("\n")
                    #         if denom <= 0 or topk_heap.size() < output_size or \
                    #                 overlap * 1.0 / denom > topk_heap.top().sim:
                    #             for field_pair in reuse_info.map:
                    #                 lfield = field_pair.first / offset_of_field_num
                    #                 rfield = field_pair.first % offset_of_field_num
                    #                 if not remained_fields.count(lfield) or not remained_fields.count(rfield):
                    #                     overlap -= field_pair.second
                    #             sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #             if topk_heap.size() == output_size:
                    #                 if topk_heap.top().sim < sim:
                    #                     topk_heap.pop()
                    #                     topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #             else:
                    #                 topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #         if compared_set.count(l_rec_idx):
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #         else:
                    #             compared_set[l_rec_idx] = uset[int]()
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #         if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #             active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #     else:
                    #         # overlap = new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #         #                           l_tok_idx, r_tok_idx)
                    #         overlap = new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #                                   0, 0)
                    #
                    #         # if l_rec_idx == 3482 and r_rec_idx == 4047:
                    #         #     printf("left2.2\n")
                    #         #     printf("%d\n", overlap)
                    #
                    #         if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #             # overlap += active_dict[l_rec_idx][r_rec_idx]
                    #             active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #         sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #         if topk_heap.size() == output_size:
                    #             if topk_heap.top().sim < sim:
                    #                 topk_heap.pop()
                    #                 topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #         else:
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #         if compared_set.count(l_rec_idx):
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #         else:
                    #             compared_set[l_rec_idx] = uset[int]()
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     total_compared_pairs += 1
                    else:
                        if active_dict.count(l_rec_idx):
                            # printf("left3.1\n")
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                # printf("left3.1.1\n")
                                value = active_dict[l_rec_idx][r_rec_idx]
                                if value == prefix_match_max_size:
                                    # printf("left3.1.1.1\n")
                                    if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                                        reuse_count += 1
                                        reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                                        overlap = reuse_info.overlap
                                        denom = l_len + r_len - overlap
                                        # if l_rec_idx == 3482 and r_rec_idx == 4047:
                                        #     printf("left3\n")
                                        #     printf("%d %d %d %d\n", reuse_info.overlap, l_len, r_len, denom)
                                        #     for field_pair in reuse_info.map:
                                        #         printf("%d %d ", field_pair.first, field_pair.second)
                                        #     printf("\n")
                                        if denom <= 0 or topk_heap.size() < output_size or \
                                                overlap * 1.0 / denom > topk_heap.top().sim:
                                            for field_pair in reuse_info.map:
                                                lfield = field_pair.first / offset_of_field_num
                                                rfield = field_pair.first % offset_of_field_num
                                                if not remained_fields.count(lfield) or not remained_fields.count(rfield):
                                                    overlap -= field_pair.second
                                            sim = overlap * 1.0 / (l_len + r_len - overlap)
                                            if topk_heap.size() == output_size:
                                                if topk_heap.top().sim < sim:
                                                    topk_heap.pop()
                                                    topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                            else:
                                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                        if compared_set.count(l_rec_idx):
                                            compared_set[l_rec_idx].insert(r_rec_idx)
                                        else:
                                            compared_set[l_rec_idx] = uset[int]()
                                            compared_set[l_rec_idx].insert(r_rec_idx)

                                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                                            active_dict[l_rec_idx].erase(r_rec_idx)

                                    else:
                                        # overlap = value
                                        overlap = new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                                   0, 0)
                                        active_dict[l_rec_idx].erase(r_rec_idx)

                                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                                        if topk_heap.size() == output_size:
                                            if topk_heap.top().sim < sim:
                                                topk_heap.pop()
                                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                        else:
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                        if compared_set.count(l_rec_idx):
                                            compared_set[l_rec_idx].insert(r_rec_idx)
                                        else:
                                            compared_set[l_rec_idx] = uset[int]()
                                            compared_set[l_rec_idx].insert(r_rec_idx)

                                    total_compared_pairs += 1
                                else:
                                    active_dict[l_rec_idx][r_rec_idx] += 1
                            else:
                                # printf("left3.1.2\n")
                                active_dict[l_rec_idx][r_rec_idx] = 1
                        else:
                            active_dict[l_rec_idx] = umap[int, short]()
                            active_dict[l_rec_idx][r_rec_idx] = 1
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)

            if l_tok_idx + 1 < l_len:
                threshold = min(1 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, l_rec_idx, l_tok_idx + 1))

            # if not l_inverted_index.count(token):
            #     l_inverted_index[token] = oset[pair[int, int]]()
            # l_inverted_index[token].insert(pair[int, int](l_rec_idx, l_tok_idx))

            topk_heap_sim_index = 0.0
            if topk_heap.size() > 0:
                topk_heap_sim_index = topk_heap.top().sim
            index_thres = 0.0
            if l_len + l_tok_idx - prefix_match_max_size <= 0:
                index_thres = 1.0
            else:
                index_thres = (l_len - l_tok_idx + prefix_match_max_size) * 1.0 /\
                              (l_len + l_tok_idx - prefix_match_max_size)
            if index_thres >= topk_heap_sim_index:
                if not l_inverted_index.count(token):
                    l_inverted_index[token] = oset[pair[int, int]]()
                l_inverted_index[token].insert(pair[int, int](l_rec_idx, l_tok_idx))
        else:
            r_rec_idx = event.rec_idx
            r_tok_idx = event.tok_idx
            token = rtoken_vector[r_rec_idx][r_tok_idx]
            r_len = rtoken_vector[r_rec_idx].size()
            if l_inverted_index.count(token):
                l_records = l_inverted_index[token]
                for l_rec_tuple in l_records:
                    l_rec_idx = l_rec_tuple.first
                    l_tok_idx = l_rec_tuple.second
                    l_len = ltoken_vector[l_rec_idx].size()

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if l_tok_idx + 1 == l_len or r_tok_idx + 1 == r_len:
                        overlap = 1
                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                            overlap += active_dict[l_rec_idx][r_rec_idx]
                            active_dict[l_rec_idx].erase(r_rec_idx)

                        # if l_rec_idx == 3482 and r_rec_idx == 4047:
                        #     printf("right1\n")
                        #     printf("%d\n", overlap)

                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                        if topk_heap.size() == output_size:
                            if topk_heap.top().sim < sim:
                                topk_heap.pop()
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                        else:
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                        total_compared_pairs += 1
                    # elif ltoken_vector[l_rec_idx][l_tok_idx + 1] == rtoken_vector[r_rec_idx][r_tok_idx + 1]:
                    #     if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                    #         reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                    #         overlap = reuse_info.overlap
                    #         denom = l_len + r_len - overlap
                    #         # if l_rec_idx == 3482 and r_rec_idx == 4047:
                    #         #     printf("right2.1\n")
                    #         #     printf("%d %d %d %d\n", reuse_info.overlap, l_len, r_len, denom)
                    #         #     for field_pair in reuse_info.map:
                    #         #         printf("%d %d ", field_pair.first, field_pair.second)
                    #         #     printf("\n")
                    #         if denom <= 0 or topk_heap.size() < output_size or \
                    #                 overlap * 1.0 / denom > topk_heap.top().sim:
                    #             for field_pair in reuse_info.map:
                    #                 lfield = field_pair.first / offset_of_field_num
                    #                 rfield = field_pair.first % offset_of_field_num
                    #                 if not remained_fields.count(lfield) or not remained_fields.count(rfield):
                    #                     overlap -= field_pair.second
                    #             sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #             if topk_heap.size() == output_size:
                    #                 if topk_heap.top().sim < sim:
                    #                     topk_heap.pop()
                    #                     topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #             else:
                    #                 topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #         if compared_set.count(l_rec_idx):
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #         else:
                    #             compared_set[l_rec_idx] = uset[int]()
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #         if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #             active_dict[l_rec_idx].erase(r_rec_idx)
                    #     else:
                    #         overlap = new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                    #                                   0, 0)
                    #         # if l_rec_idx == 3482 and r_rec_idx == 4047:
                    #         #     printf("right2.2\n")
                    #         #     printf("%d\n", overlap)
                    #         if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                    #             # overlap += active_dict[l_rec_idx][r_rec_idx]
                    #             active_dict[l_rec_idx].erase(r_rec_idx)
                    #
                    #         sim = overlap * 1.0 / (l_len + r_len - overlap)
                    #         if topk_heap.size() == output_size:
                    #             if topk_heap.top().sim < sim:
                    #                 topk_heap.pop()
                    #                 topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #         else:
                    #             topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    #
                    #         if compared_set.count(l_rec_idx):
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #         else:
                    #             compared_set[l_rec_idx] = uset[int]()
                    #             compared_set[l_rec_idx].insert(r_rec_idx)
                    #
                    #     total_compared_pairs += 1
                    else:
                        if active_dict.count(l_rec_idx):
                            if active_dict[l_rec_idx].count(r_rec_idx):
                                value = active_dict[l_rec_idx][r_rec_idx]
                                if value == prefix_match_max_size:
                                    if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                                        reuse_count += 1
                                        reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                                        overlap = reuse_info.overlap
                                        denom = l_len + r_len - overlap
                                        # if l_rec_idx == 3482 and r_rec_idx == 4047:
                                        #     printf("right3\n")
                                        #     printf("%d %d %d %d\n", reuse_info.overlap, l_len, r_len, denom)
                                        #     for field_pair in reuse_info.map:
                                        #         printf("%d %d ", field_pair.first, field_pair.second)
                                        #     printf("\n")
                                        if denom <= 0 or topk_heap.size() < output_size or \
                                                overlap * 1.0 / denom > topk_heap.top().sim:
                                            for field_pair in reuse_info.map:
                                                lfield = field_pair.first / offset_of_field_num
                                                rfield = field_pair.first % offset_of_field_num
                                                if not remained_fields.count(lfield) or not remained_fields.count(rfield):
                                                    overlap -= field_pair.second
                                            sim = overlap * 1.0 / (l_len + r_len - overlap)
                                            # if l_rec_idx == 82 and r_rec_idx == 7743:
                                            #     printf("in right\n")
                                            #     printf("llen: %d  rlen: %d\n", l_len, r_len)
                                            #     printf("overlap: %d\n", overlap)
                                            #     printf("sim: %.6f\n", sim)
                                            #     reusetest_map = reuse_info.map
                                            #     for ppp in reusetest_map:
                                            #         printf("%ld %d\n", ppp.first, ppp.second)

                                            if topk_heap.size() == output_size:
                                                if topk_heap.top().sim < sim:
                                                    topk_heap.pop()
                                                    topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                            else:
                                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                        if compared_set.count(l_rec_idx):
                                            compared_set[l_rec_idx].insert(r_rec_idx)
                                        else:
                                            compared_set[l_rec_idx] = uset[int]()
                                            compared_set[l_rec_idx].insert(r_rec_idx)

                                        if active_dict.count(l_rec_idx) and active_dict[l_rec_idx].count(r_rec_idx):
                                            active_dict[l_rec_idx].erase(r_rec_idx)
                                    else:
                                        # overlap = value
                                        # overlap += new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                        #                            l_tok_idx, r_tok_idx)
                                        overlap = new_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                                   0, 0)
                                        active_dict[l_rec_idx].erase(r_rec_idx)

                                        sim = overlap * 1.0 / (l_len + r_len - overlap)
                                        if topk_heap.size() == output_size:
                                            if topk_heap.top().sim < sim:
                                                topk_heap.pop()
                                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                                        else:
                                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                                        if compared_set.count(l_rec_idx):
                                            compared_set[l_rec_idx].insert(r_rec_idx)
                                        else:
                                            compared_set[l_rec_idx] = uset[int]()
                                            compared_set[l_rec_idx].insert(r_rec_idx)

                                    total_compared_pairs += 1
                                else:
                                    active_dict[l_rec_idx][r_rec_idx] += 1
                            else:
                                active_dict[l_rec_idx][r_rec_idx] = 1
                        else:
                            active_dict[l_rec_idx] = umap[int, short]()
                            active_dict[l_rec_idx][r_rec_idx] = 1
                    # printf("pass check\n")

                    if total_compared_pairs % 100000 == 0 and \
                            total_compared_pairs_set.count(total_compared_pairs) <= 0:
                        total_compared_pairs_set.insert(total_compared_pairs)
                        if topk_heap.size() > 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx)

            if r_tok_idx + 1 < r_len:
                threshold = min(1 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len, 1.0)
                prefix_events.push(PrefixEvent(threshold, table_indicator, r_rec_idx, r_tok_idx + 1))

            # if not r_inverted_index.count(token):
            #     r_inverted_index[token] = oset[pair[int, int]]()
            # r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

            topk_heap_sim_index = 0.0
            if topk_heap.size() > 0:
                topk_heap_sim_index = topk_heap.top().sim
            index_thres = 0.0
            if r_len + r_tok_idx - prefix_match_max_size <= 0:
                index_thres = 1.0
            else:
                index_thres = (r_len - r_tok_idx + prefix_match_max_size) * 1.0 /\
                              (r_len + r_tok_idx - prefix_match_max_size)
            if index_thres >= topk_heap_sim_index:
                if not r_inverted_index.count(token):
                    r_inverted_index[token] = oset[pair[int, int]]()
                r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

        # printf("finish\n")

    # printf("checkpoint3\n")

    cdef double bound = 1e-6
    if prefix_events.size() > 0:
        bound = prefix_events.top().threshold

    cdef pair[int, umap[int, short]] p1
    cdef pair[int, short] p2
    for p1 in active_dict:
        l_rec_idx = p1.first
        for p2 in p1.second:
            if ltoken_vector[l_rec_idx].size() < (prefix_match_max_size + 1) / bound and\
                    rtoken_vector[p2.first].size() < (prefix_match_max_size + 1) / bound:
                value = p2.second
                sim = value * 1.0 / (ltoken_vector[l_rec_idx].size() + rtoken_vector[p2.first].size() - value)
                if topk_heap.size() == output_size:
                    if topk_heap.top().sim < sim:
                        topk_heap.pop()
                        topk_heap.push(TopPair(sim, l_rec_idx, p2.first))
                else:
                    topk_heap.push(TopPair(sim, l_rec_idx, p2.first))


    printf("number of compared pairs: %ld\n", total_compared_pairs)
    printf("number of reused pairs: %d\n", reuse_count)
    # printf("checkpoint4\n")

    return


cdef int new_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list,
                         const int l_tok_idx, const int r_tok_idx) nogil:
    cdef int overlap = 0
    cdef uint i

    cdef uset[int] rset
    for i in xrange(rtoken_list.size() - r_tok_idx):
        rset.insert(rtoken_list[i + r_tok_idx])

    for i in xrange(ltoken_list.size() - l_tok_idx):
        if rset.count(ltoken_list[i + l_tok_idx]):
            overlap += 1

    return overlap


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
cdef void new_generate_prefix_events(const vector[vector[int]]& ltable,
                                 const vector[vector[int]]& rtable,
                                 heap[PrefixEvent]& prefix_events) nogil:
    new_generate_prefix_events_impl(ltable, 0, prefix_events)
    new_generate_prefix_events_impl(rtable, 1, prefix_events)

    return


cdef void new_generate_prefix_events_impl(const vector[vector[int]]& table,
                                      const int table_indicator,
                                      heap[PrefixEvent]& prefix_events) nogil:
    cdef uint i, length
    for i in xrange(table.size()):
        length = table[i].size()
        if length > 0:
            prefix_events.push(PrefixEvent(1.0, table_indicator, i, 0))

    return

cdef double absdiff(double a, double b) nogil:
    cdef double v = a - b
    if v < 0:
        v = v * -1
    return v
