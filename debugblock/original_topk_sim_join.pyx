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
# For original topk sim join. The simplest version as described in the paper. Don't reuse or
# recording.
cdef heap[TopPair] original_topk_sim_join_plain(const vector[vector[int]]& ltoken_vector,
                                                const vector[vector[int]]& rtoken_vector,
                                                umap[int, uset[int]]& cand_set,
                                                const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    original_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    cdef heap[TopPair] topk_heap
    original_topk_sim_join_plain_impl(ltoken_vector, rtoken_vector, cand_set, prefix_events,
                                      topk_heap, output_size)
    return topk_heap


cdef void original_topk_sim_join_plain_impl(const vector[vector[int]]& ltoken_vector,
                                            const vector[vector[int]]& rtoken_vector,
                                            umap[int, uset[int]]& cand_set, heap[PrefixEvent]& prefix_events,
                                            heap[TopPair]& topk_heap, const int output_size) nogil:
    cdef uint64_t total_compared_pairs = 0
    cdef umap[int, uset[int]] compared_set

    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len
    cdef int token, overlap

    cdef double sim
    # printf("checkpoint2\n")

    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and\
                (topk_heap.top().sim >= prefix_events.top().threshold or
                 ori_absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        # printf("%0.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        table_indicator = event.table_indicator
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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])

                    sim = overlap * 1.0 / (l_len + r_len - overlap)

                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

                    if compared_set.count(l_rec_idx):
                        compared_set[l_rec_idx].insert(r_rec_idx)
                    else:
                        compared_set[l_rec_idx] = uset[int]()
                        compared_set[l_rec_idx].insert(r_rec_idx)

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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx])

                    sim = overlap * 1.0 / (l_len + r_len - overlap)

                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

                    if compared_set.count(l_rec_idx):
                        compared_set[l_rec_idx].insert(r_rec_idx)
                    else:
                        compared_set[l_rec_idx] = uset[int]()
                        compared_set[l_rec_idx].insert(r_rec_idx)

            if not r_inverted_index.count(token):
                r_inverted_index[token] = oset[pair[int, int]]()
            r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

    printf("number of compared pairs: %ld\n", total_compared_pairs)
    # printf("checkpoint3\n")

    return


cdef int original_plain_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list) nogil:
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
# For original topk sim join. Only record pre-calculated info but don't reuse.
cdef heap[TopPair] original_topk_sim_join_record(const vector[vector[int]]& ltoken_vector,
                                                 const vector[vector[int]]& rtoken_vector,
                                                 const vector[vector[int]]& lindex_vector,
                                                 const vector[vector[int]]& rindex_vector,
                                                 umap[int, uset[int]]& cand_set,
                                                 umap[int, umap[int, ReuseInfo]]& reuse_set,
                                                 const int offset_of_field_num,
                                                 const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    original_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    cdef heap[TopPair] topk_heap
    original_topk_sim_join_record_impl(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                       cand_set, reuse_set, prefix_events, topk_heap,
                                       offset_of_field_num, output_size)

    return topk_heap


cdef void original_topk_sim_join_record_impl(const vector[vector[int]]& ltoken_vector,
                                             const vector[vector[int]]& rtoken_vector,
                                             const vector[vector[int]]& lindex_vector,
                                             const vector[vector[int]]& rindex_vector,
                                             umap[int, uset[int]]& cand_set,
                                             umap[int, umap[int, ReuseInfo]]& reuse_set,
                                             heap[PrefixEvent]& prefix_events, heap[TopPair]& topk_heap,
                                             const int offset_of_field_num, const int output_size) nogil:
    cdef uint64_t total_compared_pairs = 0
    cdef umap[int, uset[int]] compared_set

    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len, token
    cdef ReuseInfo reuse_info

    cdef double sim

    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and\
            (topk_heap.top().sim >= prefix_events.top().threshold or
                 ori_absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        # printf("%0.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        table_indicator = event.table_indicator
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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    reuse_info = ReuseInfo(0)
                    original_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                               lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                               l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)

                    sim = reuse_info.overlap * 1.0 / (l_len + r_len - reuse_info.overlap)
                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue


                    reuse_info = ReuseInfo(0)
                    original_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                               lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                               l_tok_idx, r_tok_idx, reuse_info, offset_of_field_num)

                    sim = reuse_info.overlap * 1.0 / (l_len + r_len - reuse_info.overlap)
                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

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

            if not r_inverted_index.count(token):
                r_inverted_index[token] = oset[pair[int, int]]()
            r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

    printf("number of compared pairs: %ld\n", total_compared_pairs)

    return


cdef void original_reuse_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list,
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

####################################################################################################
####################################################################################################
# For original topk sim join. Only reuse pre-calculated info but don't record.
cdef heap[TopPair] original_topk_sim_join_reuse(const vector[vector[int]]& ltoken_vector,
                                       const vector[vector[int]]& rtoken_vector,
                                       uset[int]& remained_fields, umap[int, uset[int]]& cand_set,
                                       umap[int, umap[int, ReuseInfo]]& reuse_set,
                                       const int offset_of_field_num,
                                       const int output_size) nogil:
    cdef heap[PrefixEvent] prefix_events
    original_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events)

    cdef heap[TopPair] topk_heap
    original_topk_sim_join_reuse_impl(ltoken_vector, rtoken_vector, remained_fields, cand_set,
                                      reuse_set, prefix_events, topk_heap,
                                      offset_of_field_num, output_size)

    return topk_heap


cdef void original_topk_sim_join_reuse_impl(const vector[vector[int]]& ltoken_vector,
                                            const vector[vector[int]]& rtoken_vector,
                                            uset[int]& remained_fields, umap[int, uset[int]]& cand_set,
                                            umap[int, umap[int, ReuseInfo]]& reuse_set,
                                            heap[PrefixEvent]& prefix_events, heap[TopPair]& topk_heap,
                                            const int offset_of_field_num, const int output_size) nogil:
    # printf("checkpoint1\n")
    cdef uint64_t total_compared_pairs = 0
    cdef umap[int, uset[int]] compared_set

    cdef umap[int, oset[pair[int, int]]] l_inverted_index
    cdef umap[int, oset[pair[int, int]]] r_inverted_index
    cdef oset[pair[int, int]] l_records, r_records
    cdef pair[int, int] l_rec_tuple, r_rec_tuple
    # cdef heap[TopPair] topk_heap
    cdef PrefixEvent event
    cdef int table_indicator, l_rec_idx, l_tok_idx, r_rec_idx, r_tok_idx, l_len, r_len
    cdef int token, overlap, denom
    cdef ReuseInfo reuse_info

    cdef double sim
    # printf("checkpoint2\n")

    while prefix_events.size() > 0:
        if topk_heap.size() == output_size and\
                (topk_heap.top().sim >= prefix_events.top().threshold or
                 ori_absdiff(topk_heap.top().sim, prefix_events.top().threshold) <= 1e-6):
            break
        event = prefix_events.top()
        prefix_events.pop()
        # printf("%0.6f %d %d %d\n", event.threshold, event.table_indicator, event.rec_idx, event.tok_idx)
        table_indicator = event.table_indicator
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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                        reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                        overlap = reuse_info.overlap
                        denom = l_len + r_len - overlap
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

                        total_compared_pairs += 1
                        if total_compared_pairs % 100000 == 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)
                        continue

                    overlap = original_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                   l_tok_idx, r_tok_idx)

                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

                    if compared_set.count(l_rec_idx):
                        compared_set[l_rec_idx].insert(r_rec_idx)
                    else:
                        compared_set[l_rec_idx] = uset[int]()
                        compared_set[l_rec_idx].insert(r_rec_idx)

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

                    if topk_heap.size() > 0 and \
                            (l_len < topk_heap.top().sim * r_len or l_len > r_len / topk_heap.top().sim):
                        continue

                    if cand_set.count(l_rec_idx) and cand_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if compared_set.count(l_rec_idx) and compared_set[l_rec_idx].count(r_rec_idx):
                        continue

                    if reuse_set.count(l_rec_idx) and reuse_set[l_rec_idx].count(r_rec_idx):
                        reuse_info = reuse_set[l_rec_idx][r_rec_idx]
                        overlap = reuse_info.overlap
                        denom = l_len + r_len - overlap
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

                        total_compared_pairs += 1
                        if total_compared_pairs % 100000 == 0:
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)
                        continue

                    overlap = original_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                   l_tok_idx, r_tok_idx)
                    sim = overlap * 1.0 / (l_len + r_len - overlap)
                    if topk_heap.size() == output_size:
                        if topk_heap.top().sim < sim:
                            topk_heap.pop()
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))
                    else:
                        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx))

                    total_compared_pairs += 1
                    if total_compared_pairs % 100000 == 0:
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx)

                    if compared_set.count(l_rec_idx):
                        compared_set[l_rec_idx].insert(r_rec_idx)
                    else:
                        compared_set[l_rec_idx] = uset[int]()
                        compared_set[l_rec_idx].insert(r_rec_idx)

            if not r_inverted_index.count(token):
                r_inverted_index[token] = oset[pair[int, int]]()
            r_inverted_index[token].insert(pair[int, int](r_rec_idx, r_tok_idx))

    printf("number of compared pairs: %ld\n", total_compared_pairs)
    # printf("checkpoint3\n")

    return


cdef int original_get_overlap(const vector[int]& ltoken_list, const vector[int]& rtoken_list,
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
####################################################################################################
cdef void original_generate_prefix_events(const vector[vector[int]]& ltable,
                                 const vector[vector[int]]& rtable,
                                 heap[PrefixEvent]& prefix_events) nogil:
    original_generate_prefix_events_impl(ltable, 0, prefix_events)
    original_generate_prefix_events_impl(rtable, 1, prefix_events)
    return


cdef void original_generate_prefix_events_impl(const vector[vector[int]]& table,
                                      const int table_indicator,
                                      heap[PrefixEvent]& prefix_events) nogil:
    cdef uint i, length
    cdef int j
    for i in xrange(table.size()):
        length = table[i].size()
        if length > 0:
            for j in xrange(length):
                prefix_events.push(PrefixEvent(1.0 - j * 1.0 / length, table_indicator, i, j))

    return


cdef double ori_absdiff(double a, double b) nogil:
    cdef double v = a - b
    if v < 0:
        v = v * -1
    return v