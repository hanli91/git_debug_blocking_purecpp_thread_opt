#include "TopkHeader.h"


void inline update_topk_heap(Heap& topk_heap, double sim, int l_rec_idx, int r_rec_idx, int output_size) {
    if (topk_heap.size() == output_size) {
        if (topk_heap.top().sim < sim) {
            topk_heap.pop();
            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
        }
    } else {
        topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
    }
}


void inline update_compared_set(unordered_map<int, unordered_set<int>>& compared_set,
                                int l_rec_idx, int r_rec_idx) {
    if (!compared_set.count(l_rec_idx)) {
        compared_set[l_rec_idx] = unordered_set<int>();
    }
    compared_set[l_rec_idx].insert(r_rec_idx);
}


int inline reuse_calculation(const vector<int>& remained_fields, const vector<int>& removed_fields,
                             const int max_field_num, const bool use_remain, ReuseInfoArray& reuse_info) {
    if (use_remain) {
        int overlap = 0;
        int size = remained_fields.size();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                overlap += reuse_info.info[remained_fields[i]][remained_fields[j]];
            }
        }
        return overlap;
    } else {
        int overlap = reuse_info.overlap;
        int size = removed_fields.size();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < max_field_num; ++j) {
                overlap -= reuse_info.info[removed_fields[i]][j];
                overlap -= reuse_info.info[j][removed_fields[i]];
            }
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                overlap += reuse_info.info[removed_fields[i]][removed_fields[j]];
            }
        }

        return overlap;
    }
}

void inline precalculate_topk_heap(vector<Heap>& heap_vector, ReuseInfoArray& reuse_info, const vector<int>& field_list,
                                   const vector<int>& lfield_count, const vector<int>& rfield_count,
                                   int overlap, int l_len, int r_len, int max_field_num,
                                   int l_rec_idx, int r_rec_idx, int output_size) {
    for (int i = 0; i < field_list.size(); ++i) {
        int field = field_list[i];
        int new_l_len = l_len - lfield_count[field];
        int new_r_len = r_len - rfield_count[field];
        int new_overlap = overlap;
        for (int j = 0; j < max_field_num; ++j) {
            new_overlap -= reuse_info.info[field][j];
            new_overlap -= reuse_info.info[j][field];
        }
        new_overlap += reuse_info.info[field][field];

        double sim = new_overlap * 1.0 / (new_l_len + new_r_len - new_overlap);
        update_topk_heap(heap_vector[i], sim, l_rec_idx, r_rec_idx, output_size);
    }
}

void inline update_reuse_set(vector<Heap>& heap_vector, ReuseSet& reuse_set, ReuseInfoArray& reuse_info,
                             const vector<int>& field_list, const vector<int>& lfield_count, const vector<int>& rfield_count,
                             int max_field_num, int l_len, int r_len, int l_rec_idx, int r_rec_idx, int output_size) {
    if (!reuse_set.count(l_rec_idx)) {
        reuse_set[l_rec_idx] = unordered_map<int, ReuseInfoArray>();
    }
    reuse_set[l_rec_idx][r_rec_idx] = reuse_info;

    for (int i = 0; i < field_list.size(); ++i) {
        int field = field_list[i];
        int new_l_len = l_len - lfield_count[field];
        int new_r_len = r_len - rfield_count[field];
        int overlap = reuse_info.overlap;
        for (int j = 0; j < max_field_num; ++j) {
            overlap -= reuse_info.info[field][j];
            overlap -= reuse_info.info[j][field];
        }
        overlap += reuse_info.info[field][field];

        double sim = overlap * 1.0 / (new_l_len + new_r_len - overlap);
        update_topk_heap(heap_vector[i], sim, l_rec_idx, r_rec_idx, output_size);
    }
}


void new_topk_sim_join_record_impl(const Table& ltoken_vector, const Table& rtoken_vector,
                                   const Table& lindex_vector, const Table& rindex_vector,
                                   const Table& lfield_vector, const Table& rfield_vector,
                                   const vector<int>& field_list, const vector<int>& removed_fields,
                                   const int max_field_num, const bool use_remain,
                                   CandSet& cand_set, ReuseSet& reuse_set, PrefixHeap& prefix_events,
                                   Heap& topk_heap, vector<Heap>& heap_vector, const vector<TopPair>& init_topk_list,
                                   const int prefix_match_max_size, const int output_size) {
    long int total_compared_pairs = 0;
    unordered_set<long int> total_compared_pairs_set;

    unordered_map<int, unordered_set<int>> compared_set;
    unordered_map<int ,unordered_map<int, short>> active_dict;

    InvertedIndex l_inverted_index, r_inverted_index;

    long int reuse_count = 0;

    if (init_topk_list.size() > 0) {
        for (int i = 0; i < init_topk_list.size(); ++i) {
            topk_heap.push(init_topk_list[i]);
            if (!compared_set.count(init_topk_list[i].l_rec)) {
                compared_set[init_topk_list[i].l_rec] = unordered_set<int> ();
            }
            compared_set[init_topk_list[i].l_rec].insert(init_topk_list[i].r_rec);
        }
    }
    cout << "topk heap size: " << topk_heap.size() << endl;

    int init_cmp_set_count = 0;
    for (unordered_map<int, unordered_set<int>>::iterator it = compared_set.begin(); it != compared_set.end(); ++it) {
        init_cmp_set_count += (it->second).size();
    }
//    for (pair<int, unordered_set<int>> pair : compared_set) {
//        init_cmp_set_count += pair.second.size();
//    }
    cout << "********compare set size: " << init_cmp_set_count << endl;

    while (prefix_events.size() > 0) {
        if (topk_heap.size() == output_size && (topk_heap.top().sim >= prefix_events.top().threshold)) {
            break;
        }
        PrefixEvent event = prefix_events.top();
        prefix_events.pop();
        int table_indicator = event.table_indicator;
        if (table_indicator == 0) {
            int l_rec_idx = event.rec_idx;
            int l_tok_idx = event.tok_idx;
            int token = ltoken_vector[l_rec_idx][l_tok_idx];
            unsigned long l_len = ltoken_vector[l_rec_idx].size();
            if (r_inverted_index.count(token)) {
                set<pair<int, int>> r_records = r_inverted_index[token];
//                for (auto r_rec_tuple : r_records) {
                for (set<pair<int, int>>::iterator it = r_records.begin(); it != r_records.end(); ++it) {
                    pair<int, int> r_rec_tuple = *it;
                    int r_rec_idx = r_rec_tuple.first;
                    int r_tok_idx = r_rec_tuple.second;
                    unsigned long r_len = rtoken_vector[r_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        int overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        double sim = overlap * 1.0 / (l_len + r_len - overlap);
                        if (topk_heap.size() == output_size) {
                            if (topk_heap.top().sim < sim) {
                                topk_heap.pop();
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
                            }
                        } else {
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
                        }
                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                int value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    if (reuse_set.count(l_rec_idx) && reuse_set[l_rec_idx].count(r_rec_idx)) {
                                        ++reuse_count;
                                        ReuseInfoArray reuse_info = reuse_set[l_rec_idx][r_rec_idx];
                                        int overlap = reuse_info.overlap;
                                        int denom = (int)l_len + (int)r_len - overlap;
                                        if (denom <= 0 || topk_heap.size() < output_size ||
                                            overlap * 1.0 / denom > topk_heap.top().sim) {
                                            int new_overlap = reuse_calculation(field_list, removed_fields,
                                                                                max_field_num, use_remain, reuse_info);

                                            double sim = new_overlap * 1.0 / (l_len + r_len - new_overlap);
                                            update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                            update_compared_set(compared_set, l_rec_idx, r_rec_idx);
                                            precalculate_topk_heap(heap_vector, reuse_info, field_list,
                                                                   lfield_vector[l_rec_idx], rfield_vector[r_rec_idx],
                                                                   new_overlap, l_len, r_len, max_field_num, l_rec_idx, r_rec_idx, output_size);
                                        }
                                    } else {
                                        ReuseInfoArray reuse_info = ReuseInfoArray(0);
                                        new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                              lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                                              reuse_info);
                                        active_dict[l_rec_idx].erase(r_rec_idx);
                                        double sim = reuse_info.overlap * 1.0 / (l_len + r_len - reuse_info.overlap);
                                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                        update_compared_set(compared_set, l_rec_idx, r_rec_idx);
                                        // update_reuse_set(reuse_set, reuse_info, l_rec_idx, r_rec_idx);
                                        update_reuse_set(heap_vector, reuse_set, reuse_info,
                                                         field_list, lfield_vector[l_rec_idx], rfield_vector[r_rec_idx],
                                                         max_field_num, l_len, r_len, l_rec_idx, r_rec_idx, output_size);
                                        ++total_compared_pairs;
                                    }
                                } else {
                                    ++active_dict[l_rec_idx][r_rec_idx];
                                }
                            } else {
                                active_dict[l_rec_idx][r_rec_idx] = 1;
                            }
                        } else {
                            active_dict[l_rec_idx] = unordered_map<int, short> ();
                            active_dict[l_rec_idx][r_rec_idx] = 1;
                        }
                    }

                    if (total_compared_pairs % 100000 == 0 &&
                        !total_compared_pairs_set.count(total_compared_pairs)) {
                        total_compared_pairs_set.insert(total_compared_pairs);
                        if (topk_heap.size() > 0) {
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                        }
                    }
                }
            }

            if (l_tok_idx + 1 < l_len) {
                double threshold = min(1.0, 1.0 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len);
                prefix_events.push(PrefixEvent(threshold, table_indicator, l_rec_idx, l_tok_idx + 1));
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }

            double index_threshold = 1.0;
            int numer_index = l_len - l_tok_idx + prefix_match_max_size;
            int denom_index = l_len + l_tok_idx - prefix_match_max_size;
            if (denom_index > 0) {
                index_threshold = numer_index * 1.0 / denom_index;
            }

            if (index_threshold >= topk_heap_sim_index) {
                if (!l_inverted_index.count(token)) {
                    l_inverted_index[token] = set<pair<int, int>>();
                }
                l_inverted_index[token].insert(pair<int, int>(l_rec_idx, l_tok_idx));
            }

//            if (!l_inverted_index.count(token)) {
//                l_inverted_index[token] = set<pair<int, int>>();
//            }
//            l_inverted_index[token].insert(pair<int, int>(l_rec_idx, l_tok_idx));
        } else {
            int r_rec_idx = event.rec_idx;
            int r_tok_idx = event.tok_idx;
            int token = rtoken_vector[r_rec_idx][r_tok_idx];
            unsigned long r_len = rtoken_vector[r_rec_idx].size();

            if (l_inverted_index.count(token)) {
                set<pair<int, int>> l_records = l_inverted_index[token];
//                for (auto l_rec_tuple : l_records) {
                for (set<pair<int, int>>::iterator it = l_records.begin(); it != l_records.end(); ++it) {
                    pair<int, int> l_rec_tuple = *it;
                    int l_rec_idx = l_rec_tuple.first;
                    int l_tok_idx = l_rec_tuple.second;
                    unsigned long l_len = ltoken_vector[l_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        int overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        double sim = overlap * 1.0 / (l_len + r_len - overlap);
                        if (topk_heap.size() == output_size) {
                            if (topk_heap.top().sim < sim) {
                                topk_heap.pop();
                                topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
                            }
                        } else {
                            topk_heap.push(TopPair(sim, l_rec_idx, r_rec_idx));
                        }
                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                int value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    if (reuse_set.count(l_rec_idx) && reuse_set[l_rec_idx].count(r_rec_idx)) {
                                        ++reuse_count;
                                        ReuseInfoArray reuse_info = reuse_set[l_rec_idx][r_rec_idx];
                                        int overlap = reuse_info.overlap;
                                        int denom = (int)l_len + (int)r_len - overlap;
                                        if (denom <= 0 || topk_heap.size() < output_size ||
                                            overlap * 1.0 / denom > topk_heap.top().sim) {
                                            int new_overlap = reuse_calculation(field_list, removed_fields,
                                                                                max_field_num, use_remain, reuse_info);

                                            double sim = new_overlap * 1.0 / (l_len + r_len - new_overlap);
                                            update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                            update_compared_set(compared_set, l_rec_idx, r_rec_idx);
                                            precalculate_topk_heap(heap_vector, reuse_info, field_list,
                                                                   lfield_vector[l_rec_idx], rfield_vector[r_rec_idx],
                                                                   new_overlap, l_len, r_len, max_field_num, l_rec_idx, r_rec_idx, output_size);
                                        }
                                    } else {
                                        ReuseInfoArray reuse_info = ReuseInfoArray(0);
                                        new_reuse_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx],
                                                              lindex_vector[l_rec_idx], rindex_vector[r_rec_idx],
                                                              reuse_info);
                                        active_dict[l_rec_idx].erase(r_rec_idx);
                                        double sim = reuse_info.overlap * 1.0 / (l_len + r_len - reuse_info.overlap);
                                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                        update_compared_set(compared_set, l_rec_idx, r_rec_idx);
                                        // update_reuse_set(reuse_set, reuse_info, l_rec_idx, r_rec_idx);
                                        update_reuse_set(heap_vector, reuse_set, reuse_info,
                                                         field_list, lfield_vector[l_rec_idx], rfield_vector[r_rec_idx],
                                                         max_field_num, l_len, r_len, l_rec_idx, r_rec_idx, output_size);
                                        ++total_compared_pairs;
                                    }
                                } else {
                                    ++active_dict[l_rec_idx][r_rec_idx];
                                }
                            } else {
                                active_dict[l_rec_idx][r_rec_idx] = 1;
                            }
                        } else {
                            active_dict[l_rec_idx] = unordered_map<int, short> ();
                            active_dict[l_rec_idx][r_rec_idx] = 1;
                        }
                    }

                    if (total_compared_pairs % 100000 == 0 &&
                        !total_compared_pairs_set.count(total_compared_pairs)) {
                        total_compared_pairs_set.insert(total_compared_pairs);
                        if (topk_heap.size() > 0) {
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec, topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                        }
                    }
                }
            }

            if (r_tok_idx + 1 < r_len) {
                double threshold = min(1.0, 1.0 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len);
                prefix_events.push(PrefixEvent(threshold, table_indicator, r_rec_idx, r_tok_idx + 1));
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }
            double index_threshold = 1.0;
            int numer_index = r_len - r_tok_idx + prefix_match_max_size;
            int denom_index = r_len + r_tok_idx - prefix_match_max_size;
            if (denom_index > 0) {
                index_threshold = numer_index * 1.0 / denom_index;
            }
            if (index_threshold >= topk_heap_sim_index) {
                if (!r_inverted_index.count(token)) {
                    r_inverted_index[token] = set<pair<int, int>>();
                }
                r_inverted_index[token].insert(pair<int, int>(r_rec_idx, r_tok_idx));
            }

//            if (!r_inverted_index.count(token)) {
//                r_inverted_index[token] = set<pair<int, int>>();
//            }
//            r_inverted_index[token].insert(pair<int, int>(r_rec_idx, r_tok_idx));
        }
    }

    double bound = 1e-6;
    if (prefix_events.size() > 0) {
        bound = prefix_events.top().threshold;
    }
    int len_upper_bound = (prefix_match_max_size + 1) / bound;
    for (unordered_map<int ,unordered_map<int, short>>::iterator it = active_dict.begin();
            it != active_dict.end(); ++it) {
        int l_rec_idx = it->first;
        unsigned long l_len = ltoken_vector[l_rec_idx].size();
        unordered_map<int, short> temp_map = it->second;
        for (unordered_map<int, short>::iterator it2 = temp_map.begin(); it2 != temp_map.end(); ++it2) {
            unsigned long r_len = rtoken_vector[it2->first].size();
            if (l_len < len_upper_bound && r_len < len_upper_bound) {
                int value = it2->second;
                double sim = value * 1.0 / (l_len + r_len - value);
                update_topk_heap(topk_heap, sim, l_rec_idx, it2->first, output_size);
            }
        }
    }
//    for (auto p1 : active_dict) {
//        int l_rec_idx = p1.first;
//        unsigned long l_len = ltoken_vector[l_rec_idx].size();
//        for (auto p2 : p1.second) {
//            int r_len = rtoken_vector[p2.first].size();
//            if (l_len < len_upper_bound && r_len < len_upper_bound) {
//                int value = p2.second;
//                double sim = value * 1.0 / (l_len + r_len - value);
//                update_topk_heap(topk_heap, sim, l_rec_idx, p2.first, output_size);
//            }
//        }
//    }
    printf("number of compared pairs: %ld\n", total_compared_pairs);
}

Heap new_topk_sim_join_record(const Table& ltoken_vector, const Table& rtoken_vector,
                              const Table& lindex_vector, const Table& rindex_vector,
                              const Table& lfield_vector, const Table& rfield_vector,
                              const vector<int>& field_list, const vector<int>& removed_fields,
                              const int max_field_num, const bool use_remain,
                              vector<Heap>& heap_vector, vector<TopPair>& init_topk_list,
                              CandSet& cand_set, ReuseSet& reuse_set, const int prefix_match_max_size,
                              const int output_size) {
    cout << "In new topk sim record" << endl;

    PrefixHeap prefix_events;
    new_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events);

    Heap topk_heap;
    new_topk_sim_join_record_impl(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                  lfield_vector, rfield_vector, field_list, removed_fields,
                                  max_field_num, use_remain, cand_set, reuse_set, prefix_events,
                                  topk_heap, heap_vector, init_topk_list, prefix_match_max_size, output_size);

    for (int i = 0; i < heap_vector.size(); ++i) {
        Heap temp = heap_vector[i];
        unordered_map<int, unordered_set<int>> temp_set;

        int count = 0;
        while (!temp.empty()) {
            TopPair tp = temp.top();
            if (!temp_set.count(tp.l_rec)) {
                temp_set[tp.l_rec] = unordered_set<int> ();
            }
            if (temp_set[tp.l_rec].count(tp.r_rec)) {
                ++count;
            }

            temp_set[tp.l_rec].insert(tp.r_rec);
            temp.pop();
        }
        cout << "heap vector " << i << " size:" << heap_vector[i].size() <<
                                    "  " << "num of dups: " << count << endl;
    }

    return topk_heap;
}
