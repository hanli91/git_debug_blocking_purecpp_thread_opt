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


void new_topk_sim_join_plain_first_impl(const Table& ltoken_vector, const Table& rtoken_vector,
                                        Heap& topk_heap, atomic<int>& best_q, CandSet& cand_set,
                                        PrefixHeap& prefix_events, const int prefix_match_max_size,
                                        const int select_param_size, const int output_size) {
    long int total_compared_pairs = 0;
    unordered_set<long int> total_compared_pairs_set;

    unordered_map<int, unordered_set<int>> compared_set;
    unordered_map<int ,unordered_map<int, short>> active_dict;

    InvertedIndex l_inverted_index, r_inverted_index;
    PrefixEvent event;
    int table_indicator, l_rec_idx, r_rec_idx, l_tok_idx, r_tok_idx, token, overlap, value;
    unsigned long l_len, r_len;
    double sim, threshold;

    Heap param_topk_heap;

//    double topk_heap_sim_index, index_threshold;

    // Select the prefix parameter.
    while (prefix_events.size() > 0) {
        if (param_topk_heap.size() == select_param_size && (param_topk_heap.top().sim >= prefix_events.top().threshold)) {
//            cout << "Select parameter: " << prefix_match_max_size << endl;
            printf("Select parameter: %d\n", prefix_match_max_size);
            best_q = prefix_match_max_size;
            break;
        }
        if (best_q != -1 && best_q != prefix_match_max_size) {
//            cout << "Break parameter: " << prefix_match_max_size << endl;
            printf("Break job on parameter: %d\n", prefix_match_max_size);
            return;
        }

        event = prefix_events.top();
        prefix_events.pop();
        table_indicator = event.table_indicator;
        if (table_indicator == 0) {
            l_rec_idx = event.rec_idx;
            l_tok_idx = event.tok_idx;
            token = ltoken_vector[l_rec_idx][l_tok_idx];
            l_len = ltoken_vector[l_rec_idx].size();
            if (r_inverted_index.count(token)) {
                set<pair<int, int>> r_records = r_inverted_index[token];
                for (set<pair<int, int>>::iterator it = r_records.begin(); it != r_records.end(); ++it) {
                    pair<int, int> r_rec_tuple = *it;
                    r_rec_idx = r_rec_tuple.first;
                    r_tok_idx = r_rec_tuple.second;
                    r_len = rtoken_vector[r_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        sim = overlap * 1.0 / (l_len + r_len - overlap);
                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                        update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);

                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                                    active_dict[l_rec_idx].erase(r_rec_idx);

                                    sim = overlap * 1.0 / (l_len + r_len - overlap);
                                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                    update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);
                                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                                    ++total_compared_pairs;

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
                threshold = min(1.0, 1.0 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len);
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
        } else {
            r_rec_idx = event.rec_idx;
            r_tok_idx = event.tok_idx;
            token = rtoken_vector[r_rec_idx][r_tok_idx];
            r_len = rtoken_vector[r_rec_idx].size();
            if (l_inverted_index.count(token)) {
                set<pair<int, int>> l_records = l_inverted_index[token];
//                for (auto l_rec_tuple : l_records) {
                for (set<pair<int, int>>::iterator it = l_records.begin(); it != l_records.end(); ++it) {
                    pair<int, int> l_rec_tuple = *it;
                    l_rec_idx = l_rec_tuple.first;
                    l_tok_idx = l_rec_tuple.second;
                    l_len = ltoken_vector[l_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        sim = overlap * 1.0 / (l_len + r_len - overlap);
                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                        update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);

                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                                    active_dict[l_rec_idx].erase(r_rec_idx);

                                    sim = overlap * 1.0 / (l_len + r_len - overlap);
                                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                    update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);
                                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                                    ++total_compared_pairs;

                                } else {
                                    ++active_dict[l_rec_idx][r_rec_idx];
                                }
                            } else {
                                active_dict[l_rec_idx][r_rec_idx] = 1;
                            }
                        } else {
                            active_dict[l_rec_idx] = unordered_map<int, short>();
                            active_dict[l_rec_idx][r_rec_idx] = 1;
                        }
                    }

                    if (total_compared_pairs % 100000 == 0 &&
                        !total_compared_pairs_set.count(total_compared_pairs)) {
                        total_compared_pairs_set.insert(total_compared_pairs);
                        if (topk_heap.size() > 0) {
                            printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                                   total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec,
                                   topk_heap.top().r_rec,
                                   prefix_events.top().threshold, prefix_events.top().table_indicator,
                                   prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                        }
                    }
                }
            }

            if (r_tok_idx + 1 < r_len) {
                threshold = min(1.0, 1.0 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len);
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
        }
    }

//    cout << "enter the full topk join: " << prefix_match_max_size << endl;

    // Complete the topk similarity join.
    while (prefix_events.size() > 0) {
        if (topk_heap.size() == output_size && (topk_heap.top().sim >= prefix_events.top().threshold)) {
            break;
        }

        event = prefix_events.top();
        prefix_events.pop();
        table_indicator = event.table_indicator;
        if (table_indicator == 0) {
            l_rec_idx = event.rec_idx;
            l_tok_idx = event.tok_idx;
            token = ltoken_vector[l_rec_idx][l_tok_idx];
            l_len = ltoken_vector[l_rec_idx].size();
            if (r_inverted_index.count(token)) {
                set<pair<int, int>> r_records = r_inverted_index[token];
//                for (auto r_rec_tuple : r_records) {
                for (set<pair<int, int>>::iterator it = r_records.begin(); it != r_records.end(); ++it) {
                    pair<int, int> r_rec_tuple = *it;
                    r_rec_idx = r_rec_tuple.first;
                    r_tok_idx = r_rec_tuple.second;
                    r_len = rtoken_vector[r_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        sim = overlap * 1.0 / (l_len + r_len - overlap);
                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);

                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                                    active_dict[l_rec_idx].erase(r_rec_idx);

                                    sim = overlap * 1.0 / (l_len + r_len - overlap);
                                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                                    ++total_compared_pairs;

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
                threshold = min(1.0, 1.0 - (l_tok_idx + 1 - prefix_match_max_size) * 1.0 / l_len);
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
            r_rec_idx = event.rec_idx;
            r_tok_idx = event.tok_idx;
            token = rtoken_vector[r_rec_idx][r_tok_idx];
            r_len = rtoken_vector[r_rec_idx].size();
            if (l_inverted_index.count(token)) {
                set<pair<int, int>> l_records = l_inverted_index[token];
//                for (auto l_rec_tuple : l_records) {
                for (set<pair<int, int>>::iterator it = l_records.begin(); it != l_records.end(); ++it) {
                    pair<int, int> l_rec_tuple = *it;
                    l_rec_idx = l_rec_tuple.first;
                    l_tok_idx = l_rec_tuple.second;
                    l_len = ltoken_vector[l_rec_idx].size();

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (l_tok_idx + 1 == l_len || r_tok_idx + 1 == r_len) {
                        overlap = 1;
                        if (active_dict.count(l_rec_idx) && active_dict[l_rec_idx].count(r_rec_idx)) {
                            overlap += active_dict[l_rec_idx][r_rec_idx];
                            active_dict[l_rec_idx].erase(r_rec_idx);
                        }
                        sim = overlap * 1.0 / (l_len + r_len - overlap);
                        update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);

                        ++total_compared_pairs;
                    } else {
                        if (active_dict.count(l_rec_idx)) {
                            if (active_dict[l_rec_idx].count(r_rec_idx)) {
                                value = active_dict[l_rec_idx][r_rec_idx];
                                if (value == prefix_match_max_size) {
                                    overlap = new_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                                    active_dict[l_rec_idx].erase(r_rec_idx);

                                    sim = overlap * 1.0 / (l_len + r_len - overlap);
                                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                                    ++total_compared_pairs;

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
                threshold = min(1.0, 1.0 - (r_tok_idx + 1 - prefix_match_max_size) * 1.0 / r_len);
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
        l_rec_idx = it->first;
        l_len = ltoken_vector[l_rec_idx].size();
        unordered_map<int, short> temp_map = it->second;
        for (unordered_map<int, short>::iterator it2 = temp_map.begin(); it2 != temp_map.end(); ++it2) {
            r_len = rtoken_vector[it2->first].size();
            if (l_len < len_upper_bound && r_len < len_upper_bound) {
                value = it2->second;
                sim = value * 1.0 / (l_len + r_len - value);
                update_topk_heap(topk_heap, sim, l_rec_idx, it2->first, output_size);
            }
        }
    }

//    cout << "exit full topk join. prefix size: " << prefix_match_max_size << " topk size " << topk_heap.size() << endl;
}


Heap new_topk_sim_join_plain_first(const Table& ltoken_vector, const Table& rtoken_vector,
                             atomic<int>& best_q, CandSet& cand_set, const int prefix_match_max_size,
                             const int select_param_size, const int output_size) {
    cout << "In new topk sim plain first" << endl;

    PrefixHeap prefix_events;
    new_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events);

    Heap topk_heap;
    new_topk_sim_join_plain_first_impl(ltoken_vector, rtoken_vector, topk_heap, best_q, cand_set, prefix_events,
                                       prefix_match_max_size, select_param_size, output_size);
//    cout << "exit topk impl " << prefix_match_max_size << " topk size: " << topk_heap.size() << endl;

    return topk_heap;
}
