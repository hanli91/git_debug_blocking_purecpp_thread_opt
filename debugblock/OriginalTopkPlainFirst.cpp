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


void original_topk_plain_first_impl(const Table& ltoken_vector, const Table& rtoken_vector,
                                    Heap& topk_heap, PrefixHeap& prefix_events, atomic<int>& best_q, CandSet& cand_set,
                                    const int select_param_size, const int output_size) {
    long int total_compared_pairs = 0;
    unordered_set<long int> total_compared_pairs_set;

    unordered_map<int, unordered_set<int>> compared_set;

    InvertedIndex l_inverted_index, r_inverted_index;

    Heap param_topk_heap;

    // Select the prefix parameter.
    while (prefix_events.size() > 0) {
        if (param_topk_heap.size() == select_param_size && (param_topk_heap.top().sim >= prefix_events.top().threshold)) {
//            cout << "Select parameter: 0" << endl;
            printf("Select parameter: 0\n");
            best_q = 0;
            break;
        }
        if (best_q != -1 && best_q != 0) {
//            cout << "Break parameter: 0" << endl;
            printf("Break job on parameter: 0\n");
            return;
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
                for (set<pair<int, int>>::iterator it = r_records.begin(); it != r_records.end(); ++it) {
                    pair<int, int> r_rec_tuple = *it;
                    int r_rec_idx = r_rec_tuple.first;
                    unsigned long r_len = rtoken_vector[r_rec_idx].size();

                    if (topk_heap.size() == output_size &&
                        (l_len < topk_heap.top().sim * r_len || l_len > r_len / topk_heap.top().sim)) {
                        continue;
                    }

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    int overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                    double sim = overlap * 1.0 / (l_len + r_len - overlap);
                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                    update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);
                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                    ++total_compared_pairs;
                    if (total_compared_pairs % 100000 == 0) {
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec,
                               topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                    }
                }
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }

            double index_threshold = 1.0;
            int numer_index = l_len - l_tok_idx;
            int denom_index = l_len + l_tok_idx;
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
            int r_rec_idx = event.rec_idx;
            int r_tok_idx = event.tok_idx;
            int token = rtoken_vector[r_rec_idx][r_tok_idx];
            unsigned long r_len = rtoken_vector[r_rec_idx].size();
            if (l_inverted_index.count(token)) {
                set<pair<int, int>> l_records = l_inverted_index[token];
                for (set<pair<int, int>>::iterator it = l_records.begin(); it != l_records.end(); ++it) {
                    pair<int, int> l_rec_tuple = *it;
                    int l_rec_idx = l_rec_tuple.first;
                    unsigned long l_len = ltoken_vector[l_rec_idx].size();

                    if (topk_heap.size() == output_size &&
                        (l_len < topk_heap.top().sim * r_len || l_len > r_len / topk_heap.top().sim)) {
                        continue;
                    }

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    int overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                    double sim = overlap * 1.0 / (l_len + r_len - overlap);
                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                    update_topk_heap(param_topk_heap, sim, l_rec_idx, r_rec_idx, select_param_size);
                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                    ++total_compared_pairs;
                    if (total_compared_pairs % 100000 == 0) {
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec,
                               topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                    }
                }
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }
            double index_threshold = 1.0;
            int numer_index = r_len - r_tok_idx;
            int denom_index = r_len + r_tok_idx;
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

    // Complete the topk similarity join.
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
                for (set<pair<int, int>>::iterator it = r_records.begin(); it != r_records.end(); ++it) {
                    pair<int, int> r_rec_tuple = *it;
                    int r_rec_idx = r_rec_tuple.first;
                    unsigned long r_len = rtoken_vector[r_rec_idx].size();

                    if (topk_heap.size() == output_size &&
                        (l_len < topk_heap.top().sim * r_len || l_len > r_len / topk_heap.top().sim)) {
                        continue;
                    }

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    int overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                    double sim = overlap * 1.0 / (l_len + r_len - overlap);
                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                    ++total_compared_pairs;
                    if (total_compared_pairs % 100000 == 0) {
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec,
                               topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                    }
                }
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }

            double index_threshold = 1.0;
            int numer_index = l_len - l_tok_idx;
            int denom_index = l_len + l_tok_idx;
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
            int r_rec_idx = event.rec_idx;
            int r_tok_idx = event.tok_idx;
            int token = rtoken_vector[r_rec_idx][r_tok_idx];
            unsigned long r_len = rtoken_vector[r_rec_idx].size();
            if (l_inverted_index.count(token)) {
                set<pair<int, int>> l_records = l_inverted_index[token];
                for (set<pair<int, int>>::iterator it = l_records.begin(); it != l_records.end(); ++it) {
                    pair<int, int> l_rec_tuple = *it;
                    int l_rec_idx = l_rec_tuple.first;
                    unsigned long l_len = ltoken_vector[l_rec_idx].size();

                    if (topk_heap.size() == output_size &&
                        (l_len < topk_heap.top().sim * r_len || l_len > r_len / topk_heap.top().sim)) {
                        continue;
                    }

                    if (cand_set.count(l_rec_idx) && cand_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    if (compared_set.count(l_rec_idx) && compared_set[l_rec_idx].count(r_rec_idx)) {
                        continue;
                    }

                    int overlap = original_plain_get_overlap(ltoken_vector[l_rec_idx], rtoken_vector[r_rec_idx]);
                    double sim = overlap * 1.0 / (l_len + r_len - overlap);
                    update_topk_heap(topk_heap, sim, l_rec_idx, r_rec_idx, output_size);
                    update_compared_set(compared_set, l_rec_idx, r_rec_idx);

                    ++total_compared_pairs;
                    if (total_compared_pairs % 100000 == 0) {
                        printf("%ld (%.16f %d %d) (%.16f %d %d %d)\n",
                               total_compared_pairs, topk_heap.top().sim, topk_heap.top().l_rec,
                               topk_heap.top().r_rec,
                               prefix_events.top().threshold, prefix_events.top().table_indicator,
                               prefix_events.top().rec_idx, prefix_events.top().tok_idx);
                    }
                }
            }

            double topk_heap_sim_index = 0.0;
            if (topk_heap.size() == output_size) {
                topk_heap_sim_index = topk_heap.top().sim;
            }
            double index_threshold = 1.0;
            int numer_index = r_len - r_tok_idx;
            int denom_index = r_len + r_tok_idx;
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
    printf("number of compared pairs: %ld\n", total_compared_pairs);
}

Heap original_topk_sim_join_plain_first(const Table& ltoken_vector, const Table& rtoken_vector, atomic<int>& best_q,
                               CandSet& cand_set, const int select_param_size, const int output_size) {
    cout << "In original topk sim plain first" << endl;

    PrefixHeap prefix_events;
    original_generate_prefix_events(ltoken_vector, rtoken_vector, prefix_events);

    Heap topk_heap;
    original_topk_plain_first_impl(ltoken_vector, rtoken_vector, topk_heap, prefix_events, best_q,
                                   cand_set, select_param_size, output_size);

    return topk_heap;
}
