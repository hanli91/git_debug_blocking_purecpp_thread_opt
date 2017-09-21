#include "TopkListGenerator.h"

TopkListGenerator::TopkListGenerator() {

}

TopkListGenerator::~TopkListGenerator() {

}

double init_topk_list_calc_sim(const vector<int>& ltoken_list, const vector<int>& rtoken_list) {
    int overlap = 0;
    unordered_set<int> rset;
    unsigned long denom = 0;

    for (int i = 0; i < rtoken_list.size(); ++i) {
        rset.insert(rtoken_list[i]);
    }

    for (int i = 0; i < ltoken_list.size(); ++i) {
        if (rset.count(ltoken_list[i])) {
            ++overlap;
        }
    }

    denom = ltoken_list.size() + rtoken_list.size() - overlap;
    if (denom== 0) {
        return 0.0;
    }
    return overlap * 1.0 / denom;
}

void save_topk_list_to_file(const vector<int>& field_list, const string& output_path,
                            Heap topk_heap) {
    string path = output_path + "topk_";
    char buf[10];
    for (int i = 0; i <field_list.size(); ++i) {
        sprintf(buf, "%d", field_list[i]);
        if (i != 0) {
            path.append(string("_"));
        }
        path.append(buf);
    }
    path += ".txt";

    cout << "in save_topk_list_to_file: topk heap size " << topk_heap.size() << endl;
    FILE* fp = fopen(path.c_str(), "w+");
    while (topk_heap.size() > 0) {
        TopPair pair = topk_heap.top();
        topk_heap.pop();
        fprintf(fp, "%.16f %d %d\n", pair.sim, pair.l_rec, pair.r_rec);
    }
    fclose(fp);
}

void init_topk_list(const Table& ltoken_vector, const Table& rtoken_vector,
                    Heap& old_heap, vector<TopPair>& new_heap) {
    while (old_heap.size() > 0) {
        TopPair pair = old_heap.top();
        old_heap.pop();

        double new_sim = init_topk_list_calc_sim(ltoken_vector[pair.l_rec], rtoken_vector[pair.r_rec]);
        if (new_sim > 0) {
            new_heap.push_back(TopPair(new_sim, pair.l_rec, pair.r_rec));
        }
    }

    return;
}


void TopkListGenerator::generate_topklist_for_config(const Table& ltoken_vector, const Table& rtoken_vector,
                                                     const Table& lindex_vector, const Table& rindex_vector,
                                                     const Table& lfield_vector, const Table& rfield_vector,
                                                     const vector<int>& ltoken_sum_vector, const vector<int>& rtoken_sum_vector,
                                                     const vector<int>& field_list, const int max_field_num, const int removed_field,
                                                     const bool first_run, Signal& signal,
                                                     CandSet& cand_set,  ReuseSet& reuse_set, vector<Heap>& heap_vector,
                                                     const Heap& init_topk_heap, const int prefix_match_max_size, const int rec_ave_len_thres,
                                                     const int output_size, const int activate_reusing_module, const int topk_type, const string& output_path) {
    char buf[10];
    string info = string("current configuration: [");
    for (int i = 0; i <field_list.size(); ++i) {
        sprintf(buf, "%d", field_list[i]);
        if (i != 0) {
            info.append(string(", "));
        }
        info.append(buf);
    }
    info.append("]");
    cout << info << endl;

    int ltoken_total_sum = 0, rtoken_total_sum = 0;
    for (int i = 0; i < field_list.size(); ++i) {
        ltoken_total_sum += ltoken_sum_vector[field_list[i]];
        rtoken_total_sum += rtoken_sum_vector[field_list[i]];
    }

    double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size();
    double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size();

    unordered_set<int> remained_set;
    for (int i = 0; i < field_list.size(); ++i) {
        remained_set.insert(field_list[i]);
    }
    vector<int> removed_fields;
    for (int i = 0; i < max_field_num; ++i) {
        if (!remained_set.count(i)) {
            removed_fields.push_back(i);
        }
    }
    bool use_remain = false;
    if (field_list.size() < removed_fields.size()) {
        use_remain = true;
    }

    Heap return_topk_heap;
    Heap copy_init_topk_heap = init_topk_heap;
    vector<TopPair> updated_init_topk_list;

    init_topk_list(ltoken_vector, rtoken_vector, copy_init_topk_heap, updated_init_topk_list);

    int select_param_size = 50;

    if (activate_reusing_module == 0) {
        if (lrec_ave_len >= rec_ave_len_thres) {
            if (first_run) {
                if (prefix_match_max_size > 0) {
                    return_topk_heap = new_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, signal,
                                                                     cand_set, prefix_match_max_size,
                                                                     select_param_size, output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, signal, cand_set,
                                                                          select_param_size, output_size);
                }
            } else {
                if (prefix_match_max_size > 0) {
                    return_topk_heap = new_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set,
                                                              prefix_match_max_size, output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size);
                }

            }
        } else {
            if (first_run) {
                if (prefix_match_max_size > 0) {
                    return_topk_heap = new_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, signal,
                                                                     cand_set, prefix_match_max_size,
                                                                     select_param_size, output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, signal, cand_set,
                                                                          select_param_size, output_size);
                }
            } else {
                if (prefix_match_max_size > 0) {
                    return_topk_heap = new_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set,
                                                              prefix_match_max_size, output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size);
                }
            }
        }
//        return_topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size);
    } else {
        if (lrec_ave_len >= rec_ave_len_thres || rrec_ave_len >= rec_ave_len_thres) {
            if (topk_type == 0) {
                if (field_list.size() <= 1) {
                    return_topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                                    max_field_num, use_remain, field_list, removed_fields,
                                                    updated_init_topk_list, cand_set, reuse_set, prefix_match_max_size,
                                                    output_size);
                } else {
                    if (first_run) {
                        if (prefix_match_max_size > 0) {
                            return_topk_heap = new_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                lfield_vector, rfield_vector, field_list, removed_fields,
                                                                max_field_num, use_remain, heap_vector, signal,
                                                                cand_set, reuse_set,
                                                                prefix_match_max_size, select_param_size, output_size);
                        } else {
                            return_topk_heap = original_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                lfield_vector, rfield_vector, field_list, removed_fields,
                                                                max_field_num, use_remain, heap_vector, signal,
                                                                cand_set, reuse_set, select_param_size, output_size);
                        }
                    } else {
                        return_topk_heap = new_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                lfield_vector, rfield_vector, field_list, removed_fields,
                                                                max_field_num, use_remain, heap_vector,
                                                                updated_init_topk_list, cand_set, reuse_set,
                                                                prefix_match_max_size, output_size);
                    }
                }
            } else if (topk_type == 1) {
                return_topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                            max_field_num, use_remain, field_list, removed_fields,
                                            updated_init_topk_list, cand_set, reuse_set, prefix_match_max_size,
                                            output_size);
            }
        } else {
            if (topk_type == 0) {
                if (remained_set.size() <= 1) {
//                    return_topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector, max_field_num, use_remain,
//                                                 field_list, removed_fields, updated_init_topk_list,
//                                                 cand_set, reuse_set, output_size);
                    if (prefix_match_max_size > 0) {
                        return_topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                                    max_field_num, use_remain, field_list, removed_fields,
                                                    updated_init_topk_list, cand_set, reuse_set, prefix_match_max_size,
                                                    output_size);
                    } else {
                        return_topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                                    max_field_num, use_remain, field_list, removed_fields,
                                                    updated_init_topk_list, cand_set, reuse_set, output_size);
                    }
                } else {
//                    return_topk_heap = original_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
//                                                  lfield_vector, rfield_vector, field_list, removed_fields,
//                                                  max_field_num, use_remain, heap_vector, updated_init_topk_list,
//                                                  cand_set, reuse_set, output_size);
                    if (first_run) {
                        if (prefix_match_max_size > 0) {
                            return_topk_heap = new_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                lfield_vector, rfield_vector, field_list, removed_fields,
                                                                max_field_num, use_remain, heap_vector, signal,
                                                                cand_set, reuse_set,
                                                                prefix_match_max_size, select_param_size, output_size);
                        } else {
                            return_topk_heap = original_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                lfield_vector, rfield_vector, field_list, removed_fields,
                                                                max_field_num, use_remain, heap_vector, signal,
                                                                cand_set, reuse_set, select_param_size, output_size);
                        }
                    } else {
                        if (prefix_match_max_size > 0) {
                            return_topk_heap = new_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                    lfield_vector, rfield_vector, field_list, removed_fields,
                                                                    max_field_num, use_remain, heap_vector,
                                                                    updated_init_topk_list, cand_set, reuse_set,
                                                                    prefix_match_max_size, output_size);
                        } else {
                            return_topk_heap = original_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                                                    lfield_vector, rfield_vector, field_list, removed_fields,
                                                                    max_field_num, use_remain, heap_vector,
                                                                    updated_init_topk_list, cand_set, reuse_set, output_size);
                        }
                    }
                }
            } else if (topk_type == 1) {
//                    return_topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector, max_field_num, use_remain,
//                                                 field_list, removed_fields, updated_init_topk_list,
//                                                 cand_set, reuse_set, output_size);
                    if (prefix_match_max_size > 0) {
                        return_topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                                  max_field_num, use_remain, field_list, removed_fields,
                                                  updated_init_topk_list, cand_set, reuse_set, prefix_match_max_size,
                                                  output_size);
                    } else {
                        return_topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                                  max_field_num, use_remain, field_list, removed_fields,
                                                  updated_init_topk_list, cand_set, reuse_set, output_size);
                    }
            }
        }
    }

    if (first_run) {
        if (signal.value == prefix_match_max_size) {
            save_topk_list_to_file(field_list, output_path, return_topk_heap);
        }
    } else {
        save_topk_list_to_file(field_list, output_path, return_topk_heap);
    }

    cout << "finish" << endl;
}
