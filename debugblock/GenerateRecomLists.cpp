#include "GenerateRecomLists.h"

GenerateRecomLists::GenerateRecomLists() {}


GenerateRecomLists::~GenerateRecomLists() {}

void inline print_config_lists(const vector<Config>& config_lists) {
    for (int i = 0; i < config_lists.size(); ++i) {
        Config config = config_lists[i];
        for (int j = 0; j < config.field_list.size(); ++j) {
            cout << config.field_list[j] << " ";
        }
        cout << "\t\trecord_info: " << config.record_info << " finish: " << config.finish
             << " father: " << config.father << " father_config: ";
        if (config.father_config != NULL) {
            cout << config.father_config->index << endl;
        } else {
            cout << "-1" << endl;
        }
    }
}

void inline print_config_list(const Config& config) {
    for (int j = 0; j < config.field_list.size(); ++j) {
            cout << config.field_list[j] << " ";
    }
    cout << "\t\trecord_info: " << config.record_info << " father: " << config.father << endl;
}

void inline print_config_info(const Config& config, const string& prefix) {
    string str = prefix + "\tconfig: [";
    char buf[10];
    for (int i = 0; i <config.field_list.size(); ++i) {
        sprintf(buf, "%d", config.field_list[i]);
        if (i != 0) {
            str.append(string(" "));
        }
        str.append(buf);
    }
    str.append("] removed_fields: [");
    for (int i = 0; i <config.removed_field_list.size(); ++i) {
        sprintf(buf, "%d", config.removed_field_list[i]);
        if (i != 0) {
            str.append(string(" "));
        }
        str.append(buf);
    }
    str.append("]\t\trecord_info: ");
    sprintf(buf, "%d", config.record_info);
    str.append(buf);
    str.append("  father: ");
    sprintf(buf, "%d", config.father);
    str.append(buf);
    str.append(" use_remain: ");
    sprintf(buf, "%d", config.use_remain);
    str.append(buf);
    str.append(" finish: ");
    sprintf(buf, "%d", config.finish);
    str.append(buf);
    str.append("\n");

    printf("%s", str.c_str());
}

void inline copy_table_and_remove_fields(const Config& config, const Table& table_vector, const Table& index_vector,
                                         Table& new_table_vector, Table& new_index_vector) {
    unordered_set<int> field_set;
    for (int i = 0; i < config.field_list.size(); ++i) {
        field_set.insert(config.field_list[i]);
    }

    for (int i = 0; i < table_vector.size(); ++i) {
        new_table_vector.push_back(vector<int> ());
        new_index_vector.push_back(vector<int> ());
        for (int j = 0; j < table_vector[i].size(); ++j) {
            if (field_set.count(index_vector[i][j])) {
                new_table_vector[i].push_back(table_vector[i][j]);
                new_index_vector[i].push_back(table_vector[i][j]);
            }
        }
    }
}

void inline set_removed_field_list(const vector<int>& field_list, Config& config) {
    unordered_set<int> remained_set;
    for (int i = 0; i < config.field_list.size(); ++i) {
        remained_set.insert(config.field_list[i]);
    }
    vector<int> removed_fields;
    for (int i = 0; i < field_list.size(); ++i) {
        if (!remained_set.count(i)) {
            removed_fields.push_back(i);
        }
    }

    config.removed_field_list = removed_fields;
    if (config.field_list.size() < removed_fields.size()) {
        config.use_remain = true;
    }
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

//    cout << "in save_topk_list_to_file: topk heap size " << topk_heap.size() << endl;
    FILE* fp = fopen(path.c_str(), "w+");
    while (topk_heap.size() > 0) {
        TopPair pair = topk_heap.top();
        topk_heap.pop();
        fprintf(fp, "%.16f %d %d\n", pair.sim, pair.l_rec, pair.r_rec);
    }
    fclose(fp);
}

void GenerateRecomLists::generate_recom_lists(
                              Table& ltoken_vector, Table& rtoken_vector,
                              Table& lindex_vector, Table& rindex_vector,
                              Table& lfield_vector, Table& rfield_vector,
                              vector<int>& ltoken_sum_vector, vector<int>& rtoken_sum_vector, vector<int>& field_list,
                              CandSet& cand_set, uint32_t prefix_match_max_size, uint32_t rec_ave_len_thres,
                              uint32_t offset_of_field_num, uint32_t max_field_num,
                              uint32_t minimal_num_fields, double field_remove_ratio,
                              uint32_t output_size, string output_path,
                              const bool activate_reusing_module, const bool use_new_topk, const bool use_parallel) {

    vector<Config> config_lists;
    generate_config_lists(field_list, ltoken_sum_vector, rtoken_sum_vector, field_remove_ratio,
                          ltoken_vector.size(), rtoken_vector.size(), config_lists);
    print_config_lists(config_lists);

    int N_THREADS = 4;
    thread *th_first_round = new thread[N_THREADS];
    atomic<int> thread_count(0);
    atomic<int> best_q(-1);
    atomic<int> finished_config(0);

    vector<int> q_list;
    for (int i = 0; i < N_THREADS; ++i) {
        q_list.push_back(i + 3);
    }
    q_list[0] = 0;
    unordered_map<int, int> q_index_map;
    for (int i = 0; i < N_THREADS; ++i) {
        q_index_map[q_list[i]] = i;
    }
    folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> reuse_map1(10000000);
    folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> reuse_map2(10000000);
    folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> reuse_map3(10000000);
    folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> reuse_map4(10000000);


    int cur_config_pt = 0;
    for (int i = 0; i < N_THREADS; ++i) {
        if (i == 0) {
            th_first_round[i] = thread(generate_topk_list,
                ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                q_list[i], true, activate_reusing_module, ref(cand_set), ref(reuse_map1), ref(thread_count),
                ref(finished_config), ref(best_q));
        } else if (i == 1) {
            th_first_round[i] = thread(generate_topk_list,
                ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                q_list[i], true, activate_reusing_module, ref(cand_set), ref(reuse_map2), ref(thread_count),
                ref(finished_config), ref(best_q));
        } else if (i == 2) {
            th_first_round[i] = thread(generate_topk_list,
                ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                q_list[i], true, activate_reusing_module, ref(cand_set), ref(reuse_map3), ref(thread_count),
                ref(finished_config), ref(best_q));
        } else if (i == 3) {
            th_first_round[i] = thread(generate_topk_list,
                ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                q_list[i], true, activate_reusing_module, ref(cand_set), ref(reuse_map4), ref(thread_count),
                ref(finished_config), ref(best_q));
        }
    }
    ++cur_config_pt;

    int q_index = q_index_map[best_q];
    thread* th_list = new thread[config_lists.size() - 1];
    while (cur_config_pt < config_lists.size()) {
        usleep(1000);
        if (cur_config_pt < config_lists.size() && thread_count < N_THREADS) {
            if (q_index == 0) {
                th_list[cur_config_pt - 1] = thread(generate_topk_list,
                    ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                    ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                    max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                    -1, false, activate_reusing_module, ref(cand_set), ref(reuse_map1), ref(thread_count),
                    ref(finished_config), ref(best_q));
            } else if (q_index == 1) {
                th_list[cur_config_pt - 1] = thread(generate_topk_list,
                    ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                    ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                    max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                    -1, false, activate_reusing_module, ref(cand_set), ref(reuse_map2), ref(thread_count),
                    ref(finished_config), ref(best_q));
            } else if (q_index == 2) {
                th_list[cur_config_pt - 1] = thread(generate_topk_list,
                    ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                    ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                    max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                    -1, false, activate_reusing_module, ref(cand_set), ref(reuse_map3), ref(thread_count),
                    ref(finished_config), ref(best_q));
            } else if (q_index == 3) {
                th_list[cur_config_pt - 1] = thread(generate_topk_list,
                    ref(ltoken_vector), ref(rtoken_vector), ref(lindex_vector), ref(rindex_vector),
                    ref(lfield_vector), ref(rfield_vector), ref(ltoken_sum_vector), ref(rtoken_sum_vector),
                    max_field_num, rec_ave_len_thres, output_size, ref(output_path), ref(config_lists), ref(config_lists[cur_config_pt]),
                    -1, false, activate_reusing_module, ref(cand_set), ref(reuse_map4), ref(thread_count),
                    ref(finished_config), ref(best_q));
            }
            ++cur_config_pt;
            usleep(1000);
        }
    }

    for (int i = 0; i < N_THREADS; ++i) {
        th_first_round[i].join();
    }
    for (int i = 0; i < config_lists.size() - 1; ++i) {
        th_list[i].join();
    }
}


void generate_topk_list(const Table& ltoken_vector, const Table& rtoken_vector,
                        const Table& lindex_vector, const Table& rindex_vector,
                        const Table& lfield_vector, const Table& rfield_vector,
                        const vector<int>& ltoken_sum_vector, const vector<int>& rtoken_sum_vector,
                        const int max_field_num, const int rec_ave_len_thres,
                        const int output_size, const string& output_path,
                        vector<Config>& config_list, Config& config, const int q, bool select_q, const bool activate_reusing_module,
                        CandSet& cand_set, folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set,
                        atomic<int>& thread_count, atomic<int>& finished_config, atomic<int>& best_q) {
    thread_count += 1;
    print_config_info(config, "START");

    int ltoken_total_sum = 0, rtoken_total_sum = 0;
    for (int i = 0; i < config.field_list.size(); ++i) {
        ltoken_total_sum += ltoken_sum_vector[config.field_list[i]];
        rtoken_total_sum += rtoken_sum_vector[config.field_list[i]];
    }
    double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size();
    double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size();

    Heap return_topk_heap;
    int select_param_size = 50;

    if (activate_reusing_module == 0) {
        if (select_q) {
            if (q > 0) {
                return_topk_heap = new_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, best_q,
                                                                 cand_set, q, select_param_size, output_size);
            } else {
                return_topk_heap = original_topk_sim_join_plain_first(ltoken_vector, rtoken_vector, best_q, cand_set,
                                                                      select_param_size, output_size);
            }
//            return_topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size);
        } else {
            Table new_ltoken_vector, new_rtoken_vector, new_lindex_vector, new_rindex_vector;
            copy_table_and_remove_fields(config, ltoken_vector, lindex_vector,
                                         new_ltoken_vector, new_lindex_vector);
            copy_table_and_remove_fields(config, rtoken_vector, rindex_vector,
                                         new_rtoken_vector, new_rindex_vector);
            if (best_q > 0) {
                return_topk_heap = new_topk_sim_join_plain(new_ltoken_vector, new_rtoken_vector, cand_set,
                                                          best_q, output_size);
            } else {
                return_topk_heap = original_topk_sim_join_plain(new_ltoken_vector, new_rtoken_vector, cand_set, output_size);
            }
//            return_topk_heap = original_topk_sim_join_plain(new_ltoken_vector, new_rtoken_vector, cand_set, output_size);
        }
    } else {
        if (select_q) {
            if (config.field_list.size() <= 1) {
                if (lrec_ave_len >= rec_ave_len_thres || rrec_ave_len >= rec_ave_len_thres) {
                    return_topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                            max_field_num, finished_config, config_list, config.use_remain, config.field_list,
                                            config.removed_field_list, cand_set, reuse_set, q,  output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector,
                                            max_field_num, finished_config, config_list, config.use_remain, config.field_list,
                                            config.removed_field_list, cand_set, reuse_set, output_size);
                }
            } else {
                if (q > 0) {
                    return_topk_heap = new_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                            lfield_vector, rfield_vector, config.field_list, config.removed_field_list,
                                            max_field_num, config.use_remain, best_q,
                                            cand_set, reuse_set,
                                            q, select_param_size, output_size);
                } else {
                    return_topk_heap = original_topk_sim_join_record_first(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                            lfield_vector, rfield_vector, config.field_list, config.removed_field_list,
                                            max_field_num, config.use_remain, best_q,
                                            cand_set, reuse_set, select_param_size, output_size);
                }
            }
        } else {
            Table new_ltoken_vector, new_rtoken_vector, new_lindex_vector, new_rindex_vector;
            copy_table_and_remove_fields(config, ltoken_vector, lindex_vector,
                                         new_ltoken_vector, new_lindex_vector);
            copy_table_and_remove_fields(config, rtoken_vector, rindex_vector,
                                         new_rtoken_vector, new_rindex_vector);

            if (best_q > 0) {
                return_topk_heap = new_topk_sim_join_reuse(new_ltoken_vector, new_rtoken_vector,
                                      max_field_num, finished_config, config_list, config.use_remain, config.field_list,
                                      config.removed_field_list, cand_set, reuse_set, best_q, output_size);
            } else {
                return_topk_heap = original_topk_sim_join_reuse(new_ltoken_vector, new_rtoken_vector,
                                      max_field_num, finished_config, config_list, config.use_remain, config.field_list,
                                      config.removed_field_list, cand_set, reuse_set, output_size);
            }
        }
    }

    config.topk_heap = return_topk_heap;
    config.finish = true;
    finished_config += 1;

    if (select_q) {
        if (best_q == q) {
            save_topk_list_to_file(config.field_list, output_path, return_topk_heap);
        }
    } else {
        save_topk_list_to_file(config.field_list, output_path, return_topk_heap);
    }

    print_config_info(config, "END");

    thread_count -= 1;
}


void generate_config_lists(const vector<int>& field_list, const vector<int>& ltoken_sum_vector,
                           const vector<int>& rtoken_sum_vector, const double field_remove_ratio,
                           const uint32_t ltable_size, const uint32_t rtable_size,
                           vector<Config>& config_lists) {
    vector<int> feat_list_copy = field_list;
    int father = 0;
    int father_index = 0;
    Config root_config(feat_list_copy, true, 0, -1);
    config_lists.push_back(root_config);

    while (feat_list_copy.size() > 0) {
        double max_ratio = 0.0;
        uint32_t ltoken_total_sum = 0, rtoken_total_sum = 0;
        int removed_field_index = -1;
        bool has_long_field = false;

        for (int i = 0; i < feat_list_copy.size(); ++i) {
            ltoken_total_sum += ltoken_sum_vector[feat_list_copy[i]];
            rtoken_total_sum += rtoken_sum_vector[feat_list_copy[i]];
        }

        double lrec_ave_len = ltoken_total_sum * 1.0 / ltable_size;
        double rrec_ave_len = rtoken_total_sum * 1.0 / rtable_size;
        double ratio = 1 - (feat_list_copy.size() - 1) * field_remove_ratio / (1.0 + field_remove_ratio) *
                 double_max(lrec_ave_len, rrec_ave_len) / (lrec_ave_len + rrec_ave_len);

        for (int i = 0; i < feat_list_copy.size(); ++i) {
            max_ratio = double_max(max_ratio, double_max(ltoken_sum_vector[feat_list_copy[i]] * 1.0 / ltoken_total_sum,
                                                         rtoken_sum_vector[feat_list_copy[i]] * 1.0 / rtoken_total_sum));
            if (ltoken_sum_vector[feat_list_copy[i]] > ltoken_total_sum * ratio ||
                    rtoken_sum_vector[feat_list_copy[i]] > rtoken_total_sum * ratio) {
                removed_field_index = i;
                has_long_field = true;
                break;
            }
        }

        if (removed_field_index < 0) {
            removed_field_index = feat_list_copy.size() - 1;
        }

        cout << "required remove-field ratio: " << ratio << endl;
        cout << "actual max ratio: " << max_ratio << endl;
        cout << "remove field " << feat_list_copy[removed_field_index] << endl;

        for (int i = 0; i < feat_list_copy.size(); ++i) {
            vector<int> temp = feat_list_copy;
            temp.erase(temp.begin() + i);
            if (temp.size() <= 0) {
                continue;
            }
            if (i == removed_field_index) {
                Config config(temp, true, config_lists.size(), father);
                set_removed_field_list(field_list, config);
                config_lists.push_back(config);
                father_index = config_lists.size() - 1;

            } else {
                Config config(temp, false, config_lists.size(), father);
                set_removed_field_list(field_list, config);
                config_lists.push_back(config);
            }
        }
        father = father_index;

        feat_list_copy.erase(feat_list_copy.begin() + removed_field_index);
    }

    if (config_lists.size() > 0) {
        config_lists[0].father_config = new Config();
        (config_lists[0].father_config)->index = -1;

        for (int i = 1; i < config_lists.size(); ++i) {
            config_lists[i].father_config = &config_lists[config_lists[i].father];
        }
    }
}


double double_max(const double a, double b) {
    if (a > b) {
        return a;
    }
    return b;
}