#ifndef TEST_GENERATERECOMLISTS_H
#define TEST_GENERATERECOMLISTS_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <atomic>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <folly/AtomicUnorderedMap.h>

#include "ReuseInfoArray.h"
#include "TopPair.h"
#include "TopkHeader.h"

using namespace std;

typedef priority_queue<TopPair> Heap;
typedef unordered_map<int, unordered_set<int>> CandSet;
typedef unordered_map<int, unordered_map<int, ReuseInfoArray>> ReuseSet;
typedef vector<vector<int>> Table;
//typedef folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> ReuseSet;


void generate_config_lists(const vector<int>& field_list, const vector<int>& ltoken_sum_vector,
                           const vector<int>& rtoken_sum_vector, const double field_remove_ratio,
                           const uint32_t ltable_size, const uint32_t rtable_size,
                           vector<Config>& config_lists);

void generate_topk_list(const Table& ltoken_vector, const Table& rtoken_vector,
                        const Table& lindex_vector, const Table& rindex_vector,
                        const Table& lfield_vector, const Table& rfield_vector,
                        const vector<int>& ltoken_sum_vector, const vector<int>& rtoken_sum_vector,
                        const int max_field_num, const int rec_ave_len_thres,
                        const int output_size, const string& output_path,
                        vector<Config>& config_list, Config& config, const int q, bool select_q, const bool activate_reusing_module,
                        CandSet& cand_set,  folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set,
                        atomic<int>& thread_count, atomic<int>& finished_config, atomic<int>& best_q);

double double_max(const double a, double b);


class GenerateRecomLists {
public:

    void generate_recom_lists(Table& ltoken_vector, Table& rtoken_vector,
                              Table& lindex_vector, Table& rindex_vector,
                              Table& lfield_vector, Table& rfield_vector,
                              vector<int>& ltoken_sum_vector, vector<int>& rtoken_sum_vector, vector<int>& field_list,
                              CandSet& cand_set, uint32_t prefix_match_max_size, uint32_t rec_ave_len_thres,
                              uint32_t offset_of_field_num, uint32_t max_field_num,
                              uint32_t minimal_num_fields, double field_remove_ratio,
                              uint32_t output_size, string output_path,
                              const bool activate_reusing_module, const bool use_new_topk, const bool use_parallel);

    GenerateRecomLists();
    ~GenerateRecomLists();
};


#endif //TEST_GENERATERECOMLISTS_H
