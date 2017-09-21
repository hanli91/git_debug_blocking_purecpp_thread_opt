#ifndef TEST_TOPKHEADER_H
#define TEST_TOPKHEADER_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <atomic>
#include <stdio.h>
#include <folly/AtomicUnorderedMap.h>

#include "TopPair.h"
#include "ReuseInfoArray.h"
#include "PrefixEvent.h"
#include "Config.h"

using namespace std;

typedef priority_queue<TopPair> Heap;
typedef unordered_map<int, unordered_set<int>> CandSet;
typedef unordered_map<int, unordered_map<int, ReuseInfoArray>> ReuseSet;
typedef unordered_map<int, set<pair<int, int>>> InvertedIndex;
typedef vector<vector<int>> Table;
typedef priority_queue<PrefixEvent> PrefixHeap;
//typedef folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray> ReuseSet;


Heap new_topk_sim_join_plain(const Table& ltoken_vector, const Table& rtoken_vector,
                             CandSet& cand_set, const int prefix_match_max_size, const int output_size);

Heap new_topk_sim_join_plain_first(const Table& ltoken_vector, const Table& rtoken_vector,
                                   atomic<int>& best_q, CandSet& cand_set, const int prefix_match_max_size,
                                   const int select_param_size, const int output_size);

Heap new_topk_sim_join_record(const Table& ltoken_vector, const Table& rtoken_vector,
                              const Table& lindex_vector, const Table& rindex_vector,
                              const Table& lfield_vector, const Table& rfield_vector,
                              const vector<int>& field_list, const vector<int>& removed_fields,
                              const int max_field_num, const bool use_remain,
                              vector<Heap>& heap_vector, vector<TopPair>& init_topk_list,
                              CandSet& cand_set, ReuseSet& reuse_set, const int prefix_match_max_size,
                              const int output_size);

Heap new_topk_sim_join_reuse(const Table& ltoken_vector, const Table& rtoken_vector,
                             const int max_field_num, atomic<int>& finished_config, vector<Config> config_list,
                             const bool use_remain, const vector<int>& remained_fields, const vector<int> removed_fields,
                             CandSet& cand_set, folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set, const int prefix_match_max_size,
                             const int output_size);

Heap new_topk_sim_join_record_first(const Table& ltoken_vector, const Table& rtoken_vector,
                              const Table& lindex_vector, const Table& rindex_vector,
                              const Table& lfield_vector, const Table& rfield_vector,
                              const vector<int>& field_list, const vector<int>& removed_fields,
                              const int max_field_num, const bool use_remain,
                              atomic<int>& best_q, CandSet& cand_set, folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set,
                              const int prefix_match_max_size, const int select_param_size,
                              const int output_size);


Heap original_topk_sim_join_plain(const Table& ltoken_vector, const Table& rtoken_vector,
                                  CandSet& cand_set, const int output_size);

Heap original_topk_sim_join_plain_first(const Table& ltoken_vector, const Table& rtoken_vector, atomic<int>& best_q,
                                        CandSet& cand_set, const int select_param_size, const int output_size);

Heap original_topk_sim_join_record_first(const Table& ltoken_vector, const Table& rtoken_vector,
                                   const Table& lindex_vector, const Table& rindex_vector,
                                   const Table& lfield_vector, const Table& rfield_vector,
                                   const vector<int>& field_list, const vector<int>& removed_fields,
                                   const int max_field_num, const bool use_remain,
                                   atomic<int>& best_q, CandSet& cand_set, folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set,
                                   const int select_param_size, const int output_size);

Heap original_topk_sim_join_record(const Table& ltoken_vector, const Table& rtoken_vector,
                                   const Table& lindex_vector, const Table& rindex_vector,
                                   const Table& lfield_vector, const Table& rfield_vector,
                                   const vector<int>& field_list, const vector<int>& removed_fields,
                                   const int max_field_num, const bool use_remain,
                                   vector<Heap>& heap_vector, vector<TopPair>& init_topk_list,
                                   CandSet& cand_set, ReuseSet& reuse_set, const int output_size);

Heap original_topk_sim_join_reuse(const Table& ltoken_vector, const Table& rtoken_vector,
                                  const int max_field_num, atomic<int>& finished_config, vector<Config> config_list,
                                  const bool use_remain, const vector<int>& remained_fields, const vector<int> removed_fields,
                                  CandSet& cand_set, folly::AtomicUnorderedInsertMap<uint64_t, ReuseInfoArray>& reuse_set, const int output_size);

int original_plain_get_overlap(const vector<int>& ltoken_list, const vector<int>& rtoken_list);

void original_reuse_get_overlap(const vector<int> ltoken_list, const vector<int> rtoken_list,
                           const vector<int> lindex_list, const vector<int> rindex_list,
                           ReuseInfoArray& reuse_info);

void original_generate_prefix_events_impl(const Table& table, const int table_indicator,
                                          PrefixHeap& prefix_events);

void original_generate_prefix_events(const Table& ltable, const Table& rtable,
                                     PrefixHeap& prefix_events);


void new_generate_prefix_events_impl(const Table& table, const int table_indicator,
                                     PrefixHeap& prefix_events);

void new_generate_prefix_events(const Table& ltable, const Table& rtable,
                                PrefixHeap& prefix_events);

void new_reuse_get_overlap(const vector<int> ltoken_list, const vector<int> rtoken_list,
                           const vector<int> lindex_list, const vector<int> rindex_list,
                           ReuseInfoArray& reuse_info);

int new_plain_get_overlap(const vector<int>& ltoken_list, const vector<int>& rtoken_list);

#endif //TEST_TOPKHEADER_H
