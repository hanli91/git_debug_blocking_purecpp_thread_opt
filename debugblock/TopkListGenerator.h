#ifndef TEST_TOPKLISTGENERATOR_H
#define TEST_TOPKLISTGENERATOR_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <string>

#include "ReuseInfoArray.h"
#include "TopPair.h"
#include "TopkHeader.h"
#include "Signal.h"

using namespace std;

typedef priority_queue<TopPair> Heap;
typedef unordered_map<int, unordered_set<int>> CandSet;
typedef unordered_map<int, unordered_map<int, ReuseInfoArray>> ReuseSet;
typedef vector<vector<int>> Table;

class TopkListGenerator {
public:

    void generate_topklist_for_config(const Table& ltoken_vector, const Table& rtoken_vector,
                                      const Table& lindex_vector, const Table& rindex_vector,
                                      const Table& lfield_vector, const Table& rfield_vector,
                                      const vector<int>& ltoken_sum_vector, const vector<int>& rtoken_sum_vector,
                                      const vector<int>& field_list, const int max_field_num, const int removed_field,
                                      const bool first_run, Signal& signal,
                                      CandSet& cand_set,  ReuseSet& reuse_set, vector<Heap>& heap_list,
                                      const Heap& init_topk_heap, const int prefix_match_max_size, const int rec_ave_length,
                                      const int output_size, const int activate_reusing_module, const int topk_type, const string& output_path);

    TopkListGenerator();
    ~TopkListGenerator();
};


#endif //TEST_TOPKLISTGENERATOR_H
