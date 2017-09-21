#include <vector>
#include <cstddef>
#include <queue>

#include "TopPair.h"

using namespace std;

typedef priority_queue<TopPair> Heap;

class Config {
public:
    vector<int> field_list;
    vector<int> removed_field_list;
    bool record_info;
    bool use_remain;
    bool finish;

    int index;
    int father;
    Heap topk_heap;
    Config* father_config;

    Config();
    Config(const vector<int>& f_list);
    Config(const vector<int>& f_list, bool record_info, int father);
    Config(const vector<int>& f_list, bool record_info, int index, int father);
    ~Config();
};