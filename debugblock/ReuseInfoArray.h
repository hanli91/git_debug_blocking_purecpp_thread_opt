#ifndef TEST_REUSEINFOARRAY_H
#define TEST_REUSEINFOARRAY_H

#include <unordered_map>

class ReuseInfoArray {
public:
    int overlap;
    int info[8][8];

    ReuseInfoArray();
    ReuseInfoArray(int o);
    ~ReuseInfoArray();
};


#endif //TEST_REUSEINFOARRAY_H
