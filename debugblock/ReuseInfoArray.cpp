#include "ReuseInfoArray.h"

ReuseInfoArray::ReuseInfoArray() {
    for(int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            info[i][j] = 0;
        }
    }
}


ReuseInfoArray::ReuseInfoArray(int o) {
    overlap = o;
    for(int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            info[i][j] = 0;
        }
    }
}

ReuseInfoArray::~ReuseInfoArray() {}
