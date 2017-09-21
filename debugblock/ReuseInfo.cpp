#include "ReuseInfo.h"

ReuseInfo::ReuseInfo() {
    map = std::unordered_map<int, int>();
}

ReuseInfo::ReuseInfo(int o) {
    overlap = o;
    map = std::unordered_map<int, int>();
}

ReuseInfo::~ReuseInfo() { }
