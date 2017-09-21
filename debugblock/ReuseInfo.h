#include <unordered_map>

class ReuseInfo {
public:
    int overlap;
    std::unordered_map<int, int> map;

    ReuseInfo();
    ReuseInfo(int o);
    ~ReuseInfo();
};
