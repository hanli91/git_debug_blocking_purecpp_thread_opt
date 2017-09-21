#include "TopkHeader.h"


void original_generate_prefix_events_impl(const Table& table, const int table_indicator,
                                          PrefixHeap& prefix_events) {
    for (int i = 0; i < table.size(); ++i) {
        unsigned long int length = table[i].size();
        if (length > 0) {
            for (int j = 0; j < length; ++j) {
                prefix_events.push(PrefixEvent(1.0 - j * 1.0 / length, table_indicator, i, j));
            }
        }
    }
}

void original_generate_prefix_events(const Table& ltable, const Table& rtable,
                                     PrefixHeap& prefix_events) {
    original_generate_prefix_events_impl(ltable, 0, prefix_events);
    original_generate_prefix_events_impl(rtable, 1, prefix_events);
}

void new_generate_prefix_events_impl(const Table& table, const int table_indicator,
                                     PrefixHeap& prefix_events) {
    for (int i = 0; i < table.size(); ++i) {
        unsigned long int length = table[i].size();
        if (length > 0) {
            prefix_events.push(PrefixEvent(1.0, table_indicator, i, 0));
        }
    }
}

void new_generate_prefix_events(const Table& ltable, const Table& rtable,
                                PrefixHeap& prefix_events) {
    new_generate_prefix_events_impl(ltable, 0, prefix_events);
    new_generate_prefix_events_impl(rtable, 1, prefix_events);
}


void original_reuse_get_overlap(const vector<int> ltoken_list, const vector<int> rtoken_list,
                                const vector<int> lindex_list, const vector<int> rindex_list,
                                ReuseInfoArray& reuse_info) {
    unordered_map<int, int> rmap;
    for (int i = 0; i < rtoken_list.size(); ++i) {
        rmap[rtoken_list[i]] = rindex_list[i];
    }

    for (int i = 0; i < ltoken_list.size(); ++i) {
        if (rmap.count(ltoken_list[i])) {
            ++reuse_info.overlap;
            ++reuse_info.info[lindex_list[i]][rmap[ltoken_list[i]]];
        }
    }
}


void new_reuse_get_overlap(const vector<int> ltoken_list, const vector<int> rtoken_list,
                           const vector<int> lindex_list, const vector<int> rindex_list,
                           ReuseInfoArray& reuse_info) {
    unordered_map<int, int> rmap;
    for (int i = 0; i < rtoken_list.size(); ++i) {
        rmap[rtoken_list[i]] = rindex_list[i];
    }

    for (int i = 0; i < ltoken_list.size(); ++i) {
        if (rmap.count(ltoken_list[i])) {
            ++reuse_info.overlap;
            ++reuse_info.info[lindex_list[i]][rmap[ltoken_list[i]]];
        }
    }
}

int new_plain_get_overlap(const vector<int>& ltoken_list, const vector<int>& rtoken_list) {
    int overlap = 0;
    unordered_set<int> rset;

    for (int i = 0; i < rtoken_list.size(); ++i) {
        rset.insert(rtoken_list[i]);
    }

    for (int i = 0; i < ltoken_list.size(); ++i) {
        if (rset.count(ltoken_list[i])) {
            ++overlap;
        }
    }

    return overlap;
}


int original_plain_get_overlap(const vector<int>& ltoken_list, const vector<int>& rtoken_list) {
    int overlap = 0;
    unordered_set<int> rset;

    for (int i = 0; i < rtoken_list.size(); ++i) {
        rset.insert(rtoken_list[i]);
    }

    for (int i = 0; i < ltoken_list.size(); ++i) {
        if (rset.count(ltoken_list[i])) {
            ++overlap;
        }
    }

    return overlap;
}
