#include "Config.h"
using namespace std;

Config::Config() {
    this->record_info = true;
    this->father = -1;
    this->use_remain = false;
    this->finish = false;
    this->father_config = NULL;
}

Config::Config(const vector<int>& f_list) {
    for (int i = 0; i < f_list.size(); ++i) {
        field_list.push_back(f_list[i]);
    }

    this->record_info = true;
    this->father = -1;
    this->use_remain = false;
    this->finish = false;
    this->father_config = NULL;
}

Config::Config(const vector<int>& f_list, bool record_info, int father) {
    for (int i = 0; i < f_list.size(); ++i) {
        field_list.push_back(f_list[i]);
    }

    this->record_info = record_info;
    this->father = father;
    this->use_remain = false;
    this->finish = false;
    this->father_config = NULL;
}

Config::Config(const vector<int>& f_list, bool record_info, int index, int father) {
    for (int i = 0; i < f_list.size(); ++i) {
        field_list.push_back(f_list[i]);
    }

    this->record_info = record_info;
    this->father = father;
    this->index = index;
    this->use_remain = false;
    this->finish = false;
    this->father_config = NULL;
}

Config::~Config() {}
