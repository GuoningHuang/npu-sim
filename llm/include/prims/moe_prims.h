#include "prims/base.h"

class matmul_forward_moe : public MoeBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                 u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops, u_int64_t &vec_ops);
    void initialize();
    void parseJson(json j);
    matmul_forward_moe() {
        name = "matmul_forward_moe";
        param_name.insert(param_name.end(),
                          {"B", "T", "C", "OC", "K", "E_N", "is_merge",
                           "route_mode", "route_seed", "route_global_en",
                           "route_global_k", "route_local_begin"});
    }
};


class load_expert : public MoeBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                 u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops, u_int64_t &vec_ops);
    void initialize();
    load_expert() {
        name = "load_expert";
        param_name.insert(param_name.end(),
                          {"E_N", "K", "OC", "C", "strategy"});
    }
};
