#include "prims/moe_prims.h"
#include "utils/memory_utils.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(matmul_forward_moe);

namespace {
uint32_t NextLcg(uint32_t state) {
    return (1103515245u * state + 12345u) & 0x7fffffffu;
}

std::vector<int> SampleUniqueLcg(int total, int count, uint32_t seed) {
    std::vector<int> out;
    if (total <= 0 || count <= 0)
        return out;
    if (count > total)
        count = total;

    std::vector<bool> used(total, false);
    uint32_t state = seed & 0x7fffffffu;
    while ((int)out.size() < count) {
        state = NextLcg(state);
        int cand = (int)(state % (uint32_t)total);
        if (!used[cand]) {
            used[cand] = true;
            out.push_back(cand);
        }
    }
    return out;
}
} // namespace

void matmul_forward_moe::parseJson(json j) {
    json j_with_defaults = j;
    if (!j_with_defaults.contains("route_mode"))
        j_with_defaults["route_mode"] = 0;
    if (!j_with_defaults.contains("route_seed"))
        j_with_defaults["route_seed"] = 0;
    if (!j_with_defaults.contains("route_global_en"))
        j_with_defaults["route_global_en"] =
            j_with_defaults.contains("E_N") ? j_with_defaults["E_N"] : json(0);
    if (!j_with_defaults.contains("route_global_k"))
        j_with_defaults["route_global_k"] =
            j_with_defaults.contains("K") ? j_with_defaults["K"] : json(0);
    if (!j_with_defaults.contains("route_local_begin"))
        j_with_defaults["route_local_begin"] = 0;
    NpuBase::parseJson(j_with_defaults);
}

void matmul_forward_moe::initialize() {
    auto &p = param_value;
    data_chunk = {{"weight", p["OC"] * p["C"]}, {"bias", p["OC"]}};

    if (p["is_merge"]) {
        data_size_input = {p["B"] * p["T"] * p["C"] * p["K"]};
        data_chunk.push_back({"output", p["B"] * p["T"] * p["OC"]});
    } else {
        data_size_input = {p["B"] * p["T"] * p["C"]};
        data_chunk.push_back({"output", p["B"] * p["T"] * p["OC"] * p["K"]});
    }
}

void matmul_forward_moe::taskCore(TaskCoreContext &context, string prim_name,
                                  u_int64_t &dram_time, u_int64_t &exu_ops,
                                  u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    auto &selected_experts = prim_context->selected_experts_;
    auto &selected_freq = prim_context->selected_freq_;
    auto &prefetched_experts = prim_context->prefetched_experts_;

    cout << "[DEBUG] matmul_forward_moe: E_N=" << p["E_N"] 
         << " K=" << p["K"] 
         << " need_choose=" << p["need_choose"] 
         << " selected_experts.size()=" << selected_experts.size() << endl;

    // 判断是否需要重选专家
    int expert_count = p["E_N"];
    if (p["need_choose"]) {
        selected_experts.clear();

        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " Selecting experts...";

        if (p["route_mode"]) {
            int global_en = p["route_global_en"];
            int global_k = p["route_global_k"];
            int local_begin = p["route_local_begin"];
            auto chosen_global =
                SampleUniqueLcg(global_en, global_k, (uint32_t)p["route_seed"]);

            for (auto g : chosen_global) {
                if (g >= local_begin && g < local_begin + expert_count) {
                    selected_experts.push_back(g - local_begin);
                }
            }
            if ((int)selected_experts.size() != p["K"]) {
                LOG_ERROR(matmul_forward_moe.cpp)
                    << "route_mode selected_experts size mismatch: "
                    << selected_experts.size() << " != " << p["K"];
                return;
            }
        } else {
            std::vector<bool> exp_flag(expert_count, false);

            for (int i = 0; i < p["K"]; i++) {
                int s_exp;
                do {
                    s_exp = rand() % p["E_N"];
                } while (exp_flag[s_exp]);
                exp_flag[s_exp] = true;
                selected_experts.push_back(s_exp);
            }
        }

        while (selected_freq.size() < p["E_N"])
            selected_freq.push_back(0);

        for (auto e : selected_experts)
            selected_freq[e]++;

    } else {
        if (selected_experts.size() != p["K"]) {
            LOG_ERROR(matmul_forward_moe.cpp)
                << "selected_experts size mismatch: " << selected_experts.size()
                << " != " << p["K"];
            return;
        }
    }

    for (auto e : selected_experts) {
        cout << "Core" << prim_context->cid <<   " selected expert: " << e << endl;
    }

    // 优先查看是否有被prefetch的专家
    std::vector<bool> checked(expert_count, false);

    for (auto e : selected_experts) {
        // if (std::find(prefetched_experts.begin(), prefetched_experts.end(),
        //               e) == prefetched_experts.end())
        //     continue;

        auto label_weight = ETERNAL_PREFIX + prim_name + "_w_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["weight"] +
                            e * GetFromPairedVector(data_chunk, "weight"),
                        GetFromPairedVector(data_chunk, "weight"),
                        label_weight);

        auto label_bias = ETERNAL_PREFIX + prim_name + "_b_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["bias"] +
                            e * GetFromPairedVector(data_chunk, "bias"),
                        GetFromPairedVector(data_chunk, "bias"), label_bias);

        checked[e] = true;
    }

    for (auto e : selected_experts) {
        if (checked[e])
            continue;

        auto label_weight = ETERNAL_PREFIX + prim_name + "_w_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["weight"] +
                            e * GetFromPairedVector(data_chunk, "weight"),
                        GetFromPairedVector(data_chunk, "weight"),
                        label_weight);

        auto label_bias = ETERNAL_PREFIX + prim_name + "_b_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["bias"] +
                            e * GetFromPairedVector(data_chunk, "bias"),
                        GetFromPairedVector(data_chunk, "bias"), label_bias);


        checked[e] = true;
    }

    if (p["is_merge"])
        exu_ops = (u_int64_t)p["B"] * p["T"] * p["C"] * p["OC"] * p["K"] * 2 +
                  (u_int64_t)p["B"] * p["T"] * p["OC"] * p["K"];
    else
        exu_ops = (uint64_t)p["B"] * p["T"] * p["C"] * p["OC"] * p["K"] * 2;

    if (SPEC_USE_PERF_GEMM) {
        ExuConfig *exu = GetCoreHWConfig(context.cid)->exu;

        uint64_t weight_tile_x = (p["C"] + exu->x_dims - 1) / exu->x_dims;
        uint64_t weight_tile_y = (p["OC"] + exu->x_dims - 1) / exu->x_dims;

        uint64_t padding_input_x = (p["T"] * p["B"] * p["K"]) > exu->x_dims
                                       ? p["T"] * p["B"] * p["K"]
                                       : exu->x_dims;

        uint64_t performance_cycle =
            (exu->x_dims + exu->x_dims + padding_input_x) * weight_tile_x *
            weight_tile_y;

        uint64_t performance_comp =
            performance_cycle * exu->x_dims * exu->x_dims * HW_COMP_UTIL;

        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " performance_cycle " << performance_cycle;

        int loop_input_count =
            weight_tile_y - 1; // read loop_input_count Repetitive input

        for (int loop = 0; loop < loop_input_count; loop++) {
            for (int p = 0; p < data_size_input.size(); p++) {
                if (prim_context->datapass_label_->indata[p].find(DRAM_LABEL) ==
                    0) {

                    prefReadData(context, dram_time, data_size_input[p],
                                 prim_context->datapass_label_->indata[p]);
                }
            }
        }

        exu_ops = performance_comp;
    }

    cout << "Core" << prim_context->cid << " selected experts: " << endl;
}
