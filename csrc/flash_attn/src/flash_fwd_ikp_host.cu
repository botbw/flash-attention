// [IKP] intra-kernel profiler host API for FA2 (mirrors hopper fa3_ikp_*).
// Compiled by nvcc (the IKP headers contain device code).  Empty TU unless
// FLASH_ATTENTION_ENABLE_IKP is defined (env FA2_IKP_INCLUDE at build time).
#ifdef FLASH_ATTENTION_ENABLE_IKP
#include <intra_kernel_profiler/trace/trace.cuh>
#include <string>
#include <vector>

namespace {
    intra_kernel_profiler::trace::HostSession g_fa2_ikp_sess;
    bool g_fa2_ikp_armed = false;
    std::vector<std::string> g_fa2_ikp_region_names;
}

extern "C" {

// Allocate device buffers and arm IKP.  Call BEFORE the attention call.
//   per_warp_cap      : events per warp (power of 2; 8 is plenty for FA2 —
//                       each CTA records one begin/end pair on warp leaders)
//   num_ctas          : upper bound on gridDim total CTAs of the FA2 kernel
//   threads_per_block : blockDim.x (128 for the splitkv kernels)
void fa2_ikp_arm(int per_warp_cap, int num_ctas, int threads_per_block) {
    g_fa2_ikp_sess.init(per_warp_cap, num_ctas, threads_per_block);
    g_fa2_ikp_sess.reset();
    g_fa2_ikp_armed = true;
}

void fa2_ikp_set_region_names(const char** names, int n) {
    g_fa2_ikp_region_names.clear();
    for (int i = 0; i < n; ++i) g_fa2_ikp_region_names.emplace_back(names[i]);
    g_fa2_ikp_sess.set_region_names(g_fa2_ikp_region_names);
}

void fa2_ikp_get_bufs(void** events, uint32_t** counters) {
    if (!g_fa2_ikp_armed) { *events = nullptr; *counters = nullptr; return; }
    auto buf = g_fa2_ikp_sess.global_buffer();
    *events   = reinterpret_cast<void*>(buf.events);
    *counters = buf.counters;
}

void fa2_ikp_write_trace(const char* path) {
    cudaDeviceSynchronize();
    intra_kernel_profiler::trace::TraceWriteOptions opts;
    opts.scale = 1.0;  // %globaltimer is in nanoseconds on A100 too
    g_fa2_ikp_sess.write_trace(std::string(path), opts);
    g_fa2_ikp_armed = false;
}

void fa2_ikp_disarm() {
    g_fa2_ikp_armed = false;
}

} // extern "C"
#endif // FLASH_ATTENTION_ENABLE_IKP
