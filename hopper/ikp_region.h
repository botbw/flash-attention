#pragma once
#include <cstdint>

// Shared IKP region-id encoding, included by both flash_fwd_kernel_sm90.h and
// mainloop_fwd_sm90_tma_gmma_ws.hpp so the __forceinline__ definition is visible
// in every translation unit that records IKP spans (avoids ptxas "unresolved
// extern function" when an instantiation .cu pulls in the mainloop but not the
// kernel header).
//
// region id layout (16 bits):
//   bit15      producer flag (load/TMA warp)
//   bits10-14  phase (5 bits, 32 tags -- see IKP_PHASE_* below)
//   bits 5-9   bidb (5 bits -> batch <= 31)
//   bits 2-4   head (3 bits, aliases mod 8 for >8 q-heads)
//   bits 0-1   split_idx (2 bits, <= 3)
//
// PHASE TAGS (kept in sync with the host PHASE map in fa3_tile_profile_mixed.py
// and the source-line mapping in docs/IKP_PHASE_MAPPING.md). Naming convention:
//   tma_*   = TMA memory transfer (_issue = launch, _sync = data-arrival wait)
//   pipe_*  = smem-slot semaphore (_acq = acquire empty slot, _rel = release)
//   wgmma_* = warpgroup MMA       (_issue = launch, _sync = warpgroup_wait)
// CONSUMER (MMA-warp) phases, ids 0-17:
//   0  tile           whole consumer work-tile (encloses everything below)
//   1  tma_q_sync     barrier_Q.wait -- wait for Q TMA load
//   2  prologue       (reserved/unused)
//   3  tma_k_sync     consumer_wait(pipeline_k) -- wait for K data (full barrier)
//   4  tma_v_sync     consumer_wait(pipeline_v) -- wait for V data
//   5  wgsched_sync   warp_scheduler_barrier_sync -- ping-pong WG scheduler barrier
//   6  wgmma_qk_sync  warpgroup_wait after QK WGMMA (S = Q*K^T completion)
//   7  wgmma_pv_sync  warpgroup_wait after PV WGMMA (O = P*V completion)
//   8  softmax        max_get_scale + online_softmax (+ rescale_o)
//   9  mask           mask.apply (causal/local/seqlenk)
//   10 p_write        write_P_to_smem + arrive_on_P_write_barrier
//   11 pipe_k_rel     pipeline_k.consumer_release (signal K slot empty)
//   12 pipe_v_rel     pipeline_v.consumer_release (signal V slot empty)
//   13 epilogue       epilogue.store (write O + LSE)   [kernel]
//   14 sched          get_next_work (tile scheduler)   [kernel]
//   15 finalize       finalize_dispatch (softmax finalize / LSE)
//   16 wgmma_qk_issue async QK flash::gemm launch
//   17 wgmma_pv_issue async PV flash::gemm launch
// PRODUCER (load-warp) phases reuse ids (producer bit disambiguates; host map):
//   0  load (whole)   1  pipe_o_acq(barrier_O.wait)  3 pipe_k_acq  4 pipe_v_acq
//   6  tma_k_issue    7  tma_v_issue
#define IKP_PHASE_TILE          0
#define IKP_PHASE_TMA_Q_SYNC    1
#define IKP_PHASE_PROLOGUE      2
#define IKP_PHASE_TMA_K_SYNC    3
#define IKP_PHASE_TMA_V_SYNC    4
#define IKP_PHASE_WGSCHED_SYNC  5
#define IKP_PHASE_WGMMA_QK_SYNC 6
#define IKP_PHASE_WGMMA_PV_SYNC 7
#define IKP_PHASE_SOFTMAX       8
#define IKP_PHASE_MASK          9
#define IKP_PHASE_P_WRITE       10
#define IKP_PHASE_PIPE_K_REL    11
#define IKP_PHASE_PIPE_V_REL    12
#define IKP_PHASE_EPILOGUE      13
#define IKP_PHASE_SCHED         14
#define IKP_PHASE_FINALIZE      15
#define IKP_PHASE_WGMMA_QK_ISSUE 16
#define IKP_PHASE_WGMMA_PV_ISSUE 17
// producer aliases (same numeric ids; producer flag + host map disambiguate)
#define IKP_PHASE_PIPE_O_ACQ    1
#define IKP_PHASE_PIPE_K_ACQ    3
#define IKP_PHASE_PIPE_V_ACQ    4
#define IKP_PHASE_TMA_K_ISSUE   6
#define IKP_PHASE_TMA_V_ISSUE   7

namespace flash {
__device__ __forceinline__ uint16_t ikp_region_id(bool producer, int bidb, int head, int split_packed, int phase = 0) {
    int sp = split_packed & 0xFF;   // low byte = split_idx (high byte = num_splits)
    return (uint16_t)((producer ? 0x8000u : 0u)
                      | ((uint32_t(phase) & 0x1Fu) << 10)
                      | ((uint32_t(bidb)  & 0x1Fu) << 5)
                      | ((uint32_t(head)  & 0x7u)  << 2)
                      |  (uint32_t(sp)    & 0x3u));
}
}  // namespace flash
