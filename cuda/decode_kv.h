//
// Created by crabo on 2019/11/6.
//

#ifndef LEVELDB_CUDA_DECODE_KV_H
#define LEVELDB_CUDA_DECODE_KV_H

#include <vector>

#include "cuda/util.h"
#include "cuda/data.h"
#include "cuda/cuda_common.h"
#include "cuda/format.h"
#include "db/version_set.h"

namespace leveldb {
namespace gpu {

// allocate memory for the host and GPU
// maximum mem size is SST_SIZE * CUDA_MAX_COMPACTION_FILES / SST_SIZE * CUDA_MAX_COMAPCTION_FILES * 2
__global__
void GPUDecodeKernel(char **SST, int SSTIdx, GDI *gdi, int gdi_cnt, SST_kv *skv,WpSlice* slices=nullptr,int index=0);

// prefix compress for every 16 (kv_count) key-value pairs
// <<<x, y>>>
// SharedIdx = x * dim_y + y
// kv_start = (x * dim_y + y) * kSharedKeys
// SharedBlock kv_count = distance[kv_start, min(kv_start + kSharedKeys, kv_end)]
__global__
void GPUEncodeSharedKernel(SST_kv *skv, SST_kv *skv_new, int base, int skv_cnt, uint32_t *shared_size,SST_kv *l0_d_skv_=nullptr);  // default that Per-Shared have 16 KVs

// Copy the DataBlock and the SharedBlock
// <<<x, y>>>
// SharedIdx = x * dim_y + y
// kv_start = (x * dim_y + y) * kSharedKeys
// SharedBlock kv_count = distance[kv_start, min(kv_start + kSharedKeys, kv_end)]
// DataBlock restarts[] Write
__global__
void GPUEncodeCopyShared(char **SST, char *SST_new, SST_kv *skv, int base_idx, int skv_cnt,
        uint32_t *shared_offset, int shared_cnt, int base = 0,SST_kv* d_skv=nullptr,SST_kv *l0_d_skv_=nullptr);

// <<<x, y>>>
// idx = x * dim_y + y
__global__
void GPUEncodeFilter(char *SST_new, SST_kv *skv, filter_meta *fmeta, int f_cnt, int k, int base = 0);

}
}
#endif //LEVELDB_CUDA_DECODE_KV_H
