//
// Created by crabo on 2019/11/4.
//

#include <stdio.h>
#include <unistd.h>

#include "util/crc32c.h"

#include "/home/xp/flying/moderngpu/src/moderngpu/kernel_merge.hxx"
#include "/home/xp/flying/moderngpu/src/moderngpu/kernel_mergesort.hxx"
#include "cuda/data.h"
#include "cuda/decode_kv.h"
#include "cuda/format.h"
#include "cuda/util.cu"
#include "cuda/util.h"

#define M (512 * (__SST_SIZE / (1024 * 1024 * 4)))
#define N (32)
using namespace mgpu;

namespace leveldb {
namespace gpu {

// Debug
__global__ void __test(const char* src, size_t cnt) {
  for (int i = 0; i < cnt; ++i) {
    if (src[i] == 0) {
      printf("i\n", i);
      assert(0);
    }
  }
}
void Debug::Test(const char* src, size_t cnt) { __test<<<1, 1>>>(src, cnt); }

// CudaStream wrapper
Stream::Stream() { cudaStreamCreate((cudaStream_t*)&s_); }

Stream::~Stream() { cudaStreamDestroy((cudaStream_t)s_); }

void Stream::Sync() { cudaStreamSynchronize((cudaStream_t)s_); }

__host__ HostAndDeviceMemory::HostAndDeviceMemory() {
  char* ptr[CUDA_MAX_COMPACTION_FILES];
  printf("max-key:%d max-gdi:%d blk:%d M:%d MaxOutFile:%d\n",
         CUDA_MAX_KEY_PER_SST, CUDA_MAX_GDI_PER_SST,
         kSharedPerSST / kDataSharedCnt, M, 15 * 1024 * 1024);
  // allocate mem for each sst
  for (int i = 0; i < CUDA_MAX_COMPACTION_FILES; ++i) {
    char *ph_SST, *pd_SST, *pd_SST_new;
    GDI *ph_gdi, *pd_gdi;
    SST_kv *ph_skv, *pd_skv;
    uint32_t *ph_shared_size, *pd_shared_size;
    uint32_t *ph_so, *pd_so;
    filter_meta *ph_fm, *pd_fm;

    ph_SST = (char*)malloc(__SST_SIZE + 100 * 1024);  // 100KB spare mem for each sst
    ph_gdi = (GDI*)malloc(sizeof(GDI) * CUDA_MAX_GDI_PER_SST);
    ph_skv = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
    ph_shared_size = (uint32_t*)malloc(sizeof(uint32_t) * CUDA_MAX_GDI_PER_SST);
    ph_so = (uint32_t*)malloc(sizeof(uint32_t) * CUDA_MAX_GDI_PER_SST);
    ph_fm = (filter_meta*)malloc(sizeof(filter_meta) * CUDA_MAX_GDI_PER_SST);
    assert(ph_SST && ph_gdi && ph_skv && ph_shared_size && ph_so && ph_fm);

    cudaMalloc((void**)&pd_SST, __SST_SIZE + 100 * 1024);
    cudaMalloc((void**)&pd_SST_new, __SST_SIZE + 100 * 1024);
    cudaMalloc((void**)&pd_gdi, sizeof(GDI) * CUDA_MAX_GDI_PER_SST);
    cudaMalloc((void**)&pd_skv, sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
    cudaMalloc((void**)&pd_shared_size,
               sizeof(uint32_t) * CUDA_MAX_GDI_PER_SST);
    cudaMalloc((void**)&pd_so, sizeof(uint32_t) * CUDA_MAX_GDI_PER_SST);
    cudaMalloc((void**)&pd_fm, sizeof(filter_meta) * CUDA_MAX_GDI_PER_SST);
    assert(pd_SST && pd_gdi && pd_skv && pd_shared_size && pd_so);

    h_SST.push_back(ph_SST);
    h_gdi.push_back(ph_gdi);
    h_skv.push_back(ph_skv);
    h_shared_size.push_back(ph_shared_size);
    h_shared_offset.push_back(ph_so);
    h_fmeta.push_back(ph_fm);

    d_SST.push_back(pd_SST);
    ptr[i] = pd_SST;
    d_SST_new.push_back(pd_SST_new);
    d_gdi.push_back(pd_gdi);
    d_skv.push_back(pd_skv);
    d_shared_size.push_back(pd_shared_size);
    d_shared_offset.push_back(pd_so);
    d_fmeta.push_back(pd_fm);
  }

  SST_kv* L0_skv;
  for (int i = 0; i < 10; i++) {
    L0_skv = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
    q.push(L0_skv);
  }

  low_size = 50000;
  high_size = 150000;
  result_size = 200000;
  cudaMallocHost((void**)&lowSlices, sizeof(WpSlice) * low_size);
  cudaMallocHost((void**)&highSlices, sizeof(WpSlice) * high_size);
  cudaMalloc((void**)&resultSlice, sizeof(WpSlice) * result_size);

  cudaMalloc((void**)&L0_d_skv_sorted,
             sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);

  cudaMalloc((void**)&L0_d_skv_sorted_2,
             sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);

  L0_h_skv_sorted = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
  h_skv_sorted = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
  cudaMalloc((void**)&d_skv_sorted, sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
  cudaMalloc((void**)&d_skv_sorted_shared,
             sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
  cudaMalloc((void**)&L0_d_skv_sorted_shared,
             sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
  assert(h_skv_sorted && d_skv_sorted && d_skv_sorted_shared);

  cudaMalloc((void**)&d_SST_ptr, sizeof(char*) * CUDA_MAX_COMPACTION_FILES);
  cudaMemcpy((void*)d_SST_ptr, (void*)ptr,
             sizeof(char*) * CUDA_MAX_COMPACTION_FILES, cudaMemcpyHostToDevice);
}

__host__ HostAndDeviceMemory::~HostAndDeviceMemory() {
  cudaDeviceSynchronize();

  for (int i = 0; i < CUDA_MAX_COMPACTION_FILES; ++i) {
    free(h_SST[i]);
    free(h_gdi[i]);
    free(h_skv[i]);
    free(h_shared_size[i]);
    free(h_shared_offset[i]);
    free(h_fmeta[i]);

    cudaFree(d_SST[i]);
    cudaFree(d_SST_new[i]);
    cudaFree(d_gdi[i]);
    cudaFree(d_skv[i]);
    cudaFree(d_shared_size[i]);
    cudaFree(d_shared_offset[i]);
    cudaFree(d_fmeta[i]);
  }

  free(h_skv_sorted);
  free(L0_h_skv_sorted);

  cudaFreeHost(lowSlices);
  cudaFreeHost(highSlices);
  cudaFree(resultSlice);

  cudaFree(L0_d_skv_sorted);
  cudaFree(L0_d_skv_sorted_2);

  cudaFree(d_skv_sorted);
  cudaFree(d_skv_sorted_shared);
  cudaFree(L0_d_skv_sorted_shared);

  for (auto it = l0_hkv.begin(); it != l0_hkv.end(); it++) {
    free(it->second);
  }
  SST_kv* temp;
  while (!q.empty()) {
    temp = q.front();
    q.pop();
    free(temp);
  }
}

//////////// Decodde /////////////////////////////
__host__ void SSTDecode::DoDecode() {
  // 1. Read the footer to find index-block
  Slice footer_slice(h_SST_ + file_size_ - leveldb::Footer::kEncodedLength,
                     leveldb::Footer::kEncodedLength);
  Footer footer;
  footer.DecodeFrom(&footer_slice);

  // 2. Iterator index-block and decode it to GDI
  char* contents = h_SST_ + footer.index_handle_.offset_;
  size_t contents_size = footer.index_handle_.size_;

  // TODO: crc32c checksums. No Compression

  uint32_t index_num =
      DecodeFixed32((const char*)(contents + contents_size - sizeof(uint32_t)));
  uint32_t* index_restart = (uint32_t*)(contents + contents_size -
                                        sizeof(uint32_t) * (1 + index_num));

  // 2.1 Iterate all the restart array to get all DataBlock offset and
  // size(don't contain type+CRC32)
  for (uint32_t i = 0; i < index_num; ++i) {
    const char* p = contents + (index_restart[i] >> 8);
    const char* limit = (const char*)index_restart;
    uint32_t shared, non_shared, value_length;

    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    assert(shared == 0);  // no shared prefix in index-block

    Slice key(p, non_shared);  // the minimal key in this DataBlock
    Slice value(p + non_shared, value_length);
    BlockHandle data;
    data.DecodeFrom(&value);

    uint32_t data_restart_num =
        DecodeFixed32(h_SST_ + data.offset_ + data.size_ - sizeof(uint32_t));
    uint32_t* array = (uint32_t*)(h_SST_ + data.offset_ + data.size_ -
                                  sizeof(uint32_t) * (data_restart_num + 1));

    assert(data_restart_num <= 3200);
    for (uint32_t j = 0; j < data_restart_num; ++j) {
      uint32_t cnt = array[j] & 0xff;
      uint32_t off = array[j] >> 8;

      h_gdi_[shared_cnt_ + j].offset = data.offset_ + off;
      h_gdi_[shared_cnt_ + j].kv_base_idx = all_kv_;

      if (j + 1 < data_restart_num) {
        h_gdi_[shared_cnt_ + j].limit = data.offset_ + (array[j + 1] >> 8);
      } else {
        h_gdi_[shared_cnt_ + j].limit =
            data.offset_ + data.size_ -
            sizeof(uint32_t) * (data_restart_num + 1);
      }

      all_kv_ += cnt;
    }
    shared_cnt_ += data_restart_num;
  }
}

__host__ void SSTDecode::DoGPUDecode() {
  cudaMemcpy(d_SST_, h_SST_, file_size_, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gdi_, h_gdi_, sizeof(GDI) * shared_cnt_, cudaMemcpyHostToDevice);

  // cudaMemcpy: h_SST, h_gdi,  == > GPU
  GPUDecodeKernel<<<M, N>>>(d_SST_ptr_, SST_idx_, d_gdi_, shared_cnt_, d_skv_);

  // cudaMemcpy  h_skv_         <=== GPU
  cudaMemcpy(h_skv_, d_skv_, sizeof(SST_kv) * all_kv_, cudaMemcpyDeviceToHost);
}

__host__ void SSTDecode::Copy() {
  cudaStream_t s = (cudaStream_t)s_.data();
  for (int i = 0; i < all_kv_; i++) {
    SST_kv* pskv = &h_skv_[i];
    EncodeValueOffset(&pskv->value_offset, SST_idx_);
  }
  cudaMemcpyAsync(d_SST_, h_SST_, file_size_, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_skv_, h_skv_, sizeof(SST_kv) * all_kv_,
                  cudaMemcpyHostToDevice, s);
  s_.Sync();
}

__host__ void SSTDecode::DoGPUDecode_1(WpSlice* slices, int index) {
  cudaStream_t s = (cudaStream_t)s_.data();
  cudaMemcpyAsync(d_SST_, h_SST_, file_size_, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_gdi_, h_gdi_, sizeof(GDI) * shared_cnt_,
                  cudaMemcpyHostToDevice, s);

  GPUDecodeKernel<<<M, N, 0, s>>>(d_SST_ptr_, SST_idx_, d_gdi_, shared_cnt_,
                                  d_skv_, slices, index);
}

__host__ void SSTDecode::DoGPUDecode_2(WpSlice* slices, int index) {
  cudaStream_t s = (cudaStream_t)s_.data();
  cudaMemcpyAsync(h_skv_, d_skv_, sizeof(SST_kv) * all_kv_,
                  cudaMemcpyDeviceToHost, s);
  s_.Sync();
  if (slices) {
    for (int i = 0; i < all_kv_; i++) {
      slices[index + i].data_ = h_skv_[i].ikey;
    }
  }
}

// align in 32 bytes to boost mem copy
__device__ void __copy_mm(char* dst, const char* src, size_t n) {
  int i, idx;
  int x = threadIdx.x;
  uint32_t BYTE = 0x20;  // copy use 4BYTEs
  uint32_t d = (uint32_t)(dst) & (BYTE - 1);
  uint32_t head = (BYTE - d) < n ? BYTE - d : n;
  head = head % BYTE;
  uint32_t cc = n - head;
  uint32_t tail = cc % BYTE;
  uint32_t align = (n - head - tail) / 4;

  assert(align * 4 == n - head - tail);
  assert(blockDim.x <= BYTE);

  // Head
  if (threadIdx.x < head) {
    dst[threadIdx.x] = src[threadIdx.x];
  }

  // Middle aligned Data
  const char* from = src + head;
  char* to = dst + head;
  uint32_t* to4 = (uint32_t*)to;
  uint32_t tmp;

  for (i = 0; i < align / blockDim.x; ++i) {
    char* pt = (char*)&tmp;
    pt[0] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 0];
    pt[1] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 1];
    pt[2] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 2];
    pt[3] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 3];

    to4[i * blockDim.x + threadIdx.x] = tmp;
  }

  if (i * blockDim.x + threadIdx.x < align) {
    char* pt = (char*)&tmp;
    pt[0] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 0];
    pt[1] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 1];
    pt[2] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 2];
    pt[3] = from[i * blockDim.x * 4 + threadIdx.x * 4 + 3];

    to4[i * blockDim.x + threadIdx.x] = tmp;
  }

  // Tail
  idx = head + align * 4 + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

__device__ void __MemcpyKernel(char* dst, const char* src, size_t n) {
  int i = 0;
  int x = threadIdx.x;
  int y = blockDim.x;

  for (i = 0; i < n / y; ++i) {
    dst[i * y + x] = src[i * y + x];
  }

  if (i * y + x < n) {
    dst[i * y + x] = src[i * y + x];
  }
}

__global__ void MemcpyKernel(SST_kv* skv, int kv_cnt, char* base, char** SST) {
  int cur = 0;

  for (int i = 0; i < kv_cnt; ++i) {
    int idx;
    SST_kv* pskv = skv + i;

    cur += pskv->key_size;
    DecodeValueOffset(&pskv->value_offset, &idx);
    char* value = SST[idx] + pskv->value_offset;
    __MemcpyKernel(base + cur, value, pskv->value_size);
    cur += pskv->value_size;
  }
}

__global__ void GPUDecodeKernel(char** SST, int SSTIdx, GDI* gdi, int gdi_cnt,
                                SST_kv* skv, WpSlice* slices, int index) {
  int v_gdi_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (v_gdi_index >= gdi_cnt) {
    return;
  }

  uint32_t kv_idx = 1;
  GDI* cur = &gdi[v_gdi_index];
  char* d_SST = SST[SSTIdx];
  const char* p = d_SST + cur->offset;
  const char* limit = d_SST + cur->limit;
  SST_kv* pskv = &skv[cur->kv_base_idx];
  uint32_t shared, non_shared, value_length;

  // 1. Decode first KeyValue
  p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
  assert(p && !shared);
  Memcpy(pskv->ikey, p, non_shared);
  pskv->key_size = shared + non_shared;
  pskv->value_offset = p + non_shared - d_SST;
  EncodeValueOffset(&pskv->value_offset, SSTIdx);
  pskv->value_size = value_length;
  p += non_shared + value_length;

  if (slices) {
    slices[cur->kv_base_idx].ikey = pskv->ikey;
    slices[cur->kv_base_idx].key_size = pskv->key_size;
    slices[cur->kv_base_idx].value_offset = pskv->value_offset;
    slices[cur->kv_base_idx].value_size = pskv->value_size;
  }

  // 2. Decode the last keys
  while (p < limit) {
    pskv = &skv[cur->kv_base_idx + kv_idx];

    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    assert(p);

    Memcpy(pskv->ikey, (pskv - 1)->ikey, shared);  // copy the last Shared-Key
    Memcpy(pskv->ikey + shared, p, non_shared);
    pskv->key_size = shared + non_shared;
    pskv->value_offset = p + non_shared - d_SST;
    EncodeValueOffset(&pskv->value_offset, SSTIdx);
    pskv->value_size = value_length;

    p = p + non_shared + value_length;

    if (slices) {
      slices[cur->kv_base_idx + kv_idx].ikey = pskv->ikey;
      slices[cur->kv_base_idx + kv_idx].key_size = pskv->key_size;
      slices[cur->kv_base_idx + kv_idx].value_offset = pskv->value_offset;
      slices[cur->kv_base_idx + kv_idx].value_size = pskv->value_size;
    }
    ++kv_idx;
  }
}

/*
 * @skv: the SORTED KVs put in here, full key
 * @skv_new: the shared KVs, partial key
 *
 * <<<x, y>>>
 * kv_start = base + (x * Dim_y + y) * 16;
 * kv_end = min(kv_start + 16, kv_max_cnt);
 */
__global__ void GPUEncodeSharedKernel(SST_kv* skv, SST_kv* skv_new, int base,
                                      int skv_cnt, uint32_t* shared_size,
                                      SST_kv* l0_d_skv_) {
  int kv_start =
      base + (::blockIdx.x * ::blockDim.x + ::threadIdx.x) * kSharedKeys;
  int kv_count = kSharedKeys <= skv_cnt + base - kv_start
                     ? kSharedKeys
                     : skv_cnt + base - kv_start;

  SST_kv* pf = &skv_new[kv_start];
  int fkey_size = skv[kv_start].key_size;  // The first or the last
  Buffer fbuf(pf->ikey, kKeyBufferSize);   // the First-Key, and the Last-Key
  int total_size = 0;

  if (kv_start - base >= skv_cnt) return;

  // 1. Encode the First-Key
  PutVarint32(&fbuf, 0);
  PutVarint32(&fbuf, fkey_size);
  PutVarint32(&fbuf, skv[kv_start].value_size);
  Memcpy(fbuf.now(), skv[kv_start].ikey, fkey_size);
  pf->key_size = fbuf.size_ + fkey_size;
  pf->value_offset = skv[kv_start].value_offset;
  pf->value_size = skv[kv_start].value_size;

  total_size += pf->key_size + pf->value_size;

  if (l0_d_skv_) {
    Memcpy(l0_d_skv_[kv_start].ikey, skv[kv_start].ikey,
           skv[kv_start].key_size);
    l0_d_skv_[kv_start].key_size = skv[kv_start].key_size;
    l0_d_skv_[kv_start].value_size = skv[kv_start].value_size;
  }

  // 2. Encode the last keys.
  // Odd idx use key_buf[0] for LAST, and use key_buf[1] for NOW
  // even idx use key_buf[1] for LAST, and so on
  for (int i = 1; i < kv_count; ++i) {
    SST_kv* pskv = &skv[kv_start + i];
    char* key_last = skv[kv_start + i - 1].ikey;

    SST_kv* pskv_new = &skv_new[kv_start + i];
    Buffer buf(pskv_new->ikey, kKeyBufferSize);

    int key_size = pskv->key_size;
    int value_size = pskv->value_size;
    int shared = 0, non_shared = 0;

    while (shared < key_size && shared < fkey_size &&
           pskv->ikey[shared] == key_last[shared]) {
      shared++;
    }

    non_shared = key_size - shared;
    PutVarint32(&buf, shared);
    PutVarint32(&buf, non_shared);
    PutVarint32(&buf, value_size);
    Memcpy(buf.now(), pskv->ikey + shared, non_shared);
    pskv_new->key_size = buf.size_ + non_shared;
    pskv_new->value_size = pskv->value_size;
    pskv_new->value_offset = pskv->value_offset;

    if (l0_d_skv_) {
      Memcpy(l0_d_skv_[kv_start + i].ikey, skv[kv_start + i].ikey,
             skv[kv_start + i].key_size);
      l0_d_skv_[kv_start + i].key_size = skv[kv_start + i].key_size;
      l0_d_skv_[kv_start + i].value_size = skv[kv_start + i].value_size;
    }

    total_size += pskv_new->key_size + pskv_new->value_size;

    fkey_size = key_size;  // save the LAST-KEY size
  }

  shared_size[(kv_start - base) / kSharedKeys] = total_size;
}

/*
 * calc. CRC32 for data blocks in parallel
 * <<<grid, (4, y)>>> :
 * 4 threads for CRC32 of one data block
 */
__global__ void GPUEncodeCRC32(char* SST_new, uint32_t* shared_offset,
                               uint32_t* shared_size, int shared_cnt) {
  int db_idx = blockIdx.x * blockDim.y + threadIdx.y;  // db : DataBlock
  int shared_start = db_idx * kDataSharedCnt;
  int i;
  uint32_t crc32c_size = 0;

  assert(blockDim.x == 4);

  if (shared_start >= shared_cnt) {
    return;
  }

  for (i = 0; i < kDataSharedCnt && shared_start + i < shared_cnt; ++i) {
    crc32c_size += shared_size[shared_start + i];
  }
  crc32c_size += sizeof(uint32_t) * (i + 1) + 1;  // restarts[] array + Type
  uint32_t boffset = shared_offset[shared_start] >> 8;
  char* base = SST_new + boffset;

  base[crc32c_size - 1] = 0;  // Type NoCompression
  uint32_t result = gpu_crc32c::Extend(0, base, crc32c_size);

  if (threadIdx.x % 4 != 0) {
    return;
  }

  Buffer crc_buf(base + crc32c_size, sizeof(uint32_t));
  PutFixed32(&crc_buf, gpu_crc32c::Mask(result));
}

__global__ void GPUEncodeCRC32_base(char* base, size_t n) {
  assert(blockDim.x == 4);

  base[n] = 0;  // Type NoCompression
  uint32_t result = gpu_crc32c::Extend(0, base, n);

  if (threadIdx.x % 4 != 0) {
    return;
  }

  Buffer crc_buf(base + n + 1, sizeof(uint32_t));
  PutFixed32(&crc_buf, gpu_crc32c::Mask(result));
}

/*
 * Every DataBlock : kSharedKeys * kSharedCnt = 16 * 3 = 48, so, the 2, 5, 8 and
 * so on will write the restarts[] 0 1 2 3 4 5      SharedIdx 0     1
 * DataBlockIdx
 *
 */
__global__ void GPUEncodeCopyShared(char** SST, char* SST_new, SST_kv* skv,
                                    int base_idx, int skv_cnt,
                                    uint32_t* shared_offset, int shared_cnt,
                                    int __base, SST_kv* d_skv,
                                    SST_kv* l0_d_skv) {
  int shared_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int kv_start = base_idx + shared_idx * kSharedKeys;
  int kv_cnt = kSharedKeys <= skv_cnt + base_idx - kv_start
                   ? kSharedKeys
                   : skv_cnt + base_idx - kv_start;

  if (shared_idx >= shared_cnt) {
    return;
  }

  char* base = SST_new + (shared_offset[shared_idx] >> 8);
  uint32_t cur = 0;

  // copy KV from OLDPlace to NEWPlace
  for (int i = 0; i < kv_cnt; ++i) {
    SST_kv* pskv = &skv[kv_start + i];
    int idx;
    __copy_mm(base + cur, pskv->ikey, pskv->key_size);
    cur += pskv->key_size;

    // Copy Value From Old TO New, can be replace by MemcpyKernel
    DecodeValueOffset(&pskv->value_offset, &idx);
    char* value = SST[idx] + pskv->value_offset;
    __copy_mm(base + cur, value, pskv->value_size);
    if (l0_d_skv) {
      l0_d_skv[kv_start + i].value_offset = base + cur - SST_new;
    }

    if (value[0] == 0) {
      printf(" cur:%d %d\n", pskv->value_offset, idx);
      assert(0);
    }
    cur += pskv->value_size;
  }

  if ((shared_idx + 1) % kDataSharedCnt == 0 ||
      (shared_idx == shared_cnt - 1)) {
    Buffer buf(base + cur, sizeof(uint32_t) * (kDataSharedCnt + 1));
    int shared_start = shared_idx - shared_idx % kDataSharedCnt;
    uint32_t boffset = shared_offset[shared_start] >> 8;
    uint32_t crc32c_size = 0;

    for (int i = 0; i < kDataSharedCnt && shared_start + i < shared_cnt; ++i) {
      uint32_t tmp = ((shared_offset[shared_start + i] >> 8) - boffset) << 8;
      tmp |= shared_offset[shared_start + i] & 0xff;
      PutFixed32(&buf, tmp);

      crc32c_size = (shared_offset[shared_start + i] >> 8) - boffset;
    }
    PutFixed32(&buf, shared_idx - shared_start + 1);

    if (0) {
      cur += buf.size_;

      assert(blockDim.y >= kDataSharedCnt);
      __syncthreads();
      *(base + cur) = 0;
      crc32c_size += cur + 1;
      uint32_t result = gpu_crc32c::Extend(0, SST_new + boffset, crc32c_size);
      Buffer crc_buf(base + cur + 1, sizeof(uint32_t));
      PutFixed32(&crc_buf, gpu_crc32c::Mask(result));
    }
  }
}

/*
 * @skv : FULL key
 */
__global__ void GPUEncodeFilter(char* SST_new, SST_kv* skv, filter_meta* fmeta,
                                int f_cnt, int k, int __base) {
  int idx = ::blockIdx.x * ::blockDim.x + ::threadIdx.x + __base;
  if (idx >= f_cnt) return;

  char* base = SST_new + fmeta[idx].offset;
  size_t bits = fmeta[idx].cnt * kBitsPerKey;

  char tmp[128];
  char tkey[32];

  if (bits < 64) bits = 64;

  size_t bytes = (bits + 7) / 8;
  assert(bytes < 128);
  bits = bytes * 8;
  assert(bytes == fmeta[idx].filter_size - 1);

  tmp[bytes] = static_cast<char>(k);
  char* array = tmp;
  for (int i = 0; i < fmeta[idx].cnt; i++) {
    int kv_idx = fmeta[idx].start + i;
    uint32_t h =
        Hash(skv[kv_idx].ikey, skv[kv_idx].key_size - 8);  // Use User-Key
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (size_t j = 0; j < k; j++) {
      const uint32_t bitpos = h % bits;
      array[bitpos / 8] |= (1 << (bitpos % 8));
      h += delta;
    }
  }

  Memcpy(base, tmp, bytes + 1);
}

///////////// Sort /////////////////////////
__host__ Slice SSTSort::GetCurrent(std::vector<SST_kv*>& skvs,
                                   std::vector<int>& idxs,
                                   std::vector<int>& sizes, int& sst_idx) {
  if (sst_idx >= skvs.size()) {
    return Slice(NULL, 0);
  }

  SST_kv* pskv = skvs[sst_idx];
  int skv_idx = idxs[sst_idx];

  if(skv_idx >= sizes[sst_idx])
  {
    return Slice(NULL, 0);
  }
  assert(skv_idx < sizes[sst_idx]);

  return Slice(pskv[skv_idx].ikey, pskv[skv_idx].key_size,
               pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
}
__host__ void SSTSort::Next(std::vector<SST_kv*>& skvs, std::vector<int>& idxs,
                            std::vector<int>& sizes, int& sst_idx) {
  if (sst_idx >= skvs.size()) {
    return;
  }

  ++idxs[sst_idx];
  if (idxs[sst_idx] >= sizes[sst_idx]) {
    ++sst_idx;
  }
}

__host__ Slice SSTSort::FindLowSmallest() {
  if (witch_ == enLow) {
    Next(low_skvs_, low_idx_, low_sizes_, low_sst_index_);
  }
  return GetCurrent(low_skvs_, low_idx_, low_sizes_, low_sst_index_);
}

__host__ Slice SSTSort::FindHighSmallest() {
  if (witch_ == enHigh) {
    Next(high_skvs_, high_idx_, high_sizes_, high_sst_index_);
  }
  return GetCurrent(high_skvs_, high_idx_, high_sizes_, high_sst_index_);
}

__host__ Slice SSTSort::FindL0Smallest() {
  Slice min_key(NULL, 0);

  if (witch_ == enL0 || witch_ == enLow) {
    ++l0_idx_[l0_sst_index_];
  }

  l0_sst_index_ = -1;  // init the it to -1

  for (int i = 0; i < l0_skvs_.size(); ++i) {
    SST_kv* pskv = l0_skvs_[i];
    int skv_idx = l0_idx_[i];

    if (skv_idx >= l0_sizes_[i]) {
      continue;
    }

    if (l0_sst_index_ == -1) {
      min_key = Slice(pskv[skv_idx].ikey, pskv[skv_idx].key_size,
                      pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
      l0_sst_index_ = i;
    } else {
      Slice key_cur(pskv[skv_idx].ikey, pskv[skv_idx].key_size,
                    pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
      int r = key_cur.internal_compare(min_key);
      if (r < 0) {
        min_key = key_cur;
        l0_sst_index_ = i;
      }
    }
  }

  return min_key;
}

MGPU_HOST_DEVICE bool WpSlice::operator<(WpSlice& b) {
  size_t min_len = (key_size < b.key_size) ? key_size : b.key_size;
  min_len -= 8;
  for (int i = 0; i < min_len; i++) {
    if (ikey[i] < b.ikey[i]) {
      return true;
    }
    if (ikey[i] > b.ikey[i]) {
      return false;
    }
  }
  if (key_size < b.key_size) {
    return true;
  }
  if (key_size > b.key_size) {
    return false;
  }
  uint64_t anum, bnum;
  Memcpy((char*)&anum, ikey + key_size - 8, sizeof(anum));
  Memcpy((char*)&bnum, b.ikey + b.key_size - 8, sizeof(bnum));

  return anum < bnum;
}
void SSTSort::AllocLow(int size, HostAndDeviceMemory* m) {
  if (size <= m->low_size) {
    low_slices = m->lowSlices;
  } else {
    cudaFreeHost(m->lowSlices);
    cudaMallocHost((void**)&(m->lowSlices), sizeof(WpSlice) * size);
    m->low_size = size;
    low_slices = m->lowSlices;
  }
}
void SSTSort::AllocHigh(int size, HostAndDeviceMemory* m) {
  if (size <= m->high_size) {
    high_slices = m->highSlices;
  } else {
    cudaFreeHost(m->highSlices);
    cudaMallocHost((void**)&(m->highSlices), sizeof(WpSlice) * size);
    m->high_size = size;
    high_slices = m->highSlices;
  }
}

void SSTSort::AllocResult(int size, HostAndDeviceMemory* m) {
  if (size <= m->result_size) {
    result_slices = m->resultSlice;
  } else {
    cudaFree(m->resultSlice);
    cudaMalloc((void**)&(m->resultSlice), sizeof(WpSlice) * size);
    m->result_size = size;
    result_slices = m->resultSlice;
  }
}

standard_context_t context;

void SSTSort::WpSort() {
  WpSlice last_user_key;
  last_user_key.data_ = nullptr;
  uint64_t last_seq = kMaxSequenceNumber;
  WpSlice* ctest = nullptr;
  if (low_num != 0 && high_num != 0) {
    merge(low_slices, low_num, high_slices, high_num, result_slices,
          mgpu::less_t<WpSlice>(), context);
    ctest = result_slices;
  } else if (high_num == 0) {
    ctest = low_slices;
  } else if (low_num == 0) {
    ctest = high_slices;
  } else {
    out_size_ = 0;
    return;
  }
  std::vector<WpSlice> c_host;
  cudaError_t result = dtoh(c_host, ctest, num);
  if (cudaSuccess != result) throw cuda_exception_t(result);
  for (int i = 0; i < num; i++) {
    bool drop = false;
    if (last_user_key.data_) {
      if (last_user_key.key_size != c_host[i].key_size) {
        last_seq = kMaxSequenceNumber;
      } else {
        for (int j = 0; j < last_user_key.key_size - 8; j++) {
          if (last_user_key.data_[j] != c_host[i].data_[j]) {
            last_seq = kMaxSequenceNumber;
            break;
          }
        }
      }
    }

    last_user_key.data_ = c_host[i].data_;
    last_user_key.key_size = c_host[i].key_size;
    last_user_key.value_size = c_host[i].value_size;
    last_user_key.value_offset = c_host[i].value_offset;

    uint64_t inum;
    Memcpy((char*)&inum, c_host[i].data_ + c_host[i].key_size - 8,
           sizeof(inum));
    uint64_t iseq = inum >> 8;
    uint8_t itype = inum & 0xff;
    if (last_seq <= seq_) {
      drop = true;
    } else if (itype == kTypeDeletion && iseq <= seq_) {
      drop = true;
    }
    last_seq = iseq;
    if (!drop && out_) {
      Memcpy(out_[out_size_].ikey, c_host[i].data_, c_host[i].key_size);
      out_[out_size_].key_size = c_host[i].key_size;
      out_[out_size_].value_size = c_host[i].value_size;
      out_[out_size_].value_offset = c_host[i].value_offset;
      ++out_size_;
    }
  }
}

__host__ void SSTSort::Sort() {
  Slice low_key, high_key, last_user_key;
  uint64_t last_seq = kMaxSequenceNumber;

  while (true) {
    bool drop = false;

    // 1. Iterator the TWO-LEVEL and get the minimum KV
    if (l0_skvs_.empty()) {  // low level not Level0
      low_key = FindLowSmallest();
    } else {
      low_key = FindL0Smallest();
    }
    high_key = FindHighSmallest();

    if (high_key.empty() && low_key.empty()) {
      break;
    }

    if (low_key.empty()) {
      low_key = high_key;
      witch_ = enHigh;
    } else if (high_key.empty()) {
      witch_ = enLow;
    } else if (low_key.internal_compare(high_key) >= 0) {
      low_key = high_key;
      witch_ = enHigh;
    } else {
      witch_ = enLow;
    }

    // 2. Check the key
    Slice min_user_key(low_key.data(), low_key.size() - 8);
    if (!last_user_key.empty() && last_user_key.compare(min_user_key) != 0) {
      // last_user_key = min_user_key;
      last_seq = kMaxSequenceNumber;
    }
    last_user_key = min_user_key;

    uint64_t inum = DecodeFixed64(low_key.data() + low_key.size() - 8);
    uint64_t iseq = inum >> 8;
    uint8_t itype = inum & 0xff;

    if (last_seq <= seq_) {
      drop = true;
    } else if (itype == kTypeDeletion &&
#ifndef __CUDA_DEBUG
               iseq <= seq_) {
#else
               iseq <= seq_ && util_->IsBaseLevelForKey(min_user_key)) {
#endif
      drop = true;
    }

    last_seq = iseq;

    // 3. Write KV to out_
    if (!drop) {
      Memcpy(out_[out_size_].ikey, low_key.data(), low_key.size());
      out_[out_size_].key_size = low_key.size();
      out_[out_size_].value_size = low_key.value_len_;
      out_[out_size_].value_offset = low_key.offset_;

      ++out_size_;
    }
  }
}

/////////// Encode //////////////////////

/*
 * calc for each DataBlock
 */
__host__ void SSTEncode::ComputeDataBlockOffset(int SC) {  // SC: shared_count
  int kv_cnt_last = kv_count_;

  datablock_count_ = (shared_count_ + SC - 1) / SC;
  bmeta_.resize(datablock_count_);

  for (int i = 0; i < datablock_count_; ++i) {
    int cnt = 0, sc = 0;
    uint32_t boffset = cur_;
    h_fmeta_[i].start = kv_count_ - kv_cnt_last + base_;


    for (int j = 0; j < SC; ++j) {  // traverse all SharedBlock of a DataBlock
      int idx = i * SC + j;
      int sc_cnt =
          (kSharedKeys <= kv_cnt_last - cnt) ? kSharedKeys : kv_cnt_last - cnt;
      if (idx >= shared_count_) break;
      ++sc;
      cnt += sc_cnt;
      h_shared_offset_[idx] = (cur_ << 8) | sc_cnt;
      cur_ += h_shared_size_[idx];
    }

    h_fmeta_[i].cnt = cnt;
    kv_cnt_last -= h_fmeta_[i].cnt;

    cur_ += sizeof(uint32_t) * (sc + 1);

    bmeta_[i].offset = boffset;
    bmeta_[i].size = cur_ - boffset;

    cur_ += 5;  // Type, CRC
  }

  assert(kv_cnt_last == 0);
}

/*
 * calc. the FilterMap length and destination offset for each DataBlock
 */
__host__ void SSTEncode::ComputeFilter() {
  std::vector<uint32_t> offsets(cur_ / 2048 + 2, 0);
  int foffset = cur_;

  filter_handle_.offset_ = foffset;

  // traverse all Filter
  for (int i = 0; i < datablock_count_; ++i) {
    filter_meta* pfm = &h_fmeta_[i];

    pfm->offset = cur_;

    size_t bits = pfm->cnt * kBitsPerKey;
    if (bits < 64) bits = 64;
    pfm->filter_size = (bits + 7) / 8 + 1;

    assert(bmeta_[i].offset / 2048 + 1 < offsets.size());
    offsets[bmeta_[i].offset / 2048] = cur_ - foffset;

    cur_ += pfm->filter_size;
    offsets[bmeta_[i].offset / 2048 + 1] = cur_ - foffset;
  }

  filter_end_ = cur_;

  // Finish Filter
  Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * (offsets.size() + 1 + 1));
  for (int i = 0; i < offsets.size(); ++i) {
    PutFixed32(&buf, offsets[i]);
  }
  PutFixed32(&buf, filter_end_ - filter_handle_.offset_);
  cur_ += sizeof(uint32_t) * (offsets.size() + 1);

  *(h_SST_ + cur_) = 11;
  cur_ += 1;
  filter_handle_.size_ = cur_ - foffset;

  cur_ += 5;  // Type + CRC
}

__host__ void SSTEncode::WriteIndexAndFooter() {
  footer.metaindex_handle_.offset_ = cur_;
  const char* filter_name = "filter.leveldb.BuiltinBloomFilter2";
  char cbuf[128];

  // meta_handler
  {
    int name_len = strlen(filter_name);
    Buffer data(cbuf, 128);
    filter_handle_.EncodeTo(&data);

    Buffer fbuf(h_SST_ + cur_, 128);
    PutVarint32(&fbuf, 0);
    PutVarint32(&fbuf, name_len);
    PutVarint32(&fbuf, data.size_);
    cur_ += fbuf.size_;

    memcpy(h_SST_ + cur_, filter_name, name_len);
    cur_ += name_len;

    memcpy(h_SST_ + cur_, data.data(), data.size_);
    cur_ += data.size_;

    Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * 2);
    PutFixed32(&buf, 0);
    PutFixed32(&buf, 1);
    cur_ += sizeof(uint32_t) * 2;

    footer.metaindex_handle_.size_ = cur_ - footer.metaindex_handle_.offset_;

    {
      h_SST_[cur_] = 0;
      uint32_t result =
          leveldb::crc32c::Extend(0, h_SST_ + footer.metaindex_handle_.offset_,
                                  footer.metaindex_handle_.size_ + 1);
      Buffer buf(h_SST_ + cur_ + 1, 4);
      PutFixed32(&buf, leveldb::crc32c::Mask(result));
    }

    cur_ += 5;  // Type and CRC
  }

  // index_block
  {
    Buffer kv(cbuf, 128);
    std::vector<uint32_t> restarts;
    footer.index_handle_.offset_ = cur_;

    for (int i = 0; i < datablock_count_; ++i) {
      SST_kv* max_kv = &h_skv_[h_fmeta_[i].start + h_fmeta_[i].cnt - 1 - base_];
      Buffer ibuf(h_SST_ + cur_, 64);

      restarts.push_back(cur_ - footer.index_handle_.offset_);
      // Encode offset_size as VALUE, the min_key as KEY
      kv.reset();
      PutVarint64(&kv, bmeta_[i].offset);
      PutVarint64(&kv, bmeta_[i].size);

      PutVarint32(&ibuf, 0);
      PutVarint32(&ibuf, max_kv->key_size);
      PutVarint32(&ibuf, kv.size_);
      cur_ += ibuf.size_;

      memcpy(h_SST_ + cur_, max_kv->ikey, max_kv->key_size);
      cur_ += max_kv->key_size;

      memcpy(h_SST_ + cur_, kv.data(), kv.size_);
      cur_ += kv.size_;
    }

    Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * (datablock_count_ + 1));
    for (int i = 0; i < datablock_count_; ++i) {
      PutFixed32(&buf, restarts[i] << 8 | 0x01);
    }
    PutFixed32(&buf, datablock_count_);
    cur_ += sizeof(uint32_t) * (datablock_count_ + 1);

    footer.index_handle_.size_ = cur_ - footer.index_handle_.offset_;

    {
      h_SST_[cur_] = 0;
      uint32_t result =
          leveldb::crc32c::Extend(0, h_SST_ + footer.index_handle_.offset_,
                                  footer.index_handle_.size_ + 1);
      Buffer buf(h_SST_ + cur_ + 1, 4);
      PutFixed32(&buf, leveldb::crc32c::Mask(result));
    }

    cur_ += 5;  // Type CRC
  }

  // footer
  {
    Buffer buf(h_SST_ + cur_, 2 * leveldb::BlockHandle::kMaxEncodedLength +
                                  sizeof(uint32_t) * 2);
    footer.EncodeTo(&buf);
    cur_ += 48;
  }
}

__host__ void SSTEncode::DoEncode() {
  // Async
  int k = 7;
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  GPUEncodeSharedKernel<<<M, N, 0, s1>>>(d_skv_, d_skv_new_, base_, kv_count_,
                                         d_shared_size_);
  cudaMemcpyAsync(h_shared_size_, d_shared_size_,
                  sizeof(uint32_t) * shared_count_, cudaMemcpyDeviceToHost, s1);
  cudaStreamSynchronize(s1);

  ComputeDataBlockOffset();
  cudaMemcpyAsync(d_shared_offset_, h_shared_offset_,
                  sizeof(uint32_t) * shared_count_, cudaMemcpyHostToDevice, s1);

  // dim3 block(32, 4), grid(512, 1);
  // dim3 block(32, 16), grid(512, 1);
  dim3 block(32, 16), grid(M, 1);
  GPUEncodeCopyShared<<<grid, block, 0, s1>>>(d_SST_ptr, d_SST_new_, d_skv_new_,
                                              base_, kv_count_,
                                              d_shared_offset_, shared_count_);
  // dim3 cblock(4, 8), cgrid(128, 1);
  // dim3 cblock(4, 8), cgrid(512, 1);
  dim3 cblock(4, 8), cgrid(M, 1);
  GPUEncodeCRC32<<<cgrid, cblock, 0, s1>>>(d_SST_new_, d_shared_offset_,
                                           d_shared_size_, shared_count_);

  int data_blocks_size = cur_;
  ComputeFilter();
  cudaMemcpyAsync(d_fmeta_, h_fmeta_, sizeof(filter_meta) * datablock_count_,
                  cudaMemcpyHostToDevice, s2);
  GPUEncodeFilter<<<M, N, 0, s2>>>(d_SST_new_, d_skv_, d_fmeta_,
                                   datablock_count_, k, 0);

  cudaMemcpyAsync(h_SST_, d_SST_new_, data_blocks_size, cudaMemcpyDeviceToHost,
                  s1);
  cudaMemcpyAsync(
      h_SST_ + filter_handle_.offset_, d_SST_new_ + filter_handle_.offset_,
      filter_end_ - filter_handle_.offset_, cudaMemcpyDeviceToHost, s2);

  WriteIndexAndFooter();

  {
    h_SST_[filter_handle_.offset_ + filter_handle_.size_] = 0x0;
    uint32_t result = leveldb::crc32c::Extend(
        0, h_SST_ + filter_handle_.offset_, filter_handle_.size_);
    result = leveldb::crc32c::Mask(result);
    memcpy(h_SST_ + filter_handle_.offset_ + filter_handle_.size_ + 1, &result,
           4);
  }

  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}

__host__ void SSTEncode::DoEncode_1(bool f) {
  cudaStream_t s1 = (cudaStream_t)s1_.data();
  if (f)
    GPUEncodeSharedKernel<<<M, N, 0, s1>>>(d_skv_, d_skv_new_, base_, kv_count_,
                                           d_shared_size_, l0_d_skv_);
  else
    GPUEncodeSharedKernel<<<M, N, 0, s1>>>(d_skv_, d_skv_new_, base_, kv_count_,
                                           d_shared_size_);
}

__host__ void SSTEncode::DoEncode_2(bool f) {
  cudaStream_t s1 = (cudaStream_t)s1_.data();
  cudaStream_t s2 = (cudaStream_t)s2_.data();

  cudaMemcpyAsync(h_shared_size_, d_shared_size_,
                  sizeof(uint32_t) * shared_count_, cudaMemcpyDeviceToHost, s1);
  cudaStreamSynchronize(s1);


  ComputeDataBlockOffset();
  cudaMemcpyAsync(d_shared_offset_, h_shared_offset_,
                  sizeof(uint32_t) * shared_count_, cudaMemcpyHostToDevice, s1);
  // dim3 block(32, 16), grid(512, 1);
  dim3 block(32, 16), grid(M, 1);
  if (f) {
    GPUEncodeCopyShared<<<grid, block, 0, s1>>>(
        d_SST_ptr, d_SST_new_, d_skv_new_, base_, kv_count_, d_shared_offset_,
        shared_count_, 0, d_skv_, l0_d_skv_);
  } else {
    GPUEncodeCopyShared<<<grid, block, 0, s1>>>(
        d_SST_ptr, d_SST_new_, d_skv_new_, base_, kv_count_, d_shared_offset_,
        shared_count_, 0, d_skv_);
  }

  // dim3 cblock(4, 8), cgrid(512, 1);
  dim3 cblock(4, 8), cgrid(M, 1);
  GPUEncodeCRC32<<<cgrid, cblock, 0, s1>>>(d_SST_new_, d_shared_offset_,
                                           d_shared_size_, shared_count_);
}

__host__ void SSTEncode::DoEncode_3() {
  int k = 7;
  data_blocks_size_ = cur_;
  cudaStream_t s1 = (cudaStream_t)s1_.data();
  cudaStream_t s2 = (cudaStream_t)s2_.data();

  ComputeFilter();
  cudaMemcpyAsync(d_fmeta_, h_fmeta_, sizeof(filter_meta) * datablock_count_,
                  cudaMemcpyHostToDevice, s2);
  GPUEncodeFilter<<<M, N, 0, s2>>>(d_SST_new_, d_skv_, d_fmeta_,
                                   datablock_count_, k, 0);
}

__host__ void SSTEncode::DoEncode_4() {
  cudaStream_t s1 = (cudaStream_t)s1_.data();
  cudaStream_t s2 = (cudaStream_t)s2_.data();

  cudaMemcpyAsync(h_SST_, d_SST_new_, data_blocks_size_, cudaMemcpyDeviceToHost,
                  s1);
  cudaMemcpyAsync(
      h_SST_ + filter_handle_.offset_, d_SST_new_ + filter_handle_.offset_,
      filter_end_ - filter_handle_.offset_, cudaMemcpyDeviceToHost, s2);

  WriteIndexAndFooter();

  {
    h_SST_[filter_handle_.offset_ + filter_handle_.size_] = 0x0;
    uint32_t result = leveldb::crc32c::Extend(
        0, h_SST_ + filter_handle_.offset_, filter_handle_.size_);
    result = leveldb::crc32c::Mask(result);
    memcpy(h_SST_ + filter_handle_.offset_ + filter_handle_.size_ + 1, &result,
           4);
  }

  s1_.Sync();
  s2_.Sync();
}

}  // namespace gpu
}  // namespace leveldb
