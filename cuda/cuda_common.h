#ifndef LEVELDB_CUDA_COMMON_H
#define LEVELDB_CUDA_COMMON_H

#include <assert.h>
#include <map>
#include <queue>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

#define K_SHARED_KEYS (1)
#define CUDA_MAX_COMPACTION_FILES (100)
#define __SST_SIZE (16 * 1024 * 1024) // sst size
#define __MIN_KEY_SIZE (1024 + 32)  // value size
#define CUDA_MAX_KEY_PER_SST (__SST_SIZE / __MIN_KEY_SIZE + 4096)
#define CUDA_MAX_GDI_PER_SST (CUDA_MAX_KEY_PER_SST / K_SHARED_KEYS + 100)
#define CUDA_MAX_KEYS_COMPACTION \
  (CUDA_MAX_KEY_PER_SST * CUDA_MAX_COMPACTION_FILES)

#include "db/version_set.h"

#include "cuda/util.h"

namespace leveldb {
namespace gpu {

enum {
  kKeyBufferSize = 32,          // key size + 8B
  kSharedKeys = K_SHARED_KEYS,  // counts of keys in a SharedBlock
  kDataSharedCnt = 4,           // counts of SharedBlock in a DataBlock
  kBitsPerKey = 10,             // bits per key for filters
  kSharedPerSST = __SST_SIZE / __MIN_KEY_SIZE / kSharedKeys -
                  100,  // counts of SharedBlock in SST
};
struct SST_kv;
class WpSlice {
 public:
  bool operator<(WpSlice& b);

 public:
  uint32_t value_offset;
  int value_size;
  char* data_;
  size_t key_size;
  char* ikey;

  // SST_kv* skv;
};

class Stream {
 public:
  Stream();
  ~Stream();
  void Sync();
  unsigned long data() { return s_; }

  unsigned long s_;
};

class Buffer {
 public:
  Buffer(char* buf, size_t size);

  char* now();
  char* data();
  void reset();

  inline void advance(int n);

  void append(const char* data, size_t size);

  Buffer& operator=(const Buffer&) = default;

  char* base_;
  size_t total_;
  size_t size_;
};

class Slice {
 public:
  // Create an empty slice.

  Slice() : data_(""), size_(0) {}

  // Create a slice that refers to d[0,n-1].

  Slice(const char* d, size_t n) : data_(d), size_(n) {}

  Slice(const char* d, size_t n, uint32_t off, int len)
      : data_(d), size_(n), offset_(off), value_len_(len) {}

  // Create a slice that refers to s[0,strlen(s)-1]

  Slice(const char* s) : data_(s), size_(strlen(s)) {}

  // Intentionally copyable.
  Slice(const Slice&) = default;

  Slice& operator=(const Slice&) = default;

  // Return a pointer to the beginning of the referenced data

  const char* data() const { return data_; }

  // Return the length (in bytes) of the referenced data

  size_t size() const { return size_; }

  // Return true iff the length of the referenced data is zero

  bool empty() const { return size_ == 0; }

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()

  char operator[](size_t n) const {
    assert(n < size());
    return data_[n];
  }

  // Change this slice to refer to an empty array

  void clear() {
    data_ = "";
    size_ = 0;
  }

  // Drop the first "n" bytes from this slice.

  void remove_prefix(size_t n) {
    assert(n <= size());
    data_ += n;
    size_ -= n;
  }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"

  int compare(const Slice& b) const {
    assert(data_ && b.size());

    const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
    int r = memcmp(data_, b.data_, min_len);
    if (r == 0) {
      if (size_ < b.size_)
        r = -1;
      else if (size_ > b.size_)
        r = +1;
    }
    return r;
  }

  int internal_compare(const gpu::Slice& b) const;

  // Return true iff "x" is a prefix of "*this"

  bool starts_with(const Slice& x) const {
    return ((size_ >= x.size_) && (memcmp(data_, x.data_, x.size_) == 0));
  }

 public:
  uint32_t offset_;
  int value_len_;

 private:
  const char* data_;
  size_t size_;
};

class BlockHandle {  // the same as leveldb::BlockHandle
 public:
  BlockHandle() : offset_(0), size_(0) {}
  ~BlockHandle() = default;

  void EncodeTo(Buffer* dst);
  bool DecodeFrom(Slice* input);

  uint64_t offset_;
  uint64_t size_;
};

class Footer {
 public:
  Footer() {}
  ~Footer() = default;

  void EncodeTo(Buffer* dst);

  bool DecodeFrom(Slice* input);

  BlockHandle metaindex_handle_;
  BlockHandle index_handle_;
};

// Basic DataStructure
struct GDI {        // Gpu Decode Info
  uint32_t offset;  // offset within an sst
  uint32_t kv_base_idx;  // the start offset of data in SST_kv[] by the GPU.
                         // unpacked kvs appended to this offset
  uint32_t limit;
};

struct SST_kv {               // SST sorted KV pair
  char ikey[kKeyBufferSize];  // unpacked key, [key + Seq + Type]
                              // packed [shared + non_shared +
                              // value_size + partial_key ] are also stored here.
  uint32_t key_size;  // size of a key

  uint32_t value_offset;  // offset of value, including Varint size
  // uint32_t value_offset_tmp;
  uint32_t value_size;  // size of a value, including Varint size
};

// for GPU
struct filter_meta {
  int start;  // calc. filter SST_kv[] from which kv
  int cnt;    // counts of kvs for filter

  uint32_t offset;  // dest. of this filter
  int filter_size;  // size of this fileter, including the last byte for k_
};

// for CPU, calc. index_block
struct block_meta {
  uint64_t offset;  // the beginning offset of each DataBlock
  uint64_t size;    // size of each DataBlock, exclude type or CRC
};

// mem for the host and gpu
class HostAndDeviceMemory {
 public:
  HostAndDeviceMemory();
  ~HostAndDeviceMemory();

  SST_kv* getL0skv() {
    SST_kv* L0_skv;
    if (q.empty()) {
      L0_skv = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
      return L0_skv;
    }
    L0_skv = q.front();
    q.pop();
    return L0_skv;
  }
  void pushL0skv(SST_kv* L0_skv) { q.push(L0_skv); }

  std::unordered_map<std::string, SST_kv*> l0_hkv;
  std::unordered_map<std::string, int> l0_knum;
  std::queue<SST_kv*> q;

// each element for one unpack or pack meta
  std::vector<char*> h_SST;  // buffer for old and new ssts
  std::vector<char*> d_SST;
  std::vector<char*> d_SST_new;

  std::vector<GDI*> h_gdi;
  std::vector<GDI*> d_gdi;

  std::vector<SST_kv*> h_skv;
  std::vector<SST_kv*> d_skv;

  SST_kv* L0_d_skv_sorted;

  SST_kv* L0_d_skv_sorted_2;
  SST_kv* L0_d_skv_sorted_shared;

  // sorted after unpacked
  SST_kv* L0_h_skv_sorted;

  SST_kv* h_skv_sorted;
  SST_kv* d_skv_sorted;
  SST_kv* d_skv_sorted_shared;
  
  // access SSTs from the GPU
  char** d_SST_ptr;

  std::vector<uint32_t*> h_shared_size;
  std::vector<uint32_t*> d_shared_size;

  std::vector<uint32_t*> h_shared_offset;
  std::vector<uint32_t*> d_shared_offset;

  std::vector<filter_meta*> h_fmeta;
  std::vector<filter_meta*> d_fmeta;

  WpSlice* lowSlices;
  int low_size;
  WpSlice* highSlices;
  int high_size;
  WpSlice* resultSlice;
  int result_size;
};

class SSTDecode {
 public:
  SSTDecode(const char* filename, int filesize, char* SST, std::string fname)
      : all_kv_(0), shared_cnt_(0), h_SST_(SST), file_size_(filesize) {
    // TODO: open file and read it to h_SST_
    inMem = false;
    FILE* file = ::fopen(filename, "rb");
    assert(file);
    size_t n = ::fread(h_SST_, sizeof(char), file_size_, file);
    assert(n == file_size_);
    ::fclose(file);
  }

  void SetMemory(int idx, HostAndDeviceMemory* m) {
    d_SST_ptr_ = m->d_SST_ptr;
    SST_idx_ = idx;

    d_SST_ = m->d_SST[idx];

    h_skv_ = m->h_skv[idx];
    d_skv_ = m->d_skv[idx];

    h_gdi_ = m->h_gdi[idx];
    d_gdi_ = m->d_gdi[idx];
  }

  ~SSTDecode() = default;
  void DoDecode();
  void DoGPUDecode();

  // Async Decode
  void DoGPUDecode_1(WpSlice* slices = nullptr, int index = 0);
  void DoGPUDecode_2(WpSlice* slices = nullptr, int index = 0);
  void Copy();

  int all_kv_;

  // private:

  bool FindInMem(std::string fname, HostAndDeviceMemory* m) {
    auto it = m->l0_hkv.find(fname);
    auto it2 = m->l0_knum.find(fname);
    if (it == m->l0_hkv.end()) {
      return false;
    }
    assert(it2 != m->l0_knum.end());
    SST_kv* skv = it->second;
    int num = it2->second;
    memcpy(h_skv_, skv, sizeof(SST_kv) * num);
    all_kv_ = num;
    inMem = true;
    m->pushL0skv(skv);
    m->l0_hkv.erase(it);
    m->l0_knum.erase(it2);
    return true;
  }

  bool inMem;

  char* h_SST_;
  char* d_SST_;

  int file_size_;
  GDI* h_gdi_;
  GDI* d_gdi_;

  SST_kv* h_skv_;
  SST_kv* d_skv_;

  char** d_SST_ptr_;
  int SST_idx_;
  int shared_cnt_;  // the count of all restarts

  Stream s_;
};

class SSTCompactionUtil {
 public:
  SSTCompactionUtil(leveldb::Version* input, int level)
      : input_version_(input), level_(level) {}
  ~SSTCompactionUtil() {}

  // safely remove keys marked delete
  bool IsBaseLevelForKey(const Slice& __user_key) {
    leveldb::Slice user_key(__user_key.data(), __user_key.size());
    const Comparator* user_cmp = BytewiseComparator();

    for (int lvl = level_ + 2; lvl < config::kNumLevels; lvl++) {
      const std::vector<FileMetaData*>& files = input_version_->files_[lvl];
      for (; level_ptrs_[lvl] < files.size();) {
        FileMetaData* f = files[level_ptrs_[lvl]];
        if (user_cmp->Compare(user_key, f->largest.user_key()) <= 0) {
          // We've advanced far enough
          if (user_cmp->Compare(user_key, f->smallest.user_key()) >= 0) {
            // Key falls in this file's range, so definitely not base level
            return false;
          }
          break;
        }
        level_ptrs_[lvl]++;
      }
    }
    return true;
  }

 private:
  leveldb::Version* input_version_;
  size_t level_ptrs_[config::kNumLevels];
  int level_;
};

class SSTSort {
 public:
  enum SSTSortType { enNULL = -1, enL0 = 5, enLow = 8, enHigh = 10 };
  SSTSort(uint64_t seq, SST_kv* out, SSTCompactionUtil* util,
          SST_kv* d_kv = nullptr)
      : seq_(seq),
        witch_(enNULL),
        l0_sst_index_(-1),
        util_(util),
        out_(out),
        out_size_(0),
        key_(),
        low_sst_index_(0),
        high_sst_index_(0),
        d_kvs_(d_kv) {}

  ~SSTSort() {}

  void AddL0(int size, SST_kv* skv) {
    l0_sizes_.push_back(size);
    l0_idx_.push_back(0);
    l0_skvs_.push_back(skv);
  }

  void AddLow(int size, SST_kv* skv) {
    low_sizes_.push_back(size);
    low_idx_.push_back(0);
    low_skvs_.push_back(skv);
  }

  void AddHigh(int size, SST_kv* skv) {
    high_sizes_.push_back(size);
    high_idx_.push_back(0);
    high_skvs_.push_back(skv);
  }

  void WpSort();
  void Sort();

 private:
  Slice FindLowSmallest();
  Slice FindHighSmallest();
  Slice FindL0Smallest();
  Slice GetCurrent(std::vector<SST_kv*>& skvs, std::vector<int>& idxs,
                   std::vector<int>& sizes, int& sst_idx);
  void Next(std::vector<SST_kv*>& skvs, std::vector<int>& idxs,
            std::vector<int>& sizes, int& sst_idx);

  Slice key_;          // current key_.
  SSTSortType witch_;  // level of this key comes from: -1:null, 0:l0, 1:low, 2:high
  uint64_t seq_;       // the currently min. seq.

  // xx_sizes_ : counts of unpacked kvs
  // xx_idx_   : counts of traversed kvs
  std::vector<int> l0_sizes_;
  std::vector<int> l0_idx_;
  std::vector<SST_kv*> l0_skvs_;

  std::vector<int> low_sizes_;
  std::vector<int> low_idx_;
  std::vector<SST_kv*> low_skvs_;

  std::vector<int> high_sizes_;
  std::vector<int> high_idx_;
  std::vector<SST_kv*> high_skvs_;

  int low_sst_index_;   // low level first SST
  int high_sst_index_;  // high level first SST
  int l0_sst_index_;    // Level 0 current minimum SST

  SSTCompactionUtil* util_;

 public:
  SST_kv* out_;   // outptu sorted kvs
  int out_size_;  // counts of kvs

  SST_kv* d_kvs_;
  WpSlice* low_slices = nullptr;
  WpSlice* high_slices = nullptr;
  WpSlice* result_slices = nullptr;
  int num;
  int low_num;
  int high_num;
  int low_index = 0;
  int high_index = 0;
  void AddLowSlice(int size, SST_kv* skv);
  void AddHighSlice(int size, SST_kv* skv);
  void AllocLow(int size, HostAndDeviceMemory* m);
  void AllocHigh(int size, HostAndDeviceMemory* m);
  void AllocResult(int size, HostAndDeviceMemory* m);
};

class SSTEncode {
 public:
  SSTEncode(char* SST, int kv_cnt, int SST_idx)
      : cur_(0), h_SST_(SST), SST_idx_(SST_idx), kv_count_(kv_cnt) {
    shared_count_ = (kv_cnt + kSharedKeys - 1) / kSharedKeys;
  }
  ~SSTEncode() {}

  void SetMemory(HostAndDeviceMemory* m, int base, bool isFlush = false) {
    base_ = base;
    if(SST_idx_>=CUDA_MAX_COMPACTION_FILES)
    {
      fprintf(stderr,"errrrrrrrrrrrrrrrrrrr\n");
    }

    if (!isFlush) {
      h_skv_ = m->h_skv_sorted + base;
      d_skv_ = m->d_skv_sorted;
      d_skv_new_ = m->d_skv_sorted_shared;

      d_SST_ptr = m->d_SST_ptr;
      d_SST_new_ = m->d_SST_new[SST_idx_];

      h_shared_size_ = m->h_shared_size[SST_idx_];
      d_shared_size_ = m->d_shared_size[SST_idx_];

      h_shared_offset_ = m->h_shared_offset[SST_idx_];
      d_shared_offset_ = m->d_shared_offset[SST_idx_];

      h_fmeta_ = m->h_fmeta[SST_idx_];
      d_fmeta_ = m->d_fmeta[SST_idx_];

      l0_d_skv_ = m->L0_d_skv_sorted;
    } else {
      h_skv_ = m->L0_h_skv_sorted + base;
      d_skv_ = m->L0_d_skv_sorted_2;
      d_skv_new_ = m->L0_d_skv_sorted_shared;

      d_SST_ptr = m->d_SST_ptr;
      d_SST_new_ = m->d_SST_new[SST_idx_];

      h_shared_size_ = m->h_shared_size[SST_idx_];
      d_shared_size_ = m->d_shared_size[SST_idx_];

      h_shared_offset_ = m->h_shared_offset[SST_idx_];
      d_shared_offset_ = m->d_shared_offset[SST_idx_];

      h_fmeta_ = m->h_fmeta[SST_idx_];
      d_fmeta_ = m->d_fmeta[SST_idx_];

      l0_d_skv_ = m->L0_d_skv_sorted;
    }
  }

  void ComputeDataBlockOffset(
      int sc = kDataSharedCnt);  // counts of shared areas in a DataBlock
  void ComputeFilter();  // calc. the filter offset and length for each DataBlock
  void WriteIndexAndFooter();  // footer

  void DoEncode();

  // AsyncEncode
  void DoEncode_1(bool f = false);
  void DoEncode_2(bool f = false);
  void DoEncode_3();
  void DoEncode_4();

  char* h_SST_;
  char* d_SST_new_;
  uint32_t cur_;  // write place

  char** d_SST_ptr;
  int SST_idx_;

  SST_kv* l0_d_skv_;

  SST_kv* h_skv_;
  SST_kv* d_skv_;  // Full Key-Value Not the shared-KV
  SST_kv* d_skv_new_;
  int kv_count_;
  int base_;

  int datablock_count_;  // counts of KV DataBlock

  /* for shared blocks */
  uint32_t* h_shared_size_;
  uint32_t* d_shared_size_;
  int shared_count_;

  uint32_t* h_shared_offset_;  // offset to start for each shared
  uint32_t* d_shared_offset_;

  filter_meta*
      h_fmeta_;  // meta for eache DataBlock, including counts of KV and starting from which KV
  filter_meta*
      d_fmeta_;  // meta for eache DataBlock, including counts of KV and starting from which KV

  std::vector<block_meta> bmeta_;  // metadata for each DataBlock, offset、size

  BlockHandle filter_handle_;
  int filter_end_;
  Footer footer;

  int data_blocks_size_;

  Stream s1_, s2_;
};

// CUDA function
void cudaMemHtD(void* dst, void* src, size_t size);
void cudaMemDtH(void* dst, void* src, size_t size);

class Debug {
 public:
  void Test(const char* src, size_t cnt);
};

}  // namespace gpu
}  // namespace leveldb
#endif  // LEVELDB_CUDA_DECODE_KV_H
