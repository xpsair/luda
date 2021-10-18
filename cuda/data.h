//
// Created by crabo on 2019/11/1.
//

#ifndef LEVELDB_CUDA_DATA_H
#define LEVELDB_CUDA_DATA_H

#include "cuda/cuda_common.h"
#include "cuda/util.h"

namespace leveldb {
namespace gpu {
// std::string can not use in GPU-Device, so we use class Buffer
__host__ __device__
Buffer::Buffer(char *buf, size_t size) : base_(buf), total_(size), size_(0) {}

__host__ __device__
char *Buffer::now() {return base_ + size_;}
__host__ __device__
char *Buffer::data() { return base_; }
__host__ __device__
void Buffer::reset() { size_ = 0;}

    
   
__host__ __device__
inline void Buffer::advance(int n) {
    assert(size_ + n <= total_);
    size_ += n;
}


__host__ __device__
void Buffer::append(const char *data, size_t size) {
    //assert(size_ + size <= total_);
    Memcpy(base_ + size_, data, size);
    advance(size);
}

int Slice::internal_compare(const gpu::Slice &b) const {
    gpu::Slice user_key_a(data_, size_ - 8);
    gpu::Slice user_key_b(b.data(), b.size() - 8);

    int r = user_key_a.compare(user_key_b);
    if (r == 0) {
        const uint64_t anum = gpu::DecodeFixed64((const char *)user_key_a.data() + user_key_a.size());
        const uint64_t bnum = gpu::DecodeFixed64((const char *)user_key_b.data() + user_key_b.size());
        if (anum > bnum) {
            r = -1;
        } else if (anum < bnum) {
            r = +1;
        }
    }
    return r;
}

/*
bool operator==(const gpu::Slice &x, const gpu::Slice &y) {
    return ((x.size() == y.size()) &&
            (memcmp(x.data(), y.data(), x.size()) == 0));
}

bool operator!=(const gpu::Slice &x, const gpu::Slice &y) { return !(x == y); }
*/

}
}
#endif //LEVELDB_DATA_H
