//
// Created by crabo on 2019/11/1.
//

#ifndef LEVELDB_CUDA_UTIL_H
#define LEVELDB_CUDA_UTIL_H

#include <stdint.h>
namespace leveldb {
namespace gpu {

class Buffer;
class Slice;

void Memcpy(char *dst, const char *src, size_t n);

void EncodeFixed32(char *dst, uint32_t value);

void EncodeFixed64(char *dst, uint64_t value);

void PutFixed32(Buffer *dst, uint32_t value);

void PutFixed64(Buffer *dst, uint64_t value);

char *EncodeVarint32(char *dst, uint32_t v);

void PutVarint32(Buffer *dst, uint32_t v);

char *EncodeVarint64(char *dst, uint64_t v);

void PutVarint64(Buffer *dst, uint64_t v);


uint32_t DecodeFixed32(const char *ptr);

uint64_t DecodeFixed64(const char *ptr);


const char *GetVarint32PtrFallback(const char *p, const char *limit, uint32_t *value);


inline const char *GetVarint32Ptr(const char *p, const char *limit, uint32_t *value);

bool GetVarint32(Slice *input, uint32_t *value);


const char *GetVarint64Ptr(const char *p, const char *limit, uint64_t *value);

bool GetVarint64(Slice *input, uint64_t *value);

void EncodeValueOffset(uint32_t *offset, int Idx);
void DecodeValueOffset(uint32_t *offset, int *Idx);


const char* DecodeEntry(const char* p, const char* limit,
                    uint32_t* shared, uint32_t* non_shared,
                    uint32_t* value_length);

uint32_t Hash(const char* data, size_t n, uint32_t seed = 0xbc9f1d34);

// GPU CRC32C
namespace gpu_crc32c {
    uint32_t Extend(uint32_t init_crc, const char *data, size_t n);

    inline uint32_t Mask(uint32_t crc);

    // Return the crc whose masked representation is masked_crc.
    inline uint32_t Unmask(uint32_t masked_crc);
}  // namespace crc32

}  // namespace gpu
}  // namespace leveldb

#endif //LEVELDB_UTIL_H
