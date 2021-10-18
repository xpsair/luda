//
// Created by crabo on 2019/10/14.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <sys/types.h>
#include <fstream>

#include "leveldb/cache.h"
#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/filter_policy.h"
#include "leveldb/write_batch.h"
#include "leveldb/table.h"
#include "port/port.h"
#include "util/crc32c.h"
#include "util/histogram.h"
#include "util/mutexlock.h"
#include "util/random.h"
#include "util/testutil.h"
#include "util/coding.h"
#include "db/db_impl.h"

#include "cuda/cuda_common.h"
//#include "cuda/util.h"
//#include "cuda/decode_kv.h"

#include "sys/time.h"
#include "pthread.h"

using namespace std;
using namespace leveldb;

static uint64_t  get_now_micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec) * 1000000 + tv.tv_usec;
}

#define START(ts) (ts) = get_now_micros()
#define END(ts) (get_now_micros() - (ts))
#define COUNT (100 * 10000)

void create_db(bool write = false) {
    leveldb::DB *db;
    leveldb::Options options;

    options.create_if_missing = true;
    options.write_buffer_size = 15 * 1024 * 1024;
    options.max_file_size = 15 * 1024 * 1024;
    options.filter_policy = NewBloomFilterPolicy(10);
    leveldb::Status status = leveldb::DB::Open(options, "/mnt/OPTANE280G/wp/testdb4", &db);
    assert(status.ok());

    string value(1024 * 1, 'a');
    uint64_t ts, te;
    START(ts);
    if (write) {
        for (int i = 0; i < COUNT; ++i) {
            status = db->Put(WriteOptions(), to_string(i), value);
            //assert(status.ok());
            if (!status.ok())
                cout << status.ToString() << endl;
        }
    }
    te = END(ts);
    cout << "Write Time: " << te << endl;

    if (!write) {
        for (int i = 0; i < COUNT; ++i) {
            string res;
            status = db->Get(ReadOptions(), to_string(i), &res);
            assert(res == value);
        }
    }

    delete db;
}

void check_db() {
    leveldb::DB *db;
    leveldb::DBImpl *impl;
    leveldb::Options options;
    leveldb::Version* ver;

    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, "/mnt/OPTANE280G/wp/testdb4", &db);
    assert(status.ok());

    impl = (leveldb::DBImpl*) db;
    ver = impl->versions_->current_;

    for (int i = 0; i < leveldb::config::kNumLevels; ++i) {
        std::cout << "Level: " << i << std::endl;
        for (auto meta : ver->files_[i]) {
            std::string large(meta->largest.user_key().data(), meta->largest.user_key().size());
            std::string small(meta->smallest.user_key().data(), meta->smallest.user_key().size());
            std::cout << "[ " << meta->number
                    << " (" << small << ":" << large << ") ]" << std::endl;
        }
        std::cout << std::endl;
    }

    delete db;
}

void sort_testl0l1() {
    std::vector<std::string> v;
    std::vector<int> sizes;
    std::vector<bool> vf;
    std::vector<std::string> result;
    int SIZE = 100;
    int i, cnt;

    for (int i = 0; i < SIZE; ++i) {
        std::string str = std::to_string(i);
        v.push_back(str);
        vf.push_back(true);
    }
    std::sort(v.begin(), v.end());
    for (auto i : v)
        cout << i << " ";
    cout << endl;

    // level0 : 3x 7x
    // level1 : rest
    gpu::HostAndDeviceMemory memory;
    gpu::SST_kv* pskv = NULL;

    pskv = memory.h_skv[0];
    for (i = 0, cnt = 0; i < SIZE; i += 3) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    pskv = memory.h_skv[1];
    for (i = 0, cnt = 0; i < SIZE; i += 7) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    pskv = memory.h_skv[2];
    for (i = 0, cnt = 0; i < SIZE; i += 1) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    gpu::SSTSort sort(0, memory.h_skv[3], NULL);
    sort.AddL0(sizes[0], memory.h_skv[0]);
    sort.AddL0(sizes[1], memory.h_skv[1]);
    sort.AddHigh(sizes[2], memory.h_skv[2]);
    sort.Sort();

    gpu::SST_kv *p = sort.out_;
    for (i = 0; i < sort.out_size_; ++i) {
        string tmp(p[i].ikey, p[i].key_size);
        result.push_back(tmp);
        cout << tmp << " ";
    }

    return ;
}
void sort_testl1l2() {
    std::vector<std::string> v;
    std::vector<int> sizes;
    std::vector<bool> vf;
    std::vector<std::string> result;
    int SIZE = 100;
    int i, cnt;

    for (int i = 0; i < SIZE; ++i) {
        std::string str = std::to_string(i);
        v.push_back(str);
        vf.push_back(true);
    }
    std::sort(v.begin(), v.end());
    for (auto i : v)
        cout << i << " ";
    cout << endl;

    // level1 : 3x
    // level2 : 0-50 50-100
    gpu::HostAndDeviceMemory memory;
    gpu::SST_kv* pskv = NULL;

    pskv = memory.h_skv[0];
    for (i = 0, cnt = 0; i < SIZE; i += 3) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    pskv = memory.h_skv[1];
    for (i = 0, cnt = 0; i < SIZE/2; i += 1) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    pskv = memory.h_skv[2];
    for (i = SIZE/2, cnt = 0; i < SIZE; i += 1) {
        if (vf[i]) {
            vf[i] = false;
            cout << v[i] << " ";
            sprintf(pskv[cnt].ikey, "%d", stoi(v[i]));
            gpu::EncodeFixed64(pskv[cnt].ikey + v[i].size(), 1);
            pskv[cnt].key_size = v[i].size();
            ++ cnt;
        }
    }
    cout << endl;
    sizes.push_back(cnt);

    gpu::SSTSort sort(0, memory.h_skv[3], NULL);
    sort.AddLow(sizes[0], memory.h_skv[0]);
    sort.AddHigh(sizes[1], memory.h_skv[1]);
    sort.AddHigh(sizes[2], memory.h_skv[2]);
    sort.Sort();

    gpu::SST_kv *p = sort.out_;
    for (i = 0; i < sort.out_size_; ++i) {
        string tmp(p[i].ikey, p[i].key_size);
        result.push_back(tmp);
        cout << tmp << " ";
    }

    return ;
}
void test_En_DecodeValueOffset() {
    uint32_t off1 = 128;
    int idx = 5;
    int idxx = 0;

    gpu::EncodeValueOffset(&off1, idx);
    gpu::DecodeValueOffset(&off1, &idxx);

    for (int i = 0; i < 10; ++i ) {
        int a = i & 0x1 ^ 0x1;
        int b = i & 0x1;
        cout << a << " " << b << endl;
    }

    cout << idxx << " " << off1 << endl;
}

void *multi_do(void *args) {
    gpu::SSTEncode *pe = (gpu::SSTEncode *)args;
    pe->DoEncode();
    delete pe;
}

/*
 * 这里的测试暂时只做关于一个SST的完整测试，包括
 * 1. SST Decode
 * 2. SST Sort
 * 3. SST Encode & Write it to file [SSTs]
 * 最后检测最后生成的SSTs能够被leveldb原生的代码跑通
 */
void test_full_CPU_GPU(const char *f, uint64_t size) {
    const char *filename = f;
    int filesize = size;
    uint64_t ts, te;
    leveldb::gpu::HostAndDeviceMemory m;

    // 1. Decode
    START(ts);
    leveldb::gpu::SSTDecode SST(filename, filesize, m.h_SST[0],std::string(f));
    SST.SetMemory(0, &m);
    SST.DoDecode();
    SST.DoGPUDecode();
    te = END(ts);
    cout << SST.all_kv_ << " : "  << te << endl;

    gpu::SST_kv *pkv = m.h_skv[0];
    for (int i = 800; i < 900; ++i) {
        //cout << pkv[i].key_size << " " << pkv[i].value_size << endl;
    }

    // 2. Sort
    START(ts);
    gpu::SSTSort sort(0, m.h_skv_sorted, NULL);
    sort.AddLow(SST.all_kv_, m.h_skv[0]);
    sort.Sort();
    te = END(ts);
    cout << sort.out_size_ << " : " << te << endl;

    //cudaMemcpy(m.d_skv_sorted, sort.out_, sizeof(gpu::SST_kv) * sort.out_size_, cudaMemcpyHostToDevice);
    gpu::cudaMemHtD(m.d_skv_sorted, sort.out_, sizeof(gpu::SST_kv) * sort.out_size_);

    // 3. Encode
    // Divide the skv[] to server SST
    int last_keys = sort.out_size_;
    int keys_per_SST = gpu::kDataSharedCnt * gpu::kSharedKeys * gpu::kSharedPerSST;
    int SST_cnt = (sort.out_size_  + keys_per_SST - 1) / keys_per_SST;

    for (int i = 0; i < SST_cnt; ++i) {
        int kv_cnt = keys_per_SST <= last_keys ? keys_per_SST : last_keys;
        START(ts);
        gpu::SSTEncode encode(m.h_SST[0], kv_cnt, i);
        encode.SetMemory(&m, sort.out_size_ - last_keys);
        last_keys -= kv_cnt;
        encode.DoEncode();


        {
           FILE *file = ::fopen("sst1.ldb", "wb");
           ::fwrite(encode.h_SST_, 1, encode.cur_, file);
           ::fclose(file);
        }
        te = END(ts);
        cout << encode.cur_ << " : " << te << endl;
    }

    return ;

#define N 4

    START(ts);
    pthread_t tids[N];
    for (int i = 0; i < N; ++i) {
        //int kv_cnt = keys_per_SST <= last_keys ? keys_per_SST : last_keys;
        //int kv_cnt = last_keys;
        int kv_cnt = last_keys/N;
        //gpu::SSTEncode encode(m.h_SST[0], kv_cnt, i);
        gpu::SSTEncode *p = new gpu::SSTEncode(m.h_SST[i], kv_cnt, i);
        p->SetMemory(&m, kv_cnt * i);
        //p->SetMemory(&m, kv_cnt * i);
        //p->d_skv_ = m.d_skv_sorted + kv_cnt * i;
        //last_keys -= kv_cnt;
        //encode.DoEncode();
        pthread_create(&tids[i], NULL, multi_do, p);
        //p->DoEncode();
        //delete p;
        //break;
    }

    void *s;
    for (int i = 0; i < N; ++i) {
        pthread_join(tids[i], &s);
    }

    te = END(ts);
    cout <<" : " << te << endl;
}

/*
 * 使用LevelDB自己写的函数检测生成的SST是否正确
 */
void test_SST_correct(const char *f, uint64_t filesize) {
    leveldb::DB *db;
    leveldb::DBImpl *impl;
    leveldb::Options options;
    Env *env = NULL;
    uint64_t ts, te;
    //uint64_t filesize = 2108991 ;
    //uint64_t filesize = 2087122;
    std::ofstream ofile;
    string sf = f;
    sf += ".key";
    ofile.open(sf);

    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, "testdb_empty", &db);

    impl = (leveldb::DBImpl *)db;
    env = impl->env_;

    START(ts);
    leveldb::RandomAccessFile *file;
    //std::string filename = ".\\testdb\\000061.ldb";
    std::string filename(f);
    Status s = env->NewRandomAccessFile(filename, &file);
    cout << s.ToString() << endl;
    assert(s.ok());

    leveldb::Table *table = nullptr;
    leveldb::Table::Open(options, file, filesize, &table);
    leveldb::Iterator *it = table->NewIterator(ReadOptions());
    it->SeekToFirst();

    int cnt = 0;
    while (it->Valid()) {
        Slice Key = it->key();
        Slice V = it->value();
        string s(Key.data(), Key.size() - 8);
        //ofile << s << endl;
        it->Next();
        ++ cnt;
    }
    te = END(ts);

    cout << cnt << ": " << te << endl;
    ofile.close();
    delete db;
}

int main() {
    leveldb::DB *db;
    leveldb::Options options;

    bool create = false;

    create_db(true); return 0;
    const char *f = "./sst1.ldb";
    uint64_t size = 3895183;
    //size = 2100708;

    create = true;
    if (create) {
        f = "./000061.ldb";
        size = 2108991;

        f = "./000019.ldb";
        size = 3892367;
    }
    //create_db(true);
    test_full_CPU_GPU(f, size);
    //test_SST_correct(f, size);
    return 0;
}

