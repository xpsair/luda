# LUDA 

**L**evelDB with C**UDA**

LUDA: A Customized GPU Acceleration Scheme to Boost LSM Key-Value Store Compactions


### requirement

- make sure you linux run LevelDB 1.22 (https://github.com/google/leveldb) properly

- CUDA 10.2

- moderngpu from https://github.com/moderngpu/moderngpu


### configuration

- assume your `nvcc` path is `/usr/local/cuda/bin/nvcc`

- assume the path of LUDA is `/home/abc/flying/LUDA`
  - modify the path to moderngpu head filens in `/home/abc/flying/LUDA/cuda/decode_kv.cu`
  - modify the path to a test directory `leveldb::DB::Open(options, "your_directory", &db);` in `/home/abc/flying/LUDA/db/paper_test.cc`, or compile will fail

```
|-- AUTHORS
|-- build/
|-- CMakeLists.txt
|-- CONTRIBUTING.md
|-- cuda/
|   |-- cuda_common.h               # luda data structures
|   |-- decode_kv.cu                # luda kernels
|   |-- ...
|-- db/
|   |-- db_impl.cc                  # luda take over compaction
|   |-- ...
|-- doc/
|-- helpers/
|-- include/
|-- issues/
|-- LICENSE
|-- NEWS
|-- port/
|-- README.md
|-- table/
|-- TODO
|-- util/
```


```
cd /home/abc/flying/LUDA
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

- modify the following files in the `CMakeFiles` to compile and link to CUDA kernels:

  - paper_test.dir/build.make

    - add `NVCC=/usr/local/cuda/bin/nvcc` to the beginning of this file

    - replace all `/usr/bin/c++` with `$(NVCC)`

    - add the following at the end of the `CMakeFiles/paper_test.dir/util/testharness.cc.o:` target: 

      `$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paper_test.dir/util/decode_kv.cu.o -c /home/abc/flying/LUDA/cuda/decode_kv.cu`


  - paper_test.dir/flags.make

    - get the gencode of your GPU and the following at the end of `CXX_FLAGS = ...` :

      `-std=c++11 -rdc=true -Xptxas -dlcm=ca --extended-lambda  -arch=sm_61 `


  - paper_test.dir/link.txt

    - replace all `/usr/bin/c++` with the following :

      `/usr/local/cuda/bin/nvcc -arch=sm_61 CMakeFiles/paper_test.dir/util/decode_kv.cu.o`


  - leveldb.dir/build.make

    - add `NVCC=/usr/local/cuda/bin/nvcc` to the beginning of this file

    - replace all `/usr/bin/c++` with `$(NVCC)`


  - db_bench.dir/link.txt

    - replace `/usr/bin/c++` with `/usr/local/cuda/bin/nvcc -arch=sm_61 CMakeFiles/paper_test.dir/util/decode_kv.cu.o`


### compile 

make sure the above configurations a applied.

```
cd /home/abc/flying/LUDA/build
cmake --build . --target clean && cmake --build . --target paper_test && cmake --build . --target db_bench
cp ../build._static.sh && ./build_static.sh   # build libleveldb.a on demand
```
