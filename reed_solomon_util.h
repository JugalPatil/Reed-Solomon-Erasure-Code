#pragma once

#include "reed_solomon.h"
#define ALIGN_BYTES 32
#include <immintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <stdio.h>

#define ALIGNED __attribute__((aligned(ALIGN_BYTES)))


void InitializeCPUArch();

inline unsigned LastNonzeroBit32(unsigned x){
    return 31 - (unsigned)__builtin_clz(x);
}

inline unsigned NextPow2(unsigned n){
    return 2UL << LastNonzeroBit32(n - 1);
}

// x[] ^= y[]
void xor_mem(void * __restrict x, const void * __restrict y,uint64_t bytes);

// x[] ^= y[] ^ z[]
void xor_mem_2to1(void * __restrict x,const void * __restrict y,const void * __restrict z,uint64_t bytes);

// For i = {0, 1, 2, 3}: x_i[] ^= x_i[]
void xor_mem4(void * __restrict x_0, const void * __restrict y_0,void * __restrict x_1, const void * __restrict y_1,void * __restrict x_2, const void * __restrict y_2,void * __restrict x_3, const void * __restrict y_3,uint64_t bytes);

// x[] ^= y[]
void VectorXOR(const uint64_t bytes,unsigned count,void** x,void** y);

// x[] ^= y[] (Multithreaded)
void VectorXOR_Threads(const uint64_t bytes,unsigned count,void** x,void** y);


struct XORSummer{
    void* DestBuffer;
    const void* Waiting;
    void (*Add)(struct XORSummer *this,const void* src,const uint64_t bytes);
    void (*Finalize)(struct XORSummer *this,const uint64_t bytes);
};

extern const struct XORSummerClass {
    struct XORSummer(*new)(void*DestBuffer,const void* Waiting);
} XORSummer;

static const unsigned kAlignmentBytes = ALIGN_BYTES;

static inline uint8_t* SafeAllocate(size_t size){
    uint8_t* data = (uint8_t*)calloc(1, kAlignmentBytes + size);
    if (!data)
        return NULL;
    unsigned offset = (unsigned)((uintptr_t)data % kAlignmentBytes);
    data += kAlignmentBytes - offset;
    data[-1] = (uint8_t)offset;
    return data;
}

static inline void SafeFree(void* ptr){
    if (!ptr)
        return;
    uint8_t* data = (uint8_t*)ptr;
    unsigned offset = data[-1];
    if (offset >= kAlignmentBytes){
        return;
    }
    data -= kAlignmentBytes - offset;
    free(data);
}
