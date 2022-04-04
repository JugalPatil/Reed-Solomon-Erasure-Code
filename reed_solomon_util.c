#include "reed_solomon_util.h"
#include <pthread.h>

#define CPUID_EBX_AVX2    0x00000020
#define CPUID_ECX_SSSE3   0x00000200

static void _cpuid(unsigned int cpu_info[4U], const unsigned int cpu_info_type){
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;

    __asm__ __volatile__ ("xchgq %%rbx, %q1; cpuid; xchgq %%rbx, %q1" :
                          "=a" (cpu_info[0]), "=&r" (cpu_info[1]),
                          "=c" (cpu_info[2]), "=d" (cpu_info[3]) :
                          "0" (cpu_info_type), "2" (0U));

}

void InitializeCPUArch(){
    unsigned int cpu_info[4];
    _cpuid(cpu_info, 7);
}

//------------------------------------------------------------------------------
// XOR Memory

void xor_mem(void * __restrict vx, const void * __restrict vy,uint64_t bytes){
    __m256i * __restrict x32 = (__m256i *)(vx);
    const __m256i * __restrict y32 = (const __m256i *)(vy);
    while (bytes >= 128){
        const __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32),     _mm256_loadu_si256(y32));
        const __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
        const __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2), _mm256_loadu_si256(y32 + 2));
        const __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3), _mm256_loadu_si256(y32 + 3));
        _mm256_storeu_si256(x32, x0);
        _mm256_storeu_si256(x32 + 1, x1);
        _mm256_storeu_si256(x32 + 2, x2);
        _mm256_storeu_si256(x32 + 3, x3);
        x32 += 4, y32 += 4;
        bytes -= 128;
    };
    if (bytes > 0){
        const __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32),     _mm256_loadu_si256(y32));
        const __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
        _mm256_storeu_si256(x32, x0);
        _mm256_storeu_si256(x32 + 1, x1);
    }
    return;
}


void xor_mem_2to1(void * __restrict x,const void * __restrict y,const void * __restrict z,uint64_t bytes){

    __m256i * __restrict x32 = (__m256i *)(x);
    const __m256i * __restrict y32 = (const __m256i *)(y);
    const __m256i * __restrict z32 = (const __m256i *)(z);
    while (bytes >= 128){
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
        x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
        x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
        __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2), _mm256_loadu_si256(y32 + 2));
        x2 = _mm256_xor_si256(x2, _mm256_loadu_si256(z32 + 2));
        __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3), _mm256_loadu_si256(y32 + 3));
        x3 = _mm256_xor_si256(x3, _mm256_loadu_si256(z32 + 3));
        _mm256_storeu_si256(x32, x0);
        _mm256_storeu_si256(x32 + 1, x1);
        _mm256_storeu_si256(x32 + 2, x2);
        _mm256_storeu_si256(x32 + 3, x3);
        x32 += 4, y32 += 4, z32 += 4;
        bytes -= 128;
    };

    if (bytes > 0){
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32),     _mm256_loadu_si256(y32));
        x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
        x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
        _mm256_storeu_si256(x32, x0);
        _mm256_storeu_si256(x32 + 1, x1);
    }

    return;
}

void xor_mem4(void * __restrict vx_0, const void * __restrict vy_0,void * __restrict vx_1, const void * __restrict vy_1,void * __restrict vx_2, const void * __restrict vy_2,void * __restrict vx_3, const void * __restrict vy_3,uint64_t bytes){

    __m256i * __restrict       x32_0 = (__m256i *)      (vx_0);
    const __m256i * __restrict y32_0 = (const __m256i *)(vy_0);
    __m256i * __restrict       x32_1 = (__m256i *)      (vx_1);
    const __m256i * __restrict y32_1 = (const __m256i *)(vy_1);
    __m256i * __restrict       x32_2 = (__m256i *)      (vx_2);
    const __m256i * __restrict y32_2 = (const __m256i *)(vy_2);
    __m256i * __restrict       x32_3 = (__m256i *)      (vx_3);
    const __m256i * __restrict y32_3 = (const __m256i *)(vy_3);
    while (bytes >= 128){
        const __m256i x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0),     _mm256_loadu_si256(y32_0));
        const __m256i x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1), _mm256_loadu_si256(y32_0 + 1));
        const __m256i x2_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 2), _mm256_loadu_si256(y32_0 + 2));
        const __m256i x3_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 3), _mm256_loadu_si256(y32_0 + 3));
        _mm256_storeu_si256(x32_0, x0_0);
        _mm256_storeu_si256(x32_0 + 1, x1_0);
        _mm256_storeu_si256(x32_0 + 2, x2_0);
        _mm256_storeu_si256(x32_0 + 3, x3_0);
        x32_0 += 4, y32_0 += 4;
        const __m256i x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1),     _mm256_loadu_si256(y32_1));
        const __m256i x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1), _mm256_loadu_si256(y32_1 + 1));
        const __m256i x2_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 2), _mm256_loadu_si256(y32_1 + 2));
        const __m256i x3_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 3), _mm256_loadu_si256(y32_1 + 3));
        _mm256_storeu_si256(x32_1, x0_1);
        _mm256_storeu_si256(x32_1 + 1, x1_1);
        _mm256_storeu_si256(x32_1 + 2, x2_1);
        _mm256_storeu_si256(x32_1 + 3, x3_1);
        x32_1 += 4, y32_1 += 4;
        const __m256i x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2),     _mm256_loadu_si256(y32_2));
        const __m256i x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1), _mm256_loadu_si256(y32_2 + 1));
        const __m256i x2_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 2), _mm256_loadu_si256(y32_2 + 2));
        const __m256i x3_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 3), _mm256_loadu_si256(y32_2 + 3));
        _mm256_storeu_si256(x32_2, x0_2);
        _mm256_storeu_si256(x32_2 + 1, x1_2);
        _mm256_storeu_si256(x32_2 + 2, x2_2);
        _mm256_storeu_si256(x32_2 + 3, x3_2);
        x32_2 += 4, y32_2 += 4;
        const __m256i x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3),     _mm256_loadu_si256(y32_3));
        const __m256i x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1), _mm256_loadu_si256(y32_3 + 1));
        const __m256i x2_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 2), _mm256_loadu_si256(y32_3 + 2));
        const __m256i x3_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 3), _mm256_loadu_si256(y32_3 + 3));
        _mm256_storeu_si256(x32_3,     x0_3);
        _mm256_storeu_si256(x32_3 + 1, x1_3);
        _mm256_storeu_si256(x32_3 + 2, x2_3);
        _mm256_storeu_si256(x32_3 + 3, x3_3);
        x32_3 += 4, y32_3 += 4;
        bytes -= 128;
    }
    if (bytes > 0){
        const __m256i x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0),     _mm256_loadu_si256(y32_0));
        const __m256i x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1), _mm256_loadu_si256(y32_0 + 1));
        const __m256i x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1),     _mm256_loadu_si256(y32_1));
        const __m256i x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1), _mm256_loadu_si256(y32_1 + 1));
        _mm256_storeu_si256(x32_0, x0_0);
        _mm256_storeu_si256(x32_0 + 1, x1_0);
        _mm256_storeu_si256(x32_1, x0_1);
        _mm256_storeu_si256(x32_1 + 1, x1_1);
        const __m256i x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2),     _mm256_loadu_si256(y32_2));
        const __m256i x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1), _mm256_loadu_si256(y32_2 + 1));
        const __m256i x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3),     _mm256_loadu_si256(y32_3));
        const __m256i x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1), _mm256_loadu_si256(y32_3 + 1));
        _mm256_storeu_si256(x32_2,     x0_2);
        _mm256_storeu_si256(x32_2 + 1, x1_2);
        _mm256_storeu_si256(x32_3,     x0_3);
        _mm256_storeu_si256(x32_3 + 1, x1_3);
    }
    return;


}

void VectorXOR_Threads(const uint64_t bytes,unsigned count,void** x,void** y){
    if (count >= 4){
        int i_end = count - 4;
#pragma omp parallel for
        for (int i = 0; i <= i_end; i += 4){
            xor_mem4(x[i + 0], y[i + 0],x[i + 1], y[i + 1],x[i + 2], y[i + 2],x[i + 3], y[i + 3],bytes);
        }
        count %= 4;
        i_end -= count;
        x += i_end;
        y += i_end;
    }

    for (unsigned i = 0; i < count; ++i)
        xor_mem(x[i], y[i], bytes);
}

void VectorXOR(const uint64_t bytes,unsigned count,void** x,void** y){
    if (count >= 4){
        int i_end = count - 4;
        for (int i = 0; i <= i_end; i += 4){
            xor_mem4(x[i + 0], y[i + 0],x[i + 1], y[i + 1],x[i + 2], y[i + 2],x[i + 3], y[i + 3],bytes);
        }
        count %= 4;
        i_end -= count;
        x += i_end;
        y += i_end;
    }

    for (unsigned i = 0; i < count; ++i)
        xor_mem(x[i], y[i], bytes);
}


// Accumulate some source data
void Add(struct XORSummer *this,const void* src, const uint64_t bytes){
    if (this->Waiting){
        xor_mem_2to1(this->DestBuffer, src, this->Waiting, bytes);
        this->Waiting = NULL;
    }
    else
        this->Waiting = src;

}

// Finalize in the destination buffer
void Finalize(struct XORSummer *this,const uint64_t bytes){
    if (this->Waiting)
        xor_mem(this->DestBuffer, this->Waiting, bytes);
}

static struct XORSummer new(void* DestBuffer,const void* Waiting) {
    return (struct XORSummer){.DestBuffer=DestBuffer, .Waiting=Waiting, .Add=&Add, .Finalize=&Finalize};
}

const struct XORSummerClass XORSummer={.new=&new};
