#include "field_arithmetic_16.h"
#include <string.h>
#include <stdbool.h>

static const ffe_t kCantorBasis[bits] = {
    0x0001, 0xACCA, 0x3C0E, 0x163E,
    0xC582, 0xED2E, 0x914C, 0x4012,
    0x6C98, 0x10D8, 0x6A72, 0xB900,
    0xFDB8, 0xFB34, 0xFF38, 0x991E
};

// z = x + y (mod modulus)
static inline ffe_t AddMod(const ffe_t a, const ffe_t b){
    const unsigned sum = (unsigned)a + b;
    return (ffe_t)(sum + (sum >> bits));
}

// z = x - y (mod modulus)
static inline ffe_t SubMod(const ffe_t a, const ffe_t b){
    const unsigned dif = (unsigned)a - b;
    return (ffe_t)(dif + (dif >> bits));
}

// {a, b} = {a + b, a - b} (Mod Q)
static inline void FWHT_2(ffe_t* restrict a, ffe_t* restrict b){
    const ffe_t sum = AddMod(*a, *b);
    const ffe_t dif = SubMod(*a, *b);
    *a = sum;
    *b = dif;
}

static inline void FWHT_4(ffe_t* data, unsigned s){
    const unsigned s2 = s << 1;

    ffe_t t0 = data[0];
    ffe_t t1 = data[s];
    ffe_t t2 = data[s2];
    ffe_t t3 = data[s2 + s];

    FWHT_2(&t0, &t1);
    FWHT_2(&t2, &t3);
    FWHT_2(&t0, &t2);
    FWHT_2(&t1, &t3);

    data[0] = t0;
    data[s] = t1;
    data[s2] = t2;
    data[s2 + s] = t3;
}

static void FWHT(ffe_t* data, const unsigned m, const unsigned m_truncated){
    unsigned dist = 1, dist4 = 4;
    for (; dist4 <= m; dist = dist4, dist4 <<= 2){
#pragma omp parallel for
        for (int r = 0; r < (int)m_truncated; r += dist4){
            const int i_end = r + dist;
            for (int i = r; i < i_end; ++i)
                FWHT_4(data + i, dist);
        }
    }

    if (dist < m)
#pragma omp parallel for
        for (int i = 0; i < (int)dist; ++i)
            FWHT_2(&data[i], &data[i + dist]);
}


//------------------------------------------------------------------------------
// Logarithm Tables

static ffe_t LogLUT[order];
static ffe_t ExpLUT[order];


// Returns a * Log(b)
static ffe_t MultiplyLog(ffe_t a, ffe_t log_b){
    if (a == 0)
        return 0;
    return ExpLUT[AddMod(LogLUT[a], log_b)];
}

// Initialize LogLUT[], ExpLUT[]
static void InitializeLogarithmTables(){

    unsigned state = 1;
    for (unsigned i = 0; i < modulus; ++i){
        ExpLUT[state] = (ffe_t)(i);
        state <<= 1;
        if (state >= order)
            state ^= polynomial;
    }
    ExpLUT[0] = modulus;

    LogLUT[0] = 0;
    for (unsigned i = 0; i < bits; ++i){
        const ffe_t basis = kCantorBasis[i];
        const unsigned width = (unsigned)(1UL << i);

        for (unsigned j = 0; j < width; ++j)
            LogLUT[j + width] = LogLUT[j] ^ basis;
    }

    for (unsigned i = 0; i < order; ++i)
        LogLUT[i] = ExpLUT[LogLUT[i]];

    for (unsigned i = 0; i < order; ++i)
        ExpLUT[LogLUT[i]] = i;

    ExpLUT[modulus] = ExpLUT[0];
}

struct Multiply256LUT_t{
    __m256i Lo[4];
    __m256i Hi[4];
};

static const struct Multiply256LUT_t* Multiply256LUT = NULL;

#define MUL_TABLES_256(table, log_m) \
        const __m256i T0_lo_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Lo[0]); \
        const __m256i T1_lo_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Lo[1]); \
        const __m256i T2_lo_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Lo[2]); \
        const __m256i T3_lo_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Lo[3]); \
        const __m256i T0_hi_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Hi[0]); \
        const __m256i T1_hi_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Hi[1]); \
        const __m256i T2_hi_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Hi[2]); \
        const __m256i T3_hi_##table = _mm256_loadu_si256(&Multiply256LUT[log_m].Hi[3]);

#define MUL_256(value_lo, value_hi, table) { \
            __m256i data_1 = _mm256_srli_epi64(value_lo, 4); \
            __m256i data_0 = _mm256_and_si256(value_lo, clr_mask); \
            data_1 = _mm256_and_si256(data_1, clr_mask); \
            prod_lo = _mm256_shuffle_epi8(T0_lo_##table, data_0); \
            prod_hi = _mm256_shuffle_epi8(T0_hi_##table, data_0); \
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(T1_lo_##table, data_1)); \
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(T1_hi_##table, data_1)); \
            data_0 = _mm256_and_si256(value_hi, clr_mask); \
            data_1 = _mm256_srli_epi64(value_hi, 4); \
            data_1 = _mm256_and_si256(data_1, clr_mask); \
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(T2_lo_##table, data_0)); \
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(T2_hi_##table, data_0)); \
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(T3_lo_##table, data_1)); \
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(T3_hi_##table, data_1)); }

#define MULADD_256(x_lo, x_hi, y_lo, y_hi, table) { \
            __m256i prod_lo, prod_hi; \
            MUL_256(y_lo, y_hi, table); \
            x_lo = _mm256_xor_si256(x_lo, prod_lo); \
            x_hi = _mm256_xor_si256(x_hi, prod_hi); }

struct Product16Table{
    ffe_t LUT[4 * 16];
};
static const struct Product16Table* Multiply16LUT = NULL;

static void InitializeMultiplyTables(){

    Multiply256LUT = (const struct Multiply256LUT_t*)(SafeAllocate(sizeof(struct Multiply256LUT_t) * order));

#pragma omp parallel for
    for (int log_m = 0; log_m < (int)order; ++log_m){
        for (unsigned i = 0, shift = 0; i < 4; ++i, shift += 4){
            uint8_t prod_lo[16], prod_hi[16];
            for (ffe_t x = 0; x < 16; ++x){
                const ffe_t prod = MultiplyLog(x << shift, (ffe_t)(log_m));
                prod_lo[x] = (uint8_t)(prod);
                prod_hi[x] = (uint8_t)(prod >> 8);
            }

            const __m128i value_lo = _mm_loadu_si128((__m128i*)prod_lo);
            const __m128i value_hi = _mm_loadu_si128((__m128i*)prod_hi);

            _mm256_storeu_si256((__m256i*)&Multiply256LUT[log_m].Lo[i],
                _mm256_broadcastsi128_si256(value_lo));
            _mm256_storeu_si256((__m256i*)&Multiply256LUT[log_m].Hi[i],
                _mm256_broadcastsi128_si256(value_hi));
        }
    }
}


static void mul_mem(void * restrict x, const void * restrict y,ffe_t log_m, uint64_t bytes){
    MUL_TABLES_256(0, log_m);

    const __m256i clr_mask = _mm256_set1_epi8(0x0f);

    __m256i * restrict x32 = (__m256i *)(x);
    const __m256i * restrict y32 = (const __m256i *)(y);

    do{
#define MUL_256_LS(x_ptr, y_ptr) { \
        const __m256i data_lo = _mm256_loadu_si256(y_ptr); \
        const __m256i data_hi = _mm256_loadu_si256(y_ptr + 1); \
        __m256i prod_lo, prod_hi; \
        MUL_256(data_lo, data_hi, 0); \
        _mm256_storeu_si256(x_ptr, prod_lo); \
        _mm256_storeu_si256(x_ptr + 1, prod_hi); }

        MUL_256_LS(x32, y32);
        y32 += 2, x32 += 2;

        bytes -= 64;
    } while (bytes > 0);

    return;
}

// Twisted factors used in FFT
static ffe_t FFTSkew[modulus];

// Factors used in the evaluation of the error locator polynomial
static ffe_t LogWalsh[order];


static void FFTInitialize(){
    ffe_t temp[bits - 1];

    // Generate FFT skew vector {1}:

    for (unsigned i = 1; i < bits; ++i)
        temp[i - 1] = (ffe_t)(1UL << i);

    for (unsigned m = 0; m < (bits - 1); ++m){
        const unsigned step = 1UL << (m + 1);

        FFTSkew[(1UL << m) - 1] = 0;

        for (unsigned i = m; i < (bits - 1); ++i){
            const unsigned s = (1UL << (i + 1));
            for (unsigned j = (1UL << m) - 1; j < s; j += step)
                FFTSkew[j + s] = FFTSkew[j] ^ temp[i];
        }

        temp[m] = modulus - LogLUT[MultiplyLog(temp[m], LogLUT[temp[m] ^ 1])];

        for (unsigned i = m + 1; i < (bits - 1); ++i){
            const ffe_t sum = AddMod(LogLUT[temp[i] ^ 1], temp[m]);
            temp[i] = MultiplyLog(temp[i], sum);
        }
    }

    for (unsigned i = 0; i < modulus; ++i)
        FFTSkew[i] = LogLUT[FFTSkew[i]];

    for (unsigned i = 0; i < order; ++i)
        LogWalsh[i] = LogLUT[i];
    LogWalsh[0] = 0;

    FWHT(LogWalsh, order, order);
}

// 2-way butterfly
static void IFFT_DIT2(void * restrict x, void * restrict y,ffe_t log_m, uint64_t bytes){

    MUL_TABLES_256(0, log_m);

    const __m256i clr_mask = _mm256_set1_epi8(0x0f);

    __m256i * restrict x32 = (__m256i *)(x);
    __m256i * restrict y32 = (__m256i *)(y);

    do{
#define IFFTB_256(x_ptr, y_ptr) { \
        __m256i x_lo = _mm256_loadu_si256(x_ptr); \
        __m256i x_hi = _mm256_loadu_si256(x_ptr + 1); \
        __m256i y_lo = _mm256_loadu_si256(y_ptr); \
        __m256i y_hi = _mm256_loadu_si256(y_ptr + 1); \
        y_lo = _mm256_xor_si256(y_lo, x_lo); \
        y_hi = _mm256_xor_si256(y_hi, x_hi); \
        _mm256_storeu_si256(y_ptr, y_lo); \
        _mm256_storeu_si256(y_ptr + 1, y_hi); \
        MULADD_256(x_lo, x_hi, y_lo, y_hi, 0); \
        _mm256_storeu_si256(x_ptr, x_lo); \
        _mm256_storeu_si256(x_ptr + 1, x_hi); }

        IFFTB_256(x32, y32);
        y32 += 2, x32 += 2;

        bytes -= 64;
    } while (bytes > 0);

    return;
}

// 4-way butterfly
static void IFFT_DIT4(uint64_t bytes,void** work,unsigned dist,const ffe_t log_m01,const ffe_t log_m23,const ffe_t log_m02){

    MUL_TABLES_256(01, log_m01);
    MUL_TABLES_256(23, log_m23);
    MUL_TABLES_256(02, log_m02);

    const __m256i clr_mask = _mm256_set1_epi8(0x0f);

    __m256i * restrict work0 = (__m256i *)(work[0]);
    __m256i * restrict work1 = (__m256i *)(work[dist]);
    __m256i * restrict work2 = (__m256i *)(work[dist * 2]);
    __m256i * restrict work3 = (__m256i *)(work[dist * 3]);

    do{
        __m256i work_reg_lo_0 = _mm256_loadu_si256(work0);
        __m256i work_reg_hi_0 = _mm256_loadu_si256(work0 + 1);
        __m256i work_reg_lo_1 = _mm256_loadu_si256(work1);
        __m256i work_reg_hi_1 = _mm256_loadu_si256(work1 + 1);

        // First layer:
        work_reg_lo_1 = _mm256_xor_si256(work_reg_lo_0, work_reg_lo_1);
        work_reg_hi_1 = _mm256_xor_si256(work_reg_hi_0, work_reg_hi_1);
        if (log_m01 != modulus)
            MULADD_256(work_reg_lo_0, work_reg_hi_0, work_reg_lo_1, work_reg_hi_1, 01);

        __m256i work_reg_lo_2 = _mm256_loadu_si256(work2);
        __m256i work_reg_hi_2 = _mm256_loadu_si256(work2 + 1);
        __m256i work_reg_lo_3 = _mm256_loadu_si256(work3);
        __m256i work_reg_hi_3 = _mm256_loadu_si256(work3 + 1);

        work_reg_lo_3 = _mm256_xor_si256(work_reg_lo_2, work_reg_lo_3);
        work_reg_hi_3 = _mm256_xor_si256(work_reg_hi_2, work_reg_hi_3);
        if (log_m23 != modulus)
            MULADD_256(work_reg_lo_2, work_reg_hi_2, work_reg_lo_3, work_reg_hi_3, 23);

        // Second layer:
        work_reg_lo_2 = _mm256_xor_si256(work_reg_lo_0, work_reg_lo_2);
        work_reg_hi_2 = _mm256_xor_si256(work_reg_hi_0, work_reg_hi_2);
        work_reg_lo_3 = _mm256_xor_si256(work_reg_lo_1, work_reg_lo_3);
        work_reg_hi_3 = _mm256_xor_si256(work_reg_hi_1, work_reg_hi_3);
        if (log_m02 != modulus){
            MULADD_256(work_reg_lo_0, work_reg_hi_0, work_reg_lo_2, work_reg_hi_2, 02);
            MULADD_256(work_reg_lo_1, work_reg_hi_1, work_reg_lo_3, work_reg_hi_3, 02);
        }

        _mm256_storeu_si256(work0, work_reg_lo_0);
        _mm256_storeu_si256(work0 + 1, work_reg_hi_0);
        _mm256_storeu_si256(work1, work_reg_lo_1);
        _mm256_storeu_si256(work1 + 1, work_reg_hi_1);
        _mm256_storeu_si256(work2, work_reg_lo_2);
        _mm256_storeu_si256(work2 + 1, work_reg_hi_2);
        _mm256_storeu_si256(work3, work_reg_lo_3);
        _mm256_storeu_si256(work3 + 1, work_reg_hi_3);

        work0 += 2, work1 += 2, work2 += 2, work3 += 2;

        bytes -= 64;
    } while (bytes > 0);

    return;
}


// Unrolled IFFT for encoder
static void IFFT_DIT_Encoder(const uint64_t bytes,const void* const* data,const unsigned m_truncated,void** work,void** xor_result,const unsigned m,const ffe_t* skewLUT){
#pragma omp parallel for
    for (int i = 0; i < (int)m_truncated; ++i)
        memcpy(work[i], data[i], bytes);
#pragma omp parallel for
    for (int i = m_truncated; i < (int)m; ++i)
        memset(work[i], 0, bytes);
    unsigned dist = 1, dist4 = 4;
    for (; dist4 <= m; dist = dist4, dist4 <<= 2){
#pragma omp parallel for
        for (int r = 0; r < (int)m_truncated; r += dist4){
            const unsigned i_end = r + dist;
            const ffe_t log_m01 = skewLUT[i_end];
            const ffe_t log_m02 = skewLUT[i_end + dist];
            const ffe_t log_m23 = skewLUT[i_end + dist * 2];

            for (int i = r; i < (int)i_end; ++i){
                IFFT_DIT4(bytes,work + i,dist,log_m01,log_m23,log_m02);
            }
        }
    }

    if (dist < m){
        const ffe_t log_m = skewLUT[dist];

        if (log_m == modulus)
            VectorXOR_Threads(bytes, dist, work + dist, work);
        else{
#pragma omp parallel for
            for (int i = 0; i < (int)dist; ++i){
                IFFT_DIT2(work[i],work[i + dist],log_m,bytes);
            }
        }
    }
    if (xor_result)
        VectorXOR_Threads(bytes, m, xor_result, work);
}

static void IFFT_DIT_Decoder(const uint64_t bytes,const unsigned m_truncated,void** work,const unsigned m,const ffe_t* skewLUT){
    unsigned dist = 1, dist4 = 4;
    for (; dist4 <= m; dist = dist4, dist4 <<= 2){
#pragma omp parallel for
        for (int r = 0; r < (int)m_truncated; r += dist4){
            const unsigned i_end = r + dist;
            const ffe_t log_m01 = skewLUT[i_end];
            const ffe_t log_m02 = skewLUT[i_end + dist];
            const ffe_t log_m23 = skewLUT[i_end + dist * 2];

            for (int i = r; i < (int)i_end; ++i){
                IFFT_DIT4(bytes,work + i,dist,log_m01,log_m23,log_m02);
            }
        }
    }

    if (dist < m){
        const ffe_t log_m = skewLUT[dist];

        if (log_m == modulus)
            VectorXOR_Threads(bytes, dist, work + dist, work);
        else{
#pragma omp parallel for
            for (int i = 0; i < (int)dist; ++i){
                IFFT_DIT2(work[i],work[i + dist],log_m,bytes);
            }
        }
    }
}

static void FFT_DIT2(void * restrict x, void * restrict y,ffe_t log_m, uint64_t bytes){
        MUL_TABLES_256(0, log_m);
        const __m256i clr_mask = _mm256_set1_epi8(0x0f);
        __m256i * restrict x32 = (__m256i *)(x);
        __m256i * restrict y32 = (__m256i *)(y);

        do{
#define FFTB_256(x_ptr, y_ptr) { \
            __m256i x_lo = _mm256_loadu_si256(x_ptr); \
            __m256i x_hi = _mm256_loadu_si256(x_ptr + 1); \
            __m256i y_lo = _mm256_loadu_si256(y_ptr); \
            __m256i y_hi = _mm256_loadu_si256(y_ptr + 1); \
            MULADD_256(x_lo, x_hi, y_lo, y_hi, 0); \
            _mm256_storeu_si256(x_ptr, x_lo); \
            _mm256_storeu_si256(x_ptr + 1, x_hi); \
            y_lo = _mm256_xor_si256(y_lo, x_lo); \
            y_hi = _mm256_xor_si256(y_hi, x_hi); \
            _mm256_storeu_si256(y_ptr, y_lo); \
            _mm256_storeu_si256(y_ptr + 1, y_hi); }

            FFTB_256(x32, y32);
            y32 += 2, x32 += 2;

            bytes -= 64;
        } while (bytes > 0);

        return;
}


// 4-way butterfly
static void FFT_DIT4(uint64_t bytes,void** work,unsigned dist,const ffe_t log_m01,const ffe_t log_m23,const ffe_t log_m02){

        MUL_TABLES_256(01, log_m01);
        MUL_TABLES_256(23, log_m23);
        MUL_TABLES_256(02, log_m02);

        const __m256i clr_mask = _mm256_set1_epi8(0x0f);

        __m256i * restrict work0 = (__m256i *)(work[0]);
        __m256i * restrict work1 = (__m256i *)(work[dist]);
        __m256i * restrict work2 = (__m256i *)(work[dist * 2]);
        __m256i * restrict work3 = (__m256i *)(work[dist * 3]);

        do{
            __m256i work_reg_lo_0 = _mm256_loadu_si256(work0);
            __m256i work_reg_hi_0 = _mm256_loadu_si256(work0 + 1);
            __m256i work_reg_lo_1 = _mm256_loadu_si256(work1);
            __m256i work_reg_hi_1 = _mm256_loadu_si256(work1 + 1);
            __m256i work_reg_lo_2 = _mm256_loadu_si256(work2);
            __m256i work_reg_hi_2 = _mm256_loadu_si256(work2 + 1);
            __m256i work_reg_lo_3 = _mm256_loadu_si256(work3);
            __m256i work_reg_hi_3 = _mm256_loadu_si256(work3 + 1);

            // First layer:
            if (log_m02 != modulus){
                MULADD_256(work_reg_lo_0, work_reg_hi_0, work_reg_lo_2, work_reg_hi_2, 02);
                MULADD_256(work_reg_lo_1, work_reg_hi_1, work_reg_lo_3, work_reg_hi_3, 02);
            }
            work_reg_lo_2 = _mm256_xor_si256(work_reg_lo_0, work_reg_lo_2);
            work_reg_hi_2 = _mm256_xor_si256(work_reg_hi_0, work_reg_hi_2);
            work_reg_lo_3 = _mm256_xor_si256(work_reg_lo_1, work_reg_lo_3);
            work_reg_hi_3 = _mm256_xor_si256(work_reg_hi_1, work_reg_hi_3);

            // Second layer:
            if (log_m01 != modulus)
                MULADD_256(work_reg_lo_0, work_reg_hi_0, work_reg_lo_1, work_reg_hi_1, 01);
            work_reg_lo_1 = _mm256_xor_si256(work_reg_lo_0, work_reg_lo_1);
            work_reg_hi_1 = _mm256_xor_si256(work_reg_hi_0, work_reg_hi_1);

            _mm256_storeu_si256(work0, work_reg_lo_0);
            _mm256_storeu_si256(work0 + 1, work_reg_hi_0);
            _mm256_storeu_si256(work1, work_reg_lo_1);
            _mm256_storeu_si256(work1 + 1, work_reg_hi_1);

            if (log_m23 != modulus)
                MULADD_256(work_reg_lo_2, work_reg_hi_2, work_reg_lo_3, work_reg_hi_3, 23);
            work_reg_lo_3 = _mm256_xor_si256(work_reg_lo_2, work_reg_lo_3);
            work_reg_hi_3 = _mm256_xor_si256(work_reg_hi_2, work_reg_hi_3);

            _mm256_storeu_si256(work2, work_reg_lo_2);
            _mm256_storeu_si256(work2 + 1, work_reg_hi_2);
            _mm256_storeu_si256(work3, work_reg_lo_3);
            _mm256_storeu_si256(work3 + 1, work_reg_hi_3);

            work0 += 2, work1 += 2, work2 += 2, work3 += 2;

            bytes -= 64;
        } while (bytes > 0);

        return;
}

static void FFT_DIT(const uint64_t bytes,void** work,const unsigned m_truncated,const unsigned m,const ffe_t* skewLUT){
    unsigned dist4 = m, dist = m >> 2;
    for (; dist != 0; dist4 = dist, dist >>= 2){
#pragma omp parallel for
        for (int r = 0; r < (int)m_truncated; r += dist4){
            const unsigned i_end = r + dist;
            const ffe_t log_m01 = skewLUT[i_end];
            const ffe_t log_m02 = skewLUT[i_end + dist];
            const ffe_t log_m23 = skewLUT[i_end + dist * 2];

            for (int i = r; i < (int)i_end; ++i){
                FFT_DIT4(bytes,work + i,dist,log_m01,log_m23,log_m02);
            }
        }
    }

    if (dist4 == 2){
#pragma omp parallel for
        for (int r = 0; r < (int)m_truncated; r += 2){
            const ffe_t log_m = skewLUT[r + 1];

            if (log_m == modulus)
                xor_mem(work[r + 1], work[r], bytes);
            else{
                FFT_DIT2(work[r],work[r + 1],log_m,bytes);
            }
        }
    }
}


//------------------------------------------------------------------------------
// Reed-Solomon Encode

void ReedSolomonEncode(uint64_t buffer_bytes,unsigned original_count,unsigned recovery_count,unsigned m,const void* const * data,void** work){
    const ffe_t* skewLUT = FFTSkew + m - 1;

    IFFT_DIT_Encoder(buffer_bytes,data,original_count < m ? original_count : m,work,NULL,m,skewLUT);

    const unsigned last_count = original_count % m;
    if (m >= original_count)
        goto skip_body;

    for (unsigned i = m; i + m <= original_count; i += m){
        data += m;
        skewLUT += m;

        IFFT_DIT_Encoder(buffer_bytes,data,m,work + m,work,m,skewLUT);
    }

    if (last_count != 0){
        data += m;
        skewLUT += m;
        IFFT_DIT_Encoder(buffer_bytes,data,last_count,work + m,work,m,skewLUT);
    }

skip_body:

    FFT_DIT(buffer_bytes,work,recovery_count,m,FFTSkew - 1);
}

//------------------------------------------------------------------------------
// Reed-Solomon Decode

void ReedSolomonDecode(uint64_t buffer_bytes,unsigned original_count,unsigned recovery_count,unsigned m,unsigned n,const void* const * const original,const void* const * const recovery,void** work){

    ffe_t error_locations[order] = {};
    for (unsigned i = 0; i < recovery_count; ++i)
        if (!recovery[i])
            error_locations[i] = 1;
    for (unsigned i = recovery_count; i < m; ++i)
        error_locations[i] = 1;
    for (unsigned i = 0; i < original_count; ++i){
        if (!original[i]){
            error_locations[i + m] = 1;
        }
    }

    FWHT(error_locations, order, m + original_count);

#pragma omp parallel for
    for (int i = 0; i < (int)order; ++i)
        error_locations[i] = ((unsigned)error_locations[i] * (unsigned)LogWalsh[i]) % modulus;

    FWHT(error_locations, order, order);

#pragma omp parallel for
    for (int i = 0; i < (int)recovery_count; ++i){
        if (recovery[i])
            mul_mem(work[i], recovery[i], error_locations[i], buffer_bytes);
        else
            memset(work[i], 0, buffer_bytes);
    }
#pragma omp parallel for
    for (int i = recovery_count; i < (int)m; ++i)
        memset(work[i], 0, buffer_bytes);


#pragma omp parallel for
    for (int i = 0; i < (int)original_count; ++i){
        if (original[i])
            mul_mem(work[m + i], original[i], error_locations[m + i], buffer_bytes);
        else
            memset(work[m + i], 0, buffer_bytes);
    }
#pragma omp parallel for
    for (int i = m + original_count; i < (int)n; ++i)
        memset(work[i], 0, buffer_bytes);


    IFFT_DIT_Decoder(buffer_bytes,m + original_count,work,n,FFTSkew - 1);

    for (unsigned i = 1; i < n; ++i){
        const unsigned width = ((i ^ (i - 1)) + 1) >> 1;

        if (width < 8){
            VectorXOR(buffer_bytes,width,work + i - width,work + i);
        }else{
            VectorXOR_Threads(buffer_bytes,width,work + i - width,work + i);
        }
    }

    const unsigned output_count = m + original_count;

    FFT_DIT(buffer_bytes, work, output_count, n, FFTSkew - 1);

    for (unsigned i = 0; i < original_count; ++i)
        if (!original[i])
            mul_mem(work[i], work[i + m], modulus - error_locations[i + m], buffer_bytes);
}

static bool IsInitialized = false;

bool Initialize(){
    if (IsInitialized)
        return true;

    InitializeLogarithmTables();
    InitializeMultiplyTables();
    FFTInitialize();

    IsInitialized = true;
    return true;
}


