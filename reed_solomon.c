#include "reed_solomon.h"
#include "reed_solomon_util.h"
#include "field_arithmetic_16.h"
#include <string.h>
#include <stdbool.h>

static bool m_Initialized = false;

extern int rs_init_() {

    InitializeCPUArch();

    if (!Initialize())
        return invalid_platform;

    m_Initialized = true;
    return success;
}

extern const char* result_string(ReedSolomonResult result) {
    switch (result) {
    case success: return "Operation succeeded";
    case need_more_data: return "Not enough recovery data received";
    case too_much_data: return "Buffer counts are too high";
    case invalid_size: return "Buffer size must be a multiple of 64 bytes";
    case invalid_count: return "Invalid counts provided";
    case invalid_input: return "A function parameter was invalid";
    case invalid_platform: return "Platform is unsupported";
    case invalid_call_initialize: return "Call rs_init_() first";
    }
    return "Unknown";
}


extern unsigned rs_encode_work_count(unsigned original_count, unsigned recovery_count) {
    if (original_count == 1)
        return recovery_count;
    if (recovery_count == 1)
        return 1;
    return NextPow2(recovery_count) * 2;
}

static void EncodeM1(uint64_t buffer_bytes, unsigned original_count, const void* const * const original_data, void* recovery_data) {
    memcpy(recovery_data, original_data[0], buffer_bytes);

    struct XORSummer summer = XORSummer.new(&summer,recovery_data);
    // summer.XORSummerInitialize(&summer,recovery_data);

    for (unsigned i = 1; i < original_count; ++i)
        summer.Add(&summer,original_data[i], buffer_bytes);

    summer.Finalize(&summer,buffer_bytes);

}

extern ReedSolomonResult rs_encode(uint64_t buffer_bytes, unsigned original_count, unsigned recovery_count, unsigned work_count, const void* const * const original_data, void** work_data) {
    if (buffer_bytes <= 0 || buffer_bytes % 64 != 0)
        return invalid_size;

    if (recovery_count <= 0 || recovery_count > original_count)
        return invalid_count;

    if (!original_data || !work_data)
        return invalid_input;

    if (!m_Initialized)
        return invalid_call_initialize;

    if (original_count == 1) {
        for (unsigned i = 0; i < recovery_count; ++i)
            memcpy(work_data[i], original_data[i], buffer_bytes);
        return success;
    }

    if (recovery_count == 1) {
        EncodeM1(buffer_bytes, original_count, original_data, work_data[0]);
        return success;
    }

    const unsigned m = NextPow2(recovery_count);
    const unsigned n = NextPow2(m + original_count);

    if (work_count != m * 2)
        return invalid_count;

    if (n <= order)
        ReedSolomonEncode(buffer_bytes, original_count, recovery_count, m, original_data, work_data);
    else
        return too_much_data;

    return success;
}

extern unsigned rs_decode_work_count(unsigned original_count, unsigned recovery_count) {
    if (original_count == 1 || recovery_count == 1)
        return original_count;
    const unsigned m = NextPow2(recovery_count);
    const unsigned n = NextPow2(m + original_count);
    return n;
}

static void DecodeM1(uint64_t buffer_bytes, unsigned original_count, const void* const * original_data, const void* recovery_data, void* work_data) {
    memcpy(work_data, recovery_data, buffer_bytes);

    struct XORSummer summer = XORSummer.new(&summer,work_data);
    // summer.XORSummerInitialize(&summer,work_data);

    for (unsigned i = 0; i < original_count; ++i)
        if (original_data[i])
            summer.Add(&summer,original_data[i], buffer_bytes);

    summer.Finalize(&summer,buffer_bytes);
}

extern ReedSolomonResult rs_decode(uint64_t total_bytes,uint64_t buffer_bytes, unsigned original_count, unsigned recovery_count, unsigned work_count, const void* const * const original_data, const void* const * const recovery_data, void** work_data) {
    if (buffer_bytes <= 0 || buffer_bytes % 64 != 0)
        return invalid_size;

    if (recovery_count <= 0 || recovery_count > original_count)
        return invalid_count;

    if (!original_data || !recovery_data || !work_data)
        return invalid_input;

    if (!m_Initialized)
        return invalid_call_initialize;

    unsigned original_loss_count = 0;
    unsigned original_loss_i = 0;
    for (unsigned i = 0; i < original_count; ++i) {
        if (!original_data[i]) {
            ++original_loss_count;
            original_loss_i = i;
        }
    }
    unsigned recovery_got_count = 0;
    unsigned recovery_got_i = 0;
    for (unsigned i = 0; i < recovery_count; ++i) {
        if (recovery_data[i]) {
            ++recovery_got_count;
            recovery_got_i = i;
        }
    }
    if (recovery_got_count < original_loss_count)
        return need_more_data;

    if (original_count == 1) {
        memcpy(work_data[0], recovery_data[recovery_got_i], buffer_bytes);
        return success;
    }

    if (recovery_count == 1) {
        DecodeM1(buffer_bytes, original_count, original_data, recovery_data[0], work_data[original_loss_i]);
        return success;
    }

    const unsigned m = NextPow2(recovery_count);
    const unsigned n = NextPow2(m + original_count);

    if (work_count != n)
        return invalid_count;

    if (n <= order)
        ReedSolomonDecode(buffer_bytes, original_count, recovery_count, m, n, original_data, recovery_data, work_data);
    else
        return too_much_data;

    return success;
}


