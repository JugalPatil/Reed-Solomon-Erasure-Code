#ifndef RESULT_H
#define RESULT_H
#include <stdint.h>

extern int rs_init_();

typedef enum ErrorResultReason{
    success           =  0,           // Operation succeeded
    need_more_data      = -1,         // Not enough recovery data received
    too_much_data       = -2,         // Buffer counts are too high
    invalid_size       = -3,          // Buffer size must be a multiple of 64 bytes
    invalid_count     = -4,           // Invalid counts provided
    invalid_input      = -5,          // A function parameter was invalid
    invalid_platform          = -6,   // Platform is unsupported
    invalid_call_initialize    = -7,  // Call rs_init_() first
} ReedSolomonResult;

extern const char* result_string(ReedSolomonResult result);

extern unsigned rs_encode_work_count(unsigned original_count,unsigned recovery_count);

extern ReedSolomonResult rs_encode(
    uint64_t buffer_bytes,                    // Number of bytes in each data buffer
    unsigned original_count,                  // Number of original_data[] buffer pointers
    unsigned recovery_count,                  // Number of recovery_data[] buffer pointers
    unsigned work_count,                      // Number of work_data[] buffer pointers, from rs_encode_work_count()
    const void* const * const original_data,  // Array of pointers to original data buffers
    void** work_data);                        // Array of work buffers

extern unsigned rs_decode_work_count(unsigned original_count,unsigned recovery_count);

extern ReedSolomonResult rs_decode(
    uint64_t total_bytes,                     // size of file without padding
    uint64_t buffer_bytes,                    // Number of bytes in each data buffer
    unsigned original_count,                  // Number of original_data[] buffer pointers
    unsigned recovery_count,                  // Number of recovery_data[] buffer pointers
    unsigned work_count,                      // Number of buffer pointers in work_data[]
    const void* const * const original_data,  // Array of original data buffers
    const void* const * const recovery_data,  // Array of recovery data buffers
    void** work_data);                        // Array of work data buffers

#endif // RESULT_H
