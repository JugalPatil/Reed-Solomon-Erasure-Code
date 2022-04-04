#pragma once
#include "reed_solomon_util.h"
#include<stdbool.h>

// Number of bits per element
#define bits 16

// Finite field order: Number of elements in the field
#define order 65536

// Modulus for field operations
#define modulus 65535

// Finite field element type
typedef uint16_t ffe_t;

// LFSR Polynomial that generates the field elements
static const unsigned polynomial = 0x1002D;

// Returns false if the self-test fails
bool Initialize();

void ReedSolomonEncode(uint64_t buffer_bytes,unsigned original_count,unsigned recovery_count,unsigned m,const void* const * const data,void** work);

void ReedSolomonDecode(uint64_t buffer_bytes,unsigned original_count,unsigned recovery_count,unsigned m,unsigned n,const void* const * const original,const void* const * const recovery,void** work);
