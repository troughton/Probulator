#pragma once

#include <stdint.h>
#include <assert.h>

namespace Probulator
{
	typedef uint8_t u8;
	typedef uint16_t u16;
	typedef uint32_t u32;
	typedef uint64_t u64;

	typedef int8_t s8;
	typedef int16_t s16;
	typedef int32_t s32;
	typedef int64_t s64;
}

const float ggxAlpha = 0.01f;
const bool specularFixedNormal = true;
