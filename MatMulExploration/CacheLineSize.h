#pragma once

// Adapted from Nick Strupat (original at https://github.com/NickStrupat/CacheLineSize)

#include <stddef.h>

// Returns the cache line size (in bytes) of the processor, or 0 on failure
size_t CacheLineSize();

// Returns the cache size (in bytes) of the processor, or 0 on failure
size_t CacheSize();
