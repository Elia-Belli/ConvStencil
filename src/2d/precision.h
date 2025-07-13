#pragma once

// Choose floating point precision
#define USE_DOUBLE_PRECISION
// #define USE_FLOAT_PRECISION

#ifdef USE_DOUBLE_PRECISION
  using real_t = double;
  #define REAL_FMT "%lf"
#elif USE_FLOAT_PRECISION
  using real_t = float;
  #define REAL_FMT "%f"
#else
    #error "Define either USE_FLOAT_PRECISION or USE_DOUBLE_PRECISION"
#endif

template <typename T>
const char* precision_name(T) noexcept
{
    if (std::is_same<T, float>::value)
        return "float";
    else if (std::is_same<T, double>::value)
        return "double";
    else
        return "unknown";
}
