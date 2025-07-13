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
constexpr const char* precision_name(T) noexcept
{
    if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, double>)
        return "double";
    else
        return "unknown";
}