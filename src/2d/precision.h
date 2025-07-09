
#ifdef USE_DOUBLE_PRECISION
  #define REAL double
  #define REAL_FMT "%lf"
#elif USE_FLOAT_PRECISION
  #define REAL float
  #define REAL_FMT "%f"
#endif