#ifdef USE_DOUBLE_PRECISION
  using real_t = double;
  #define REAL_FMT "%lf"
#elif USE_FLOAT_PRECISION
  using real_t = float;
  #define REAL_FMT "%f"
#endif