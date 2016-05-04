#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif

#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)

kernel void batchEP(global double* y, global double* x, global double* beta) {
  size_t i = get_global_id(0);
  beta[i] = y[i] + x[i];
}

#else

kernel void batchEP(global float* y, global float* x, global float* beta) {
  size_t i = get_global_id(0);
  beta[i] = y[i] + x[i];
}

#endif

