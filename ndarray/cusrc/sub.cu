extern "C"
__global__ void sub_f32(
    const float* x,
    const float* y,
          float* z,
          int    len
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x; 
            i < len;
            i += gridDim.x * blockDim.x
    ) {
        z[i] = x[i] - y[i];
    }
}

extern "C"
__global__ void sub_f64(
    const double* x,
    const double* y,
          double* z,
          int    len
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x; 
            i < len;
            i += gridDim.x * blockDim.x
    ) {
        z[i] = x[i] - y[i];
    }
}

extern "C"
__global__ void sub_i32(
    const int* x,
    const int* y,
          int* z,
          int  len
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x; 
            i < len;
            i += gridDim.x * blockDim.x
    ) {
        z[i] = x[i] - y[i];
    }
}

extern "C"
__global__ void sub_i64(
    const long* x,
    const long* y,
          long* z,
          int   len
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x; 
            i < len;
            i += gridDim.x * blockDim.x
    ) {
        z[i] = x[i] - y[i];
    }
}