__global__ void extdot_f32(
    const float*     x,
    const float*     y,
          float*     z,
          int     leni,
          int     lenj,
          int     lenk
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < leni; 
            i += blockDim.x * gridDim.x
    ) {
    for (
        int j = blockIdx.y * blockDim.y + threadIdx.y;
            j < lenj; 
            j += blockDim.y * gridDim.y
    ) {
        float delta_zij = 0;
    for (
        int k = 0; 
            k < lenk;
            k ++
    ) {
        delta_zij += x[i*lenk + k] * y[j*lenk + k];
    }
        z[i*lenj + j] += delta_zij;
    }}
}

__global__ void extdot_f64(
    const double*    x,
    const double*    y,
          double*    z,
          int     leni,
          int     lenj,
          int     lenk
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < leni; 
            i += blockDim.x * gridDim.x
    ) {
    for (
        int j = blockIdx.y * blockDim.y + threadIdx.y;
            j < lenj; 
            j += blockDim.y * gridDim.y
    ) {
        double delta_zij = 0;
    for (
        int k = 0; 
            k < lenk;
            k ++
    ) {
        delta_zij += x[i*lenk + k] * y[j*lenk + k];
    }
        z[i*lenj + j] += delta_zij;
    }}
}

__global__ void extdot_i32(
    const int*       x,
    const int*       y,
          int*       z,
          int     leni,
          int     lenj,
          int     lenk
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < leni; 
            i += blockDim.x * gridDim.x
    ) {
    for (
        int j = blockIdx.y * blockDim.y + threadIdx.y;
            j < lenj; 
            j += blockDim.y * gridDim.y
    ) {
        int delta_zij = 0;
    for (
        int k = 0; 
            k < lenk;
            k ++
    ) {
        delta_zij += x[i*lenk + k] * y[j*lenk + k];
    }
        z[i*lenj + j] += delta_zij;
    }}
}

__global__ void extdot_i64(
    const long*      x,
    const long*      y,
          long*      z,
          int     leni,
          int     lenj,
          int     lenk
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < leni; 
            i += blockDim.x * gridDim.x
    ) {
    for (
        int j = blockIdx.y * blockDim.y + threadIdx.y;
            j < lenj; 
            j += blockDim.y * gridDim.y
    ) {
        long delta_zij = 0;
    for (
        int k = 0; 
            k < lenk;
            k ++
    ) {
        delta_zij += x[i*lenk + k] * y[j*lenk + k];
    }
        z[i*lenj + j] += delta_zij;
    }}
}