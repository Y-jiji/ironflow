// implement all basic arthimetical things

#define XID (blockDim.x * blockIdx.x + threadIdx.x)
#define YID (blockDim.y * blockIdx.y + threadIdx.y)
#define ZID (blockDim.z * blockIdx.z + threadIdx.z)
#define XSTEP (blockDim.x * gridDim.x)
#define YSTEP (blockDim.y * gridDim.y)
#define ZSTEP (blockDim.z * gridDim.z)
#define ISTART (XID + YID * XSTEP + ZID * YSTEP * XSTEP)
#define ISTEP  (XSTEP * YSTEP * ZSTEP)


// incase of add-assigning, all bounds should be checked 
#define IMPL_ARITH(NAME, DATATYPE, OPS)       \
extern "C"                                    \
__global__ void NAME(                         \
    const DATATYPE*   x,                      \
    const DATATYPE*   y,                      \
          DATATYPE*   z,                      \
          int       len,                      \
    void* const   lower,                      \
    void* const   upper                       \
) {                                           \
    int trunclen = 0;                         \
    if (x + len > upper)                      \
        trunclen = (DATATYPE *)upper - x;     \
    else if (y + len > upper)                 \
        trunclen = (DATATYPE *)upper - y;     \
    else if (z + len > upper)                 \
        trunclen = (DATATYPE *)upper - z;     \
    int i = ISTART;                           \
    for (; i < trunclen; i += ISTEP)          \
        z[i] = x[i] OPS y[i];                 \
    if (x + len > upper)                      \
        x = (DATATYPE*)lower;                 \
    if (y + len > upper)                      \
        y = (DATATYPE*)lower;                 \
    if (z + len > upper)                      \
        z = (DATATYPE*)lower;                 \
    for (; i < len; i += ISTEP)               \
        z[i] = x[i] OPS y[i];                 \
}

IMPL_ARITH(add_f32, float, +)
IMPL_ARITH(add_f64, double, +)
IMPL_ARITH(add_i32, int, +)
IMPL_ARITH(add_i64, long long, +)

IMPL_ARITH(sub_f32, float, -)
IMPL_ARITH(sub_f64, double, -)
IMPL_ARITH(sub_i32, int, -)
IMPL_ARITH(sub_i64, long long, -)

IMPL_ARITH(mul_f32, float, *)
IMPL_ARITH(mul_f64, double, *)
IMPL_ARITH(mul_i32, int, *)
IMPL_ARITH(mul_i64, long long, *)

IMPL_ARITH(div_f32, float, /)
IMPL_ARITH(div_f64, double, /)
IMPL_ARITH(div_i32, int, /)
IMPL_ARITH(div_i64, long long, /)

IMPL_ARITH(rem_i32, int, %)
IMPL_ARITH(rem_i64, long long, %)

#undef XID
#undef YID
#undef ZID
#undef XSTEP
#undef YSTEP
#undef ZSTEP
#undef ISTART
#undef ISTEP 

#undef IMPL

