#define ISTEP (blockDim.x * gridDim.x)
#define JSTEP (blockDim.y * gridDim.y * blockDim.z * gridDim.z)
#define KSTEP (1)

#define ISTART (blockDim.x * blockIdx.x + threadIdx.x)
#define JSTART ((blockDim.y * blockIdx.y + threadIdx.y) * (blockDim.z * gridDim.z) + (blockDim.z * blockIdx.z + threadIdx.z))
#define KSTART (0)

#define IMPL_EXTDOT(NAME, DATATYPE)                  \
extern "C"                                           \
__global__ void NAME(                                \
    const DATATYPE*  x,                              \
    const DATATYPE*  y,                              \
          DATATYPE*  z,                              \
          int     leni,                              \
          int     lenj,                              \
          int     lenk,                              \
    void* const  lower,                              \
    void* const  upper                               \
) {                                                  \
	DATATYPE delta = 0;                              \
	int i = ISTART, j = JSTART, k = KSTART;          \
	if      (&x[leni * lenk] >= upper) goto XCROSS;  \
	else if (&y[lenj * lenk] >= upper) goto YCROSS;  \
	else if (&z[leni * lenj] >= upper) goto ZCROSS;  \
	else                               goto FINAL;   \
XCROSS:                                              \
	for (; i < leni; i += ISTEP)                     \
	for (; j < lenj; j += JSTEP) {                   \
		for (; k < lenk; ++k) {                      \
			if (&x[i*lenk + k + 1] > upper) {        \
                x = (DATATYPE*) lower;               \
                goto XFINAL;                         \
            }                                        \
			delta += x[i*lenk + k] * y[j*lenk + k];  \
		}                                            \
		z[i*lenj + j] = delta;                       \
	}                                                \
YCROSS:                                              \
	for (; i < leni; i += ISTEP)                     \
	for (; j < lenj; j += JSTEP) {                   \
		for (; k < lenk; ++k) {                      \
			if (&y[i*lenk + k + 1] > upper) {        \
                y = (DATATYPE*) lower;               \
                goto YFINAL;                         \
            }                                        \
			delta += x[i*lenk + k] * y[j*lenk + k];  \
		}                                            \
		z[i*lenj + j] = delta;                       \
	}                                                \
ZCROSS:                                              \
	for (; i < leni; i += ISTEP)                     \
	for (; j < lenj; j += JSTEP) {                   \
		if (&z[i*lenj + j + 1] > upper) {            \
		    z = (DATATYPE*) lower;                   \
            goto ZFINAL;                             \
        }                                            \
		for (; k < lenk; ++k)                        \
			delta += x[i*lenk + k] * y[j*lenk + k];  \
		z[i*lenj + j] = delta;                       \
	}                                                \
FINAL:                                               \
	for (; i < leni; i += ISTEP)                     \
	for (; j < lenj; j += JSTEP) {                   \
		ZFINAL:                                      \
		for (; k < lenk; ++k) {                      \
			XFINAL: YFINAL:                          \
			delta += x[i*lenk + k] * y[j*lenk + k];  \
		}                                            \
		z[i*lenj + j] = delta;                       \
	}                                                \
}

IMPL_EXTDOT(extdot_f32, float);
IMPL_EXTDOT(extdot_f64, double);
IMPL_EXTDOT(extdot_i32, int);
IMPL_EXTDOT(extdot_i64, long long);

#undef ISTEP
#undef JSTEP
#undef KSTEP

#undef ISTART
#undef JSTART
#undef KSTART

#undef IMPL_EXTDOT(NAME, DATATYPE)

