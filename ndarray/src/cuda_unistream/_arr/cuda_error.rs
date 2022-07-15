//  definition of CudaError lives in cuda_binding
use super::cuda_binding::*; 

use std::fmt::{Display, Debug};
use std::error::Error;

impl Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for CudaError {}

pub(crate) type CudaResult<T> = Result<T, CudaError>;

impl CudaError {
    pub(crate) fn wrap_val<T>(self, val: T) -> CudaResult<T> {
        match self {
            Self::CUDA_SUCCESS => Ok(val),
            _else => Err(_else),
        }
    }
}
