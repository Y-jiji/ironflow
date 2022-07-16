/// extended dot product
mod _extdot;
pub use _extdot::*;

/// EINstein SUMmation convention for tensor contraction
mod _einsum;
pub use _einsum::*;

/// permute the index
mod _permute;
pub use _permute::*;

/// reshape (nothing to do with real data)
mod _reshape;
pub use _reshape::*;

/// clip masked data to 0
mod _dropout;
pub use _dropout::*;

/// fancy indexing like ndarray in python
mod _index;
pub use _index::*;

/// matrix multiplication
mod _matmul;
pub use _matmul::*;