use std::marker::PhantomData;
use crate::*;

struct NDArray<ValT, DevT>
where DevT: Device 
{
    /// marker of value type
    pub(crate)
    _val : PhantomData<ValT>,
    /// buffer might be actively modified by device
    pub(crate)
    buff : DevBuf<DevT>, 
    /// meta data about shape
    pub(crate)
    size : Vec<usize>,
}

