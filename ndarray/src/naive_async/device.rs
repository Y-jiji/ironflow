use crate::{Device, DevPtr, MemPtr};

#[derive(Debug, Clone)]
pub struct SingleThreadNaive;

/* Empty implementation that just returns nothing ...*/
#[allow(unused_variables)]
impl DevPtr for () {fn nullptr() -> Self { () }}
#[allow(unused_variables)]
impl MemPtr for () {fn from_vec<ValT>(v: &Vec<ValT>) -> Self { () }}

/* Empty implementation that panics everything ...*/
#[allow(unused_variables)]
impl Device for SingleThreadNaive {
    type DevPtr = ();           /* nothing */
    type MemPtr = ();           /* nothing */
    type MemErr = &'static str; /* memory error, never returned because we don't alloc memory */
    fn new(capacity: usize
    ) -> Result<Self, Self::MemErr> { Ok(SingleThreadNaive) }
    fn malloc(self,
        size_in_bytes: usize
    ) -> Result<Self::DevPtr, Self::MemErr> { panic!("not implemented") }
    fn memcpy_as_dst(self,
        src: Self::MemPtr, dst: Self::DevPtr, 
        size_in_bytes: usize
    ) -> Result<(), Self::MemErr> { panic!("not implemented") }
    fn memcpy_as_src(self,
        src: Self::DevPtr, dst: Self::MemPtr, 
        size_in_bytes: usize
    ) -> Result<(), Self::MemErr> { panic!("not implemented") }
    fn memcpy_inside(self,
        src: Self::DevPtr, dst: Self::DevPtr,
        size_in_bytes: usize,
    ) -> Result<(), Self::MemErr> { panic!("not implemented") }
}