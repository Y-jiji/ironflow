use crate::*;

#[derive(Debug, Clone)]
pub struct CPUUniStream;

type Ptr = *mut std::ffi::c_void;

impl Device 
for CPUUniStream {
    type Err = String;
    fn new_err(
        msg: &str
    ) -> Result<(), Self::Err> {
        Err(String::from(msg))    
    }
    fn memcpy(
        &self,
        dst: Ptr, _dstnum: usize,
        src: Ptr, _srcnum: usize,
        len: usize,
    ) -> Result<(), Self::Err> {
        /* fine if there is no overlap [dst ...] | [src ...] */
        /* bad situations looks like this:  */
        /* [dst ... dst+len]                */
        /*      [src ... src+len]           */
        unsafe { libc::memcpy(dst, src, len) };
        Ok(())
    }
    fn memset(
        &self,
        dst: Ptr, _dstnum: usize,
        len: usize, _0: bool,
    ) -> Result<(), Self::Err> {
        unsafe { libc::memset(dst, match _0 { true => 0, false => -1 }, len) };
        Ok(())
    }
    fn del_buff(
        &mut self,
        lower: Ptr,
        _upper: Ptr,
        _devnum: usize,
    ) -> Result<(), Self::Err> {
        unsafe { libc::free(lower) };
        Ok(())
    }
    fn new_buff(
        &mut self,
        len: usize,
    ) -> Result<(Ptr, Ptr, usize), Self::Err> {
        let lower = unsafe { libc::malloc(len) };
        if lower.is_null() { Err(String::from("Memory allocation failed"))? }
        let upper = unsafe { lower.offset(len as isize) };
        let devnum = 0;
        Ok((lower, upper, devnum))
    }
}

