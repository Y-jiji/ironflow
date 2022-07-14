mod _proto;
pub mod cpu_unistream;
pub mod cuda_unistream;
pub use _proto::*;
pub mod async_rt;

#[cfg(test)]
mod test_cpu_unistream {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn zero_initialize(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(Mutex::new(cpu_unistream::CPUUniStream));
        let zero = NDArray::<i32, cpu_unistream::CPUUniStream>::zero(device, vec![10,11,12])?;
        println!("{zero:?}");
        Ok(())
    }

    #[test]
    fn zero_add_zero(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(Mutex::new(cpu_unistream::CPUUniStream));
        let zero_1 = NDArray::<i32, cpu_unistream::CPUUniStream>::zero(device.clone(), vec![10,11,12])?;
        let zero_2 = NDArray::<i32, cpu_unistream::CPUUniStream>::zero(device.clone(), vec![10,11,12])?;
        println!("{:?}", async_rt::task::block_on(zero_1 + zero_2)?);
        Ok(())
    }
}