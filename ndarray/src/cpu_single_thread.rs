use async_std::task::{spawn, JoinHandle};
use std::ops::*;
use std::sync::{Arc, Mutex};

pub trait Device {
    /* wrapped pointer to device memory */
    type DevPtr;
}

#[derive(Clone, Copy, Debug)]
pub struct CPUSingleThread;
#[derive(Clone, Copy, Debug)]
pub struct Nothing;

impl Device for CPUSingleThread {
    type DevPtr = Nothing;
}

trait DeepClone {
    type Output;
    fn deepclone(&self) -> Self::Output;
}

#[derive(Debug, Clone)]
pub struct NDArray<ValT, DevT: Device> {
    /* ValT: value type, DevT: device type */
    pub size: Arc<Mutex<Vec<usize>>>,
    pub data: Arc<Mutex<Vec<ValT>>>,
    pub device: Arc<Mutex<DevT>>,
    pub devptr: Arc<Mutex<DevT::DevPtr>>
}

/* Here we don't implement broadcast in NDArray */
impl Add for &NDArray<i32, CPUSingleThread> {
    type Output = JoinHandle<NDArray<i32, CPUSingleThread>>;
    fn add(self, rhs: Self) -> Self::Output {
        if let (Ok(lhslock), Ok(rhslock)) = (self.size.lock(), rhs.size.lock()) {
            assert!(*lhslock == *rhslock);
        }
        let lhs = self.clone();
        let rhs = rhs.clone();
        spawn(async move {
            let mut new_data = (*lhs.data.lock().unwrap()).clone();
            {let rhs_data = &*rhs.data.lock().unwrap();
            for i in 0..new_data.len() {
                new_data[i] += rhs_data[i];
            }} /* we don't need rhs_data any longer */
            NDArray::<i32, CPUSingleThread> {
                size: Arc::new(Mutex::new(lhs.size.lock().unwrap().clone())),
                data: Arc::new(Mutex::new(new_data)),
                device: Arc::new(Mutex::new(lhs.device.lock().unwrap().clone())),
                devptr: Arc::new(Mutex::new(lhs.devptr.lock().unwrap().clone())),
            }
        })
    }
}

/* Here we don't implement broadcast in NDArray */
impl Add for NDArray<i32, CPUSingleThread> {
    type Output = JoinHandle<NDArray<i32, CPUSingleThread>>;
    fn add(self, rhs: Self) -> Self::Output {
        if let (Ok(lhslock), Ok(rhslock)) = (self.size.lock(), rhs.size.lock()) {
            assert!(*lhslock == *rhslock);
        }
        let lhs = self.clone();
        spawn(async move {
            let mut new_data = (*lhs.data.lock().unwrap()).clone();
            {let rhs_data = &*rhs.data.lock().unwrap();
            for i in 0..new_data.len() {
                new_data[i] += rhs_data[i];
            }} /* we don't need rhs_data any longer */
            drop(self);
            drop(rhs);
            NDArray::<i32, CPUSingleThread> {
                size: Arc::new(Mutex::new(lhs.size.lock().unwrap().clone())),
                data: Arc::new(Mutex::new(new_data)),
                device: Arc::new(Mutex::new(lhs.device.lock().unwrap().clone())),
                devptr: Arc::new(Mutex::new(lhs.devptr.lock().unwrap().clone())),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CPUSingleThread, Nothing, NDArray};
    use async_std::task::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_refered_add() {
        let a = NDArray::<i32, CPUSingleThread> {
            size: Arc::new(Mutex::new(vec![4,5,6])),
            data: Arc::new(Mutex::new(vec![1; 4*5*6])),
            device: Arc::new(Mutex::new(CPUSingleThread)),
            devptr: Arc::new(Mutex::new(Nothing)),
        };
        let b = NDArray::<i32, CPUSingleThread> {
            size: Arc::new(Mutex::new(vec![4,5,6])),
            data: Arc::new(Mutex::new(vec![-1; 4*5*6])),
            device: Arc::new(Mutex::new(CPUSingleThread)),
            devptr: Arc::new(Mutex::new(Nothing)),
        };
        println!("{:#?}", block_on(&a + &b));
    }

    #[test]
    fn test_add() {
        let a = NDArray::<i32, CPUSingleThread> {
            size: Arc::new(Mutex::new(vec![4,5,6])),
            data: Arc::new(Mutex::new(vec![1; 4*5*6])),
            device: Arc::new(Mutex::new(CPUSingleThread)),
            devptr: Arc::new(Mutex::new(Nothing)),
        };
        let b = NDArray::<i32, CPUSingleThread> {
            size: Arc::new(Mutex::new(vec![4,5,6])),
            data: Arc::new(Mutex::new(vec![-1; 4*5*6])),
            device: Arc::new(Mutex::new(CPUSingleThread)),
            devptr: Arc::new(Mutex::new(Nothing)),
        };
        println!("{:#?}", block_on(a + b));
    }
}
