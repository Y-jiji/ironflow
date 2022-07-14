mod _add;
pub use _add::*;

use super::_arr::*;
use crate::*;

#[macro_use]
pub(crate) mod helpers {
    macro_rules! aspawn {
        ($x: expr) => {
            crate::async_rt::task::spawn(async move {
                $x
            })
        };
    }
    pub(crate) use aspawn;
}