pub trait Extdot<T>
where Self : Sized,
         T : Sized + Into<Self> 
{
    type Output;
    fn extdot(
        lhs: T,
        rhs: T,
        extdim : usize,
        dotdim : usize,
    ) -> Self::Output;
}