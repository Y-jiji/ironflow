pub struct Symbol;

pub trait Einsum<T>
where Self : Sized,
         T : Sized + Into<Self> 
{
    type Output;
    fn einsum(
        operand : Vec<T>, 
        symbol  : Symbol
    ) -> Self::Output;
}