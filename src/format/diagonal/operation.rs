use format::Diagonal;
use operation::Transpose;
use Element;

impl<T: Element> Transpose for Diagonal<T> {
    #[inline(always)]
    fn transpose(&self) -> Self {
        self.clone()
    }
}
