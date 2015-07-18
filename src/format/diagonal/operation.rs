use Element;
use format::Diagonal;
use operation::Transpose;

impl<T: Element> Transpose for Diagonal<T> {
    #[inline(always)]
    fn transpose(&self) -> Self {
        self.clone()
    }
}
