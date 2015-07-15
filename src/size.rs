/// A size.
pub trait Size {
    /// Return the number of rows.
    fn rows(&self) -> usize;

    /// Return the number of columns.
    fn columns(&self) -> usize;

    /// Return the number of rows and columns.
    #[inline(always)]
    fn dimensions(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }
}

impl Size for (usize, usize) {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.0
    }

    #[inline(always)]
    fn columns(&self) -> usize {
        self.1
    }
}

impl Size for usize {
    #[inline(always)]
    fn rows(&self) -> usize {
        *self
    }

    #[inline(always)]
    fn columns(&self) -> usize {
        *self
    }
}
