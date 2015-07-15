/// A position.
pub trait Position {
    /// Return the row.
    fn row(&self) -> usize;

    /// Return the column.
    fn column(&self) -> usize;

    /// Return the row and column.
    #[inline(always)]
    fn coordinates(&self) -> (usize, usize) {
        (self.row(), self.column())
    }
}

impl Position for (usize, usize) {
    #[inline(always)]
    fn row(&self) -> usize {
        self.0
    }

    #[inline(always)]
    fn column(&self) -> usize {
        self.1
    }
}

impl Position for usize {
    #[inline(always)]
    fn row(&self) -> usize {
        *self
    }

    #[inline(always)]
    fn column(&self) -> usize {
        *self
    }
}
