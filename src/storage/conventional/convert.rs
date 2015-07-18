use std::convert::Into;

use Element;
use storage::Conventional;

impl<T: Element> Into<Vec<T>> for Conventional<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}
