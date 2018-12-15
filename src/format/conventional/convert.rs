use std::convert::Into;

use format::Conventional;
use Element;

impl<T: Element> Into<Vec<T>> for Conventional<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}
