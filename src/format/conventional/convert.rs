use std::convert::Into;

use Element;
use format::Conventional;

impl<T: Element> Into<Vec<T>> for Conventional<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}
