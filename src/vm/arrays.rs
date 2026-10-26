//! Render-controller arrays: the elements `array.<name>[index]` reads.

use std::collections::HashMap;

use nohash_hasher::BuildNoHashHasher;

use super::name::ArrayName;
use super::value::ResourceRef;

/// The render-controller arrays an evaluation reads, by name.
///
/// `array.<name>[index]` truncates the index toward zero with the program's
/// [`Platform`](crate::numeric::Platform), reads a negative index as 0, then wraps it by the
/// array's length ([`numeric::array_index`](crate::numeric::array_index)). The element is a
/// [`Value::Resource`](super::Value::Resource).
pub trait ArrayProvider {
    /// The elements of the array `name`; `None` for an unknown array, which reads as 0 with a
    /// [`RuntimeMsg::UnknownArray`](super::RuntimeMsg::UnknownArray).
    fn array(&self, name: ArrayName) -> Option<&[ResourceRef]>;
}

/// An [`ArrayProvider`] with no array at all.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NoArrays;

impl ArrayProvider for NoArrays {
    fn array(&self, _name: ArrayName) -> Option<&[ResourceRef]> {
        None
    }
}

/// A ready-made [`ArrayProvider`]: a render controller's `arrays` section, filled by the caller.
///
/// ```
/// use molang::vm::{ArrayName, ArrayProvider, NoHostEnv, ResourceRef};
///
/// let skins = ArrayName::new("skins");
/// let textures = [ResourceRef::new("texture.a"), ResourceRef::new("texture.b")];
/// let mut env = NoHostEnv::new();
/// env.arrays.set(skins, textures);
/// assert_eq!(env.arrays.array(skins), Some(&textures[..]));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayTable {
    arrays: HashMap<ArrayName, Box<[ResourceRef]>, BuildNoHashHasher<ArrayName>>,
}

impl ArrayTable {
    /// A table without arrays.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the elements of an array, returning the previous ones. An empty array reads as 0
    /// with a [`RuntimeMsg::EmptyArray`](super::RuntimeMsg::EmptyArray).
    pub fn set(
        &mut self,
        name: ArrayName,
        elements: impl IntoIterator<Item = ResourceRef>,
    ) -> Option<Box<[ResourceRef]>> {
        self.arrays.insert(name, elements.into_iter().collect())
    }

    /// The elements of an array.
    pub fn get(&self, name: ArrayName) -> Option<&[ResourceRef]> {
        self.arrays.get(&name).map(|elements| &**elements)
    }

    /// Removes one array.
    pub fn remove(&mut self, name: ArrayName) -> Option<Box<[ResourceRef]>> {
        self.arrays.remove(&name)
    }

    /// Removes every array.
    pub fn clear(&mut self) {
        self.arrays.clear();
    }

    /// Number of arrays.
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    /// Whether the table has no array.
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }
}

impl<E: IntoIterator<Item = ResourceRef>> FromIterator<(ArrayName, E)> for ArrayTable {
    /// A later array of a name replaces an earlier one.
    fn from_iter<I: IntoIterator<Item = (ArrayName, E)>>(arrays: I) -> Self {
        Self {
            arrays: arrays
                .into_iter()
                .map(|(name, elements)| (name, elements.into_iter().collect()))
                .collect(),
        }
    }
}

impl<E: IntoIterator<Item = ResourceRef>, const LEN: usize> From<[(ArrayName, E); LEN]>
    for ArrayTable
{
    fn from(arrays: [(ArrayName, E); LEN]) -> Self {
        arrays.into_iter().collect()
    }
}

impl ArrayProvider for ArrayTable {
    fn array(&self, name: ArrayName) -> Option<&[ResourceRef]> {
        self.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(name: &str) -> ResourceRef {
        ResourceRef::new(name)
    }

    #[test]
    fn a_table_keeps_the_elements_of_each_name() {
        let skins = ArrayName::new("skins");
        let mut table = ArrayTable::new();
        assert!(table.is_empty());
        assert_eq!(table.set(skins, [texture("texture.a")]), None);
        assert_eq!(
            table.set(skins, [texture("texture.b"), texture("texture.c")]),
            Some([texture("texture.a")].into())
        );
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.array(skins),
            Some(&[texture("texture.b"), texture("texture.c")][..])
        );
        assert_eq!(table.array(ArrayName::new("other")), None);
        assert_eq!(table.remove(skins).map(|e| e.len()), Some(2));
        assert!(table.is_empty());
        table.set(skins, []);
        assert_eq!(table.get(skins), Some(&[][..]));
        table.clear();
        assert_eq!(table.get(skins), None);
    }

    #[test]
    fn a_table_builds_from_pairs_the_later_winning() {
        let a = ArrayName::new("a");
        let table = ArrayTable::from([
            (a, vec![texture("texture.x")]),
            (ArrayName::new("b"), vec![]),
            (a, vec![texture("texture.y")]),
        ]);
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(a), Some(&[texture("texture.y")][..]));
        assert_eq!(table, table.clone());
    }

    #[test]
    fn no_arrays_has_none() {
        assert_eq!(NoArrays.array(ArrayName::new("a")), None);
    }
}
