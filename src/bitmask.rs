//! What the bitmask types share: their set operators, their numeric formats and the names in
//! their `Debug` output.

/// `|`, `&` and `-` (and their assigning forms) as `union`, `intersection` and `difference` of
/// the bitmask type `$mask`.
macro_rules! set_operators {
    ($mask:ty) => {
        impl core::ops::BitOr for $mask {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl core::ops::BitOrAssign for $mask {
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(rhs);
            }
        }

        impl core::ops::BitAnd for $mask {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self {
                self.intersection(rhs)
            }
        }

        impl core::ops::BitAndAssign for $mask {
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(rhs);
            }
        }

        impl core::ops::Sub for $mask {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                self.difference(rhs)
            }
        }

        impl core::ops::SubAssign for $mask {
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(rhs);
            }
        }
    };
}

pub(crate) use set_operators;

/// `Binary`, `Octal`, `LowerHex` and `UpperHex` of the bitmask type `$mask`, formatting its bits.
macro_rules! bit_formats {
    ($mask:ty) => {
        impl core::fmt::Binary for $mask {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Binary::fmt(&self.0, f)
            }
        }

        impl core::fmt::Octal for $mask {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Octal::fmt(&self.0, f)
            }
        }

        impl core::fmt::LowerHex for $mask {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::LowerHex::fmt(&self.0, f)
            }
        }

        impl core::fmt::UpperHex for $mask {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::UpperHex::fmt(&self.0, f)
            }
        }
    };
}

pub(crate) use bit_formats;

/// Writes the names of the set bits of `bits` as a set, bit `n` named `names[n]`, and any bits
/// past the names as one hex entry.
pub(crate) fn fmt_bit_names(
    f: &mut core::fmt::Formatter<'_>,
    bits: u32,
    names: &[&str],
) -> core::fmt::Result {
    let mut set = f.debug_set();
    for (bit, name) in names.iter().enumerate() {
        if bits & (1 << bit) != 0 {
            set.entry(&format_args!("{name}"));
        }
    }
    let defined = 1_u32
        .checked_shl(names.len() as u32)
        .map_or(u32::MAX, |past| past - 1);
    let undefined = bits & !defined;
    if undefined != 0 {
        set.entry(&format_args!("{undefined:#x}"));
    }
    set.finish()
}
