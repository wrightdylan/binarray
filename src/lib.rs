//! # BinArray
//! Uses standard unsigned integers as statically sized binary arrays. This will
//! have improved performance over the BitVecs dynamically sized binary vector.
//! This is done by keeping the array out of heap allocation, improving locality,
//! and uses lower cost functions. Having a static array also permits other
//! optimisations. To further enhance performance, the functions do not provide
//! error checking due to potential branch prediction misses. It is the respon-
//! sibility of the programmer to ensure the index is within bounds, for example.
//! 
//! ## Bit Manipulation Functions
//! - get_bit() Returns the bit value at location index
//! - set_bit() Sets the bit value at location index
//! - to_bstring() converts the array into a padded binary string
//! 
//! ## Functions not included as they are already standard
//! - count_ones()
//! - count_zeros()
//! - from_be_bytes()
//! - from_le_bytes()
//! - leading_ones()
//! - leading_zeros()
//! - reverse_bits()
//! - rotate_left()
//! - rotate_right()
//! - to_be_bytes()
//! - to_le_bytes()

macro_rules! binary_array {
    ($($t:ident),*) => {
        pub trait BinaryArray {
            /// # === Single Bit Operations ===
            /// Functions for modifying or checking individual bits by index.
            
            /// Clears a bit at the index location.
            fn clear_bit(&mut self, index: usize) -> &mut Self;

            /// Retrieves the bit value from index location.
            fn get_bit(&self, index: usize) -> bool;

            /// Sets the bit at the index location.
            fn set_bit(&mut self, index: usize) -> &mut Self;

            /// Toggles the bit at the given index.
            fn toggle_bit(&mut self, index: usize) -> &mut Self;

            /// Writes the bit value at index location.
            fn write_bit(&mut self, index: usize, value: bool) -> &mut Self;

            /// # === Bitfield Operations ===
            /// Functions for working with contiguous blocks of bits.
            
            /// Apply a mask of x number of bits starting from the LSB in-place.
            fn apply_mask_lsb(&mut self, bits: usize) -> Self;

            /// Apply a mask of x number of bits starting from the MSB in-place.
            fn apply_mask_msb(&mut self, bits: usize) -> Self;

            /// Apply a mask of x number of bits starting from the LSB.
            /// A mask larger than the bit size is equivalent to the bit size.
            fn create_mask_lsb(&self, bits: usize) -> Self;

            /// Apply a mask of x number of bits starting from the MSB.
            /// A mask larger than the bit size is equivalent to the bit size.
            fn create_mask_msb(&self, bits: usize) -> Self;

            /// Zeros out a specific range.
            fn clear_bits(&mut self, start: usize, len: usize) -> &mut Self;

            /// Extracts a value from a specific range.
            fn get_bits(&self, start: usize, len: usize) -> Self;

            /// Sets all bits in a specific range.
            fn set_bits(&mut self, start: usize, len: usize) -> &mut Self;

            /// # === Bitwise Analysis ===
            /// Functions that inspect properties of the value.
            
            /// Returns true if exactly one bit is set.
            fn is_power_of_two(&self) -> bool;

            /// Returns 1 if an odd number of bits are set, or 0 if even.
            fn parity(&self) -> bool;

            /// Converts the binary array into a vector of indices where the position.
            /// is a '1'
            fn to_indices(&self) -> Vec<usize>;
            
            /// === Transformation Operations ===
            /// Functions that transform a binary array.
            
            /// Returns a value where only the lowest set bit remains.
            fn isolate_lsb(&self) -> Self;

            /// Clears the rightmost 1 bit.
            fn remove_lsb(&mut self) -> &mut Self;

            /// Swaps the values at two specific indices.
            fn swap_bits(&mut self, i: usize, j: usize) -> &mut Self;
            
            /// # === Conversion/Formatting Operations ===
            /// Functions that output the binary array in an alternative (e.g.
            /// human-readable) format.
            
            /// Formats the binary array as a padded string.
            fn to_bstring(&self) -> String;
        }

        $(
            impl BinaryArray for $t {
                // === Single Bit Operations ===
                fn clear_bit(&mut self, index: usize) -> &mut Self {
                    *self &= !((1 as $t) << index);
                    self
                }

                fn get_bit(&self, index: usize) -> bool {
                    (*self & ((1 as $t) << index)) != 0
                }

                fn set_bit(&mut self, index: usize) -> &mut Self {
                    *self |= (1 as $t) << index;
                    self
                }

                fn toggle_bit(&mut self, index: usize) -> &mut Self {
                    *self ^= (1 as $t) << index;
                    self
                }

                fn write_bit(&mut self, index: usize, value: bool) -> &mut Self {
                    let mask = (1 as $t) << index;
                    *self = (*self & !mask) | ((value as $t) << index);
                    self
                }

                // === Bitfield Operations ===
                fn apply_mask_lsb(&mut self, bits: usize) -> Self {
                    *self & self.create_mask_lsb(bits)
                }

                fn apply_mask_msb(&mut self, bits: usize) -> Self {
                    *self & self.create_mask_msb(bits)
                }

                fn create_mask_lsb(&self, bits: usize) -> Self {
                    let total_bits = std::mem::size_of::<$t>() * 8;
                    if bits >= total_bits {
                        $t::MAX
                    } else {
                        !($t::MAX << bits)
                    }
                }

                fn create_mask_msb(&self, bits: usize) -> Self {
                    let total_bits = std::mem::size_of::<$t>() * 8;
                    if bits >= total_bits {
                        $t::MAX
                    } else {
                        !($t::MAX >> bits)
                    }
                }

                fn clear_bits(&mut self, start: usize, len: usize) -> &mut Self {
                    let mask = if len >= (std::mem::size_of::<$t>() * 8) { $t::MAX } else { ((1 as $t) << len) - 1 };
                    *self &= !(mask << start);
                    self
                }

                fn get_bits(&self, start: usize, len: usize) -> Self {
                    let mask = if len >= (std::mem::size_of::<$t>() * 8) { $t::MAX } else { ((1 as $t) << len) - 1 };
                    (*self >> start) & mask
                }

                fn set_bits(&mut self, start: usize, len: usize) -> &mut Self {
                    let mask = if len >= (std::mem::size_of::<$t>() * 8) { $t::MAX } else { ((1 as $t) << len) - 1 };
                    *self |= (mask << start);
                    self
                }

                // === Bitwise Analysis ===
                fn is_power_of_two(&self) -> bool {
                    self.count_ones() == 1
                }

                fn parity(&self) -> bool {
                    self.count_ones() % 2 != 0
                }

                fn to_indices(&self) -> Vec<usize> {
                    let total_bits = std::mem::size_of::<$t>() * 8;
                    (0..total_bits)
                        .filter(|&idx| self.get_bit(idx))
                        .collect()
                }

                // === Transformation Operations ===
                fn isolate_lsb(&self) -> Self {
                    *self & self.wrapping_neg()
                }

                fn remove_lsb(&mut self) -> &mut Self {
                    if *self > 0 {
                        *self &= self.wrapping_sub(1 as $t);
                    }
                    self
                }

                fn swap_bits(&mut self, i: usize, j: usize) -> &mut Self {
                    if self.get_bit(i) != self.get_bit(j) {
                        let mask = ((1 as $t) << i) | ((1 as $t) << j);
                        *self ^= mask;
                    }
                    self
                }

                // === Conversion/Formatting Operations ===
                fn to_bstring(&self) -> String {
                    let total_bits = std::mem::size_of::<$t>() * 8;
                    format!("{:0width$b}", self, width = total_bits)
                }
            }
        )*
    };
}

// Call the macro once to generate the trait and all implementations
binary_array!(u8, u16, u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bit() {
        let test_num: u8 = 4;
        assert_eq!(test_num.get_bit(2), true);
        assert_eq!(test_num.get_bit(1), false);
    }

    #[test]
    fn test_write_bit1() {
        let mut test_num: u8 = 0;
        assert_eq!(*test_num.write_bit(2, true), 4);
    }

    #[test]
    fn test_write_bit2() {
        let mut test_num: u8 = 6;
        assert_eq!(*test_num.write_bit(1, false), 4);
    }

    #[test]
    fn test_print() {
        let num_8 = 69_u8;
        let num_16 = 10740_u16;

        assert_eq!(num_8.to_bstring(), "01000101".to_string());
        assert_eq!(num_16.to_bstring(), "0010100111110100".to_string());
    }
}
