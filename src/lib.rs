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
//! 
//! ## Functions not included as they are already standard
//! - count_ones()
//! - count_zeros()
//! - from_be_bytes()
//! - from_le_bytes()
//! - leading_ones()
//! - leading_zeros()
//! - to_be_bytes()
//! - to_le_bytes()

pub trait BinaryArray {
    /// Retrieves the bit value from index location
    fn get_bit(&self, index: usize) -> bool;

    /// Sets the bit value at index location
    fn set_bit(&mut self, index: usize, value: bool) -> Self;

    /// Formats the binary array as a padded string
    fn to_bstring(&self) -> String;
}

impl BinaryArray for u8 {
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_u8.wrapping_sub(value as u8)))
    }

    fn to_bstring(&self) -> String {
        format!("{:08b}", self)
    }
}

impl BinaryArray for u16 {
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_u16.wrapping_sub(value as u16)))
    }

    fn to_bstring(&self) -> String {
        format!("{:016b}", self)
    }
}

impl BinaryArray for u32 {
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_u32.wrapping_sub(value as u32)))
    }

    fn to_bstring(&self) -> String {
        format!("{:032b}", self)
    }
}

impl BinaryArray for u64 {
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_u64.wrapping_sub(value as u64)))
    }

    fn to_bstring(&self) -> String {
        format!("{:064b}", self)
    }
}

impl BinaryArray for u128 { 
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_u128.wrapping_sub(value as u128)))
    }

    fn to_bstring(&self) -> String {
        format!("{:0128b}", self)
    }
}

/// Ideally usize should not be used in a static binary array like this, but
/// was included anyway for completeness. One should know the architecture of
/// the target system being programmed for, and have the appropriate array size
/// selected.
impl BinaryArray for usize {
    fn get_bit(&self, index: usize) -> bool {
        (*self & (1 << index)) != 0
    }

    fn set_bit(&mut self, index: usize, value: bool) -> Self {
        let mask = 1 << index;
        *self & !mask | (mask & (0_usize.wrapping_sub(value as usize)))
    }

    fn to_bstring(&self) -> String {
        format!("{:0width$b}", self, width = std::mem::size_of::<usize>() * 8)
    }
}

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
    fn test_set_bit1() {
        let mut test_num: u8 = 0;
        assert_eq!(test_num.set_bit(2, true), 4);
    }

    #[test]
    fn test_set_bit2() {
        let mut test_num: u8 = 6;
        assert_eq!(test_num.set_bit(1, false), 4);
    }

    #[test]
    fn test_print() {
        let num_8 = 69_u8;
        let num_16 = 10740_u16;

        assert_eq!(num_8.to_bstring(), "01000101".to_string());
        assert_eq!(num_16.to_bstring(), "0010100111110100".to_string());
    }
}
