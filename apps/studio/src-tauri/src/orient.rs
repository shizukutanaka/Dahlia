//! EXIF orientation handling for source photos (W86).
//!
//! Phone photos are usually stored rotated, with the intended
//! viewing orientation recorded only in the EXIF Orientation tag.
//! The Python workers apply that orientation when loading a photo
//! (PIL's exif_transpose via workers/common.py), so the Rust side
//! must apply the same correction wherever it reads the original
//! pixels — thumbnails and face crops — or coordinates and visuals
//! would disagree with detection results.
//!
//! This module is deliberately std-only (byte-level parsing, no
//! image types) so the parsing and mapping logic stays unit-testable
//! without the imaging stack. Applying a Transform to a decoded
//! image lives next to the image loading code in lib.rs.

/// The geometric correction needed to display an image upright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    None,
    FlipHorizontal,
    Rotate180,
    FlipVertical,
    /// Mirror over the main diagonal (EXIF 5):
    /// rotate 90° clockwise, then flip horizontally.
    Transpose,
    /// Stored rotated 90° counter-clockwise (EXIF 6):
    /// rotate 90° clockwise to correct.
    Rotate90Cw,
    /// Mirror over the anti-diagonal (EXIF 7):
    /// rotate 90° clockwise, then flip vertically.
    Transverse,
    /// Stored rotated 90° clockwise (EXIF 8):
    /// rotate 90° counter-clockwise to correct.
    Rotate270Cw,
}

/// Map an EXIF Orientation value (1..=8) to its correction.
/// Absent or unknown values mean no correction.
pub fn transform_for(orientation: u8) -> Transform {
    match orientation {
        2 => Transform::FlipHorizontal,
        3 => Transform::Rotate180,
        4 => Transform::FlipVertical,
        5 => Transform::Transpose,
        6 => Transform::Rotate90Cw,
        7 => Transform::Transverse,
        8 => Transform::Rotate270Cw,
        _ => Transform::None,
    }
}

/// The correction required by a JPEG file's contents
/// (Transform::None for non-JPEG data or a missing tag).
pub fn transform_for_jpeg(bytes: &[u8]) -> Transform {
    transform_for(jpeg_orientation(bytes))
}

/// Read the EXIF Orientation tag from JPEG bytes.
/// Returns 0 when the data is not a JPEG, has no EXIF segment,
/// or the tag is absent or malformed.
pub fn jpeg_orientation(bytes: &[u8]) -> u8 {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return 0;
    }
    let mut pos = 2usize;
    while pos + 4 <= bytes.len() {
        if bytes[pos] != 0xFF {
            return 0;
        }
        let marker = bytes[pos + 1];
        if marker == 0xDA || marker == 0xD9 {
            return 0; // start of scan / end of image: no EXIF beyond here
        }
        let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        if len < 2 || pos + 2 + len > bytes.len() {
            return 0;
        }
        let segment = &bytes[pos + 4..pos + 2 + len];
        if marker == 0xE1 && segment.starts_with(b"Exif\0\0") {
            return tiff_orientation(&segment[6..]).unwrap_or(0);
        }
        pos += 2 + len;
    }
    0
}

fn tiff_orientation(tiff: &[u8]) -> Option<u8> {
    if tiff.len() < 8 {
        return None;
    }
    let little = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |b: &[u8]| -> u16 {
        if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }
    };
    let u32_at = |b: &[u8]| -> u32 {
        if little {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    if u16_at(&tiff[2..4]) != 42 {
        return None;
    }
    let ifd_off = u32_at(&tiff[4..8]) as usize;
    if ifd_off + 2 > tiff.len() {
        return None;
    }
    let count = u16_at(&tiff[ifd_off..ifd_off + 2]) as usize;
    for i in 0..count {
        let entry_off = ifd_off + 2 + i * 12;
        if entry_off + 12 > tiff.len() {
            return None;
        }
        let entry = &tiff[entry_off..entry_off + 12];
        if u16_at(&entry[0..2]) == 0x0112 {
            // Orientation is a SHORT; its value sits inline in the
            // first two bytes of the 4-byte value field either way.
            return u8::try_from(u16_at(&entry[8..10])).ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg_with_orientation(orientation: u16, little_endian: bool) -> Vec<u8> {
        let mut tiff = Vec::new();
        if little_endian {
            tiff.extend_from_slice(b"II");
            tiff.extend_from_slice(&42u16.to_le_bytes());
            tiff.extend_from_slice(&8u32.to_le_bytes());
            tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
            tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
            tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
            tiff.extend_from_slice(&1u32.to_le_bytes()); // count
            tiff.extend_from_slice(&orientation.to_le_bytes());
            tiff.extend_from_slice(&0u16.to_le_bytes());
            tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD
        } else {
            tiff.extend_from_slice(b"MM");
            tiff.extend_from_slice(&42u16.to_be_bytes());
            tiff.extend_from_slice(&8u32.to_be_bytes());
            tiff.extend_from_slice(&1u16.to_be_bytes());
            tiff.extend_from_slice(&0x0112u16.to_be_bytes());
            tiff.extend_from_slice(&3u16.to_be_bytes());
            tiff.extend_from_slice(&1u32.to_be_bytes());
            tiff.extend_from_slice(&orientation.to_be_bytes());
            tiff.extend_from_slice(&0u16.to_be_bytes());
            tiff.extend_from_slice(&0u32.to_be_bytes());
        }
        let mut segment = b"Exif\0\0".to_vec();
        segment.extend_from_slice(&tiff);
        let mut jpg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        jpg.extend_from_slice(&((segment.len() + 2) as u16).to_be_bytes());
        jpg.extend_from_slice(&segment);
        jpg
    }

    #[test]
    fn parses_little_endian_orientation() {
        assert_eq!(jpeg_orientation(&jpeg_with_orientation(6, true)), 6);
        assert_eq!(
            transform_for_jpeg(&jpeg_with_orientation(6, true)),
            Transform::Rotate90Cw
        );
    }

    #[test]
    fn parses_big_endian_orientation() {
        assert_eq!(jpeg_orientation(&jpeg_with_orientation(3, false)), 3);
        assert_eq!(
            transform_for_jpeg(&jpeg_with_orientation(8, false)),
            Transform::Rotate270Cw
        );
    }

    #[test]
    fn missing_or_malformed_exif_means_no_correction() {
        assert_eq!(jpeg_orientation(&[]), 0);
        assert_eq!(jpeg_orientation(&[0xFF, 0xD8, 0xFF, 0xD9]), 0);
        assert_eq!(jpeg_orientation(b"not a jpeg at all"), 0);
        // APP1 without the Exif header.
        let jpg = [0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x04, b'a', b'b'];
        assert_eq!(jpeg_orientation(&jpg), 0);
        assert_eq!(transform_for(0), Transform::None);
        assert_eq!(transform_for(1), Transform::None);
        assert_eq!(transform_for(9), Transform::None);
    }

    #[test]
    fn all_orientations_map() {
        assert_eq!(transform_for(2), Transform::FlipHorizontal);
        assert_eq!(transform_for(3), Transform::Rotate180);
        assert_eq!(transform_for(4), Transform::FlipVertical);
        assert_eq!(transform_for(5), Transform::Transpose);
        assert_eq!(transform_for(6), Transform::Rotate90Cw);
        assert_eq!(transform_for(7), Transform::Transverse);
        assert_eq!(transform_for(8), Transform::Rotate270Cw);
    }
}
