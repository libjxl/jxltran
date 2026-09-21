// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// In-memory model of a bare JXL codestream: one image header and a sequence of
// frames. Each frame keeps a parsed FrameHeader plus exact ranges into the
// source codestream.
//
// Frame bit layout: [frame_header][TOC][body]. For rewrite we emit a new
// frame_header, copy the TOC verbatim for now (later: re-encode a semantically
// equivalent TOC -- U32 size codewords are unique; only the permutation stream
// may have multiple encodings -- and permute body bytes to match). Then copy the
// body. The body is byte-aligned in the codestream (the TOC ends on a byte
// boundary); group sections are whole bytes.

#ifndef TOOLS_JXLTRAN_CODESTREAM_H_
#define TOOLS_JXLTRAN_CODESTREAM_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "frame_header.h"
#include "image_header.h"
#include "lfglobal.h"

namespace jxltran {

struct FramedUnit {
  FrameHeader frame;
  // Logical header for rewrite purposes is frame_header + TOC; the body is
  // everything after the TOC (byte-aligned in the codestream).
  // Bit offset from the start of this frame where the TOC begins (after the
  // frame header bits).
  size_t toc_start_bit = 0;
  // Length in bits of the TOC only (verbatim from input when writing).
  size_t toc_bit_length = 0;
  // Length in bits of compressed frame sections after the TOC; always a
  // multiple of 8 (whole bytes).
  size_t body_bit_length = 0;
  // Total byte length of this frame in the source codestream (header through
  // last frame data byte). Used to copy LF / ReferenceOnly frames verbatim.
  size_t full_frame_byte_len = 0;
  // Byte offset from the start of the codestream where this frame begins
  // (used as |original_data| base for WriteFrameHeader verbatim RF/ICC bits).
  size_t original_frame_byte_offset = 0;

  // TOC decoded for re-encode (bit-phase / photon-noise TOC rewrite). When
  // |toc_strip_perm_reorder|, the frame body is rebuilt in logical section
  // order and the rewritten TOC has no permutation. |toc_perm| empty means
  // none in source.
  std::vector<uint32_t> toc_decoded_sizes;
  std::vector<uint32_t> toc_perm;  // perm[s] = logical TOC index of stream slot s
  // When stripping TOC permutation, the body is still in original stream order
  // until write. |toc_strip_stream_sizes| holds stream-slot sizes before the
  // strip; |toc_strip_logical_to_stream| holds inv[L] = stream index of logical
  // section L (same as inverting toc_perm).
  std::vector<uint32_t> toc_strip_stream_sizes;
  std::vector<size_t> toc_strip_logical_to_stream;
  // Bits from the start of the TOC through padding before the first U32 size
  // codeword (permutation flag + optional Lehmer entropy + pad). Zero when
  // there is no TOC (num_toc_entries==0).
  size_t toc_bits_before_sizes = 0;

  // Photon-noise DC-global splice (see ApplyPhotonNoiseIso). Patch point is an
  // absolute codestream bit index (from first codestream byte = bit 0).
  bool photon_noise_edit = false;
  int32_t photon_noise_delta_bytes = 0;  // -10, 0, or +10
  size_t photon_noise_patch_abs_bit = 0;
  std::array<uint8_t, 10> photon_noise_new_bytes = {};
  bool photon_noise_toc_reencode = false;

  // Absolute codestream bit index of the first bit of the 80-bit photon-noise
  // LUT in DC global (after patches/splines). Filled by ReadCodestream when the
  // LF-global prefix parses; used by ApplyPhotonNoiseIso.
  size_t lf_global_noise_lut_abs_bit = 0;
  bool lf_global_noise_lut_abs_valid = false;
  // Offset in bits from the start of the LF-global TOC section to the first bit
  // of the 80-bit LUT (or insert point when adding noise). Set with abs_valid.
  size_t lf_global_noise_lut_rel_bit = 0;
  // Bit i set when this frame's patch dictionary reads reference slot i. Only
  // meaningful when |lf_global_noise_lut_abs_valid| (the prefix parsed).
  uint32_t lf_global_patch_ref_slot_mask = 0;
  // Set by ApplySetFrameTypes on the frames it retyped, with the type they had
  // before. Travels with the frame through --keep-listed-frames so the
  // post-pipeline validation can blame only what this run changed.
  bool frame_type_changed = false;
  uint32_t frame_type_before = 0;
  // Verbatim 80-bit LUT read at ReadCodestream time when kFrameFlagNoise (for
  // lossless undo / --set-photon-noise-weights round-trips).
  bool lf_global_noise_raw_valid = false;
  std::array<uint8_t, 10> lf_global_noise_raw_bytes{};
  // LF-global spline entropy region (bit indices; end exclusive). Valid after
  // a successful LF-global prefix parse: if kFrameFlagSplines and splines were
  // decoded, this spans the existing bundle; if the splines flag was clear,
  // start==end and marks the bit position where a new spline bundle is inserted
  // (before the noise LUT / following data).
  bool lf_global_spline_region_valid = false;
  size_t lf_global_spline_region_abs_start_bit = 0;
  size_t lf_global_spline_region_abs_end_bit = 0;
  // Same offsets relative to the first byte of logical TOC section 0 (LF
  // global bundle).
  size_t lf_global_spline_rel_start_bit = 0;
  size_t lf_global_spline_rel_end_bit = 0;
  // Decoded splines when the bitstream contained a spline bundle at read time.
  std::optional<LfGlobalSplines> lf_global_splines;

  // LF channel dequantization (libjxl DequantMatrices::DecodeDC) after noise in
  // logical LF-global section 0. Filled when LF-global prefix parse succeeds.
  bool lf_global_dc_quant_parsed = false;
  size_t lf_global_dc_quant_rel_region_start = 0;
  size_t lf_global_dc_quant_rel_region_old_len_bits = 0;
  bool lf_global_dc_quant_all_default = true;
  std::array<uint16_t, 3> lf_global_dc_quant_f16_bits{};
  bool lf_global_dc_quant_edit = false;
  int32_t lf_global_dc_quant_delta_bytes = 0;
  bool lf_global_dc_quant_toc_reencode = false;
  std::vector<uint8_t> lf_global_dc_quant_new_region_bits;
  size_t lf_global_dc_quant_new_region_len_bits = 0;

  // VarDCT: QuantizerParams (global_scale, quant_dc) after LF dequant in
  // LF-global section 0. Filled when LF-global prefix parse succeeds.
  bool lf_global_quantizer_parsed = false;
  size_t lf_global_quantizer_rel_region_start = 0;
  size_t lf_global_quantizer_rel_region_old_len_bits = 0;
  uint32_t lf_global_quantizer_global_scale = 0;
  uint32_t lf_global_quantizer_quant_dc = 0;
  bool lf_global_quantizer_edit = false;
  int32_t lf_global_quantizer_delta_bytes = 0;
  bool lf_global_quantizer_toc_reencode = false;
  std::vector<uint8_t> lf_global_quantizer_new_region_bits;
  size_t lf_global_quantizer_new_region_len_bits = 0;
  // Zero-bits appended after the LF-global chunk0 tail when replacing the
  // quantizer bundle if (new_len - old_len) mod 8 != 0 (chunk0 stays byte-sized).
  uint8_t lf_global_quantizer_tail_pad_bits = 0;
  // Re-encoded spline entropy (LF-global bit range); see ApplySplinesFromFile.
  bool spline_edit = false;
  int64_t spline_edit_delta_bits = 0;
  std::vector<uint8_t> spline_edit_new_mid_bytes;
  size_t spline_edit_new_mid_bits = 0;
  bool spline_edit_toc_reencode = false;

  // When true, WriteCodestream rebuilds the frame body in logical TOC order
  // and rewrites the TOC without permutation (e.g. --group_order=0 or
  // progressive normalization).
  bool toc_strip_perm_reorder = false;

  // Center-first (--group_order=1): body bytes are permuted in stream order.
  // |toc_body_stream_shuffle[s]| is the source stream-slot index (sizes in
  // |toc_body_shuffle_src_sizes|) whose section is written at new stream slot s
  // after |toc_decoded_sizes| / |toc_perm| were updated. Empty = disabled.
  std::vector<size_t> toc_body_stream_shuffle;
  std::vector<uint32_t> toc_body_shuffle_src_sizes;
};

struct ParsedCodestream {
  ImageHeader image;
  std::vector<FramedUnit> frames;
};

// Parses a complete codestream (0xFF0A …) into |out|. Returns false on error.
bool ReadCodestream(const uint8_t* data, size_t size, ParsedCodestream* out);

// Serializes |cs| to |out|. |original| must point at the same codestream bytes
// used when |cs| was produced by ReadCodestream (ICC entropy bits and frame RF
// ranges are copied verbatim relative to per-frame offsets).
bool WriteCodestream(const ParsedCodestream& cs, const uint8_t* original,
                     std::vector<uint8_t>* out);

// Decodes |fu|'s frame body into stream-order TOC chunks (sizes from
// |fu.toc_decoded_sizes|) using |original| as the codestream byte source.
bool ExtractFramedUnitBodyStreamChunks(const uint8_t* original, size_t orig_size,
                                       const FramedUnit& fu,
                                       std::vector<std::vector<uint8_t>>* chunks_out);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_CODESTREAM_H_
