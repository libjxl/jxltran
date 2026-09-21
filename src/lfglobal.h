// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DC / LF global bundle prefix: patches, splines, then photon-noise LUT (80
// bits). Matches libjxl FrameDecoder::ProcessDCGlobal order (dec_frame.cc).

#ifndef TOOLS_JXLTRAN_LFGLOBAL_H_
#define TOOLS_JXLTRAN_LFGLOBAL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "bits.h"
#include "frame_header.h"
#include "image_header.h"

namespace jxltran {

struct ParsedCodestream;

static constexpr size_t kNumReferenceSlots = 4;

// Padded pixel grid used by libjxl PatchDictionary::Decode (FrameDimensions
// xsize_padded / ysize_padded).
bool LfGlobalPaddedPatchCanvas(const FrameHeader& fh, const ImageMetadata& meta,
                               uint32_t canvas_w, uint32_t canvas_h,
                               size_t* xsize_padded, size_t* ysize_padded);

// After decoding frames [0, frame_index), each slot holds the padded (W,H) of
// the last frame that saved into that slot, or (0,0) if empty. Matches
// reference buffer sizes used for patch dictionary bounds checks.
void LfGlobalReferenceSlotSizes(const ParsedCodestream& cs, size_t frame_index,
                                const ImageMetadata& meta, uint32_t canvas_w,
                                uint32_t canvas_h,
                                std::array<std::pair<size_t, size_t>,
                                           kNumReferenceSlots>* out);

// Quantized spline (lib/jxl/splines.h QuantizedSpline), decoded form only.
struct QuantizedSplineData {
  std::vector<std::pair<int64_t, int64_t>> control_points;
  std::array<std::array<int32_t, 32>, 3> color_dct{};
  std::array<int32_t, 32> sigma_dct{};
};

struct LfGlobalSplines {
  int32_t quantization_adjustment = 0;
  std::vector<std::pair<float, float>> starting_points;
  std::vector<QuantizedSplineData> splines;
};

struct LfGlobalThroughNoise {
  // Bit offset within the LF-global section (|br| at entry = 0) where spline
  // entropy begins; equals |noise_lut_start_bit| when kSplines is false.
  size_t splines_entropy_start_bit = 0;
  // Bit offset from the start of the LF-global TOC section bitstream where the
  // 8×10-bit noise LUT begins (or would begin if kNoise were set).
  size_t noise_lut_start_bit = 0;
  std::optional<LfGlobalSplines> splines;
  // Bit i set when the patch dictionary reads reference slot i. Such a slot
  // must hold a frame saved before the colour transform (libjxl
  // dec_patch_dictionary.cc: "Patches cannot use frames saved post color
  // transforms").
  uint32_t patch_ref_slot_mask = 0;
  // When kFrameFlagNoise, the verbatim 80-bit LUT read from the bitstream.
  bool noise_lut_bytes_valid = false;
  std::array<uint8_t, 10> noise_lut_bytes{};

  // DequantMatrices::DecodeDC immediately after the noise block (or noise
  // insert point): 1-bit all_default, or 0 + 3×F16 wire values (pre-×1/128).
  size_t dc_quant_start_bit = 0;
  size_t dc_quant_end_bit = 0;
  bool dc_quant_all_default = true;
  std::array<uint16_t, 3> dc_quant_f16_bits{};

  // VarDCT only: QuantizerParams bundle (libjxl Quantizer::Decode) immediately
  // after the LF dequant region — global_scale U32 then quant_dc U32.
  bool quantizer_parsed = false;
  size_t quantizer_start_bit = 0;
  size_t quantizer_end_bit = 0;
  uint32_t quantizer_global_scale = 0;
  uint32_t quantizer_quant_dc = 0;
};

bool DecodeSplinesBundle(BitReader& br, size_t num_pixels,
                         LfGlobalSplines* out);

// |br| must be sized to the LF-global section bytes only (first TOC entry when
// present). Advances |br| past patches, splines, and the 80-bit noise LUT when
// kNoise; leaves |br| at the first bit after the LF channel dequantization
// bundle (DecodeDC). On success, fills |out| and |br.pos()| is the first bit
// after that bundle within the section.
bool ReadLfGlobalThroughNoise(
    BitReader& br, const FrameHeader& fh, const ImageMetadata& meta,
    uint32_t canvas_w, uint32_t canvas_h,
    const std::array<std::pair<size_t, size_t>, kNumReferenceSlots>& ref_slots,
    LfGlobalThroughNoise* out);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_LFGLOBAL_H_
