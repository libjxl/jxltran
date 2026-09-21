// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lfglobal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include "codestream.h"
#include "entropy.h"
#include "frame_header.h"
#include "photon_noise.h"

namespace jxltran {

// lib/jxl/splines.h SplineEntropyContexts (LF-global spline decode)
enum SplineContexts : size_t {
  kQuantizationAdjustmentContext = 0,
  kStartingPositionContext,
  kNumSplinesContext,
  kNumControlPointsContext,
  kControlPointsContext,
  kDCTContext,
  kNumSplineContexts
};

constexpr size_t kMaxNumControlPoints = 1u << 20u;
constexpr size_t kMaxNumControlPointsPerPixelRatio = 2;

inline int64_t UnpackSignedHybrid(uint32_t u) {
  return static_cast<int64_t>(UnpackSigned(u));
}

inline bool FrameCanSaveAsReference(const FrameHeader& fh) {
  if (fh.is_last) return false;
  if (fh.frame_type == kFrameTypeLF) return false;
  if (fh.duration != 0 && fh.save_as_reference == 0) return false;
  return true;
}

namespace {

// lib/jxl/patch_dictionary_internal.h + dec_patch_dictionary.h
enum PatchContexts : size_t {
  kNumRefPatchContext = 0,
  kReferenceFrameContext = 1,
  kPatchSizeContext = 2,
  kPatchReferencePositionContext = 3,
  kPatchPositionContext = 4,
  kPatchBlendModeContext = 5,
  kPatchOffsetContext = 6,
  kPatchCountContext = 7,
  kPatchAlphaChannelContext = 8,
  kPatchClampContext = 9,
  kNumPatchDictionaryContexts
};

enum PatchBlendMode : uint8_t {
  kPatchBlendNone = 0,
  kPatchBlendReplace = 1,
  kPatchBlendAdd = 2,
  kPatchBlendMul = 3,
  kPatchBlendAbove = 4,
  kPatchBlendBelow = 5,
  kPatchBlendAlphaWeightedAddAbove = 6,
  kPatchBlendAlphaWeightedAddBelow = 7,
};

static constexpr uint8_t kNumPatchBlendModes = 8;

inline bool PatchUsesAlpha(uint32_t mode) {
  return mode == kPatchBlendAbove || mode == kPatchBlendBelow ||
         mode == kPatchBlendAlphaWeightedAddAbove ||
         mode == kPatchBlendAlphaWeightedAddBelow;
}

inline bool PatchUsesClamp(uint32_t mode) {
  return PatchUsesAlpha(mode) || mode == kPatchBlendMul;
}

bool SkipPatchDictionary(
    BitReader& br, size_t xsize_padded, size_t ysize_padded,
    size_t num_extra_channels,
    const std::array<std::pair<size_t, size_t>, kNumReferenceSlots>& ref_slots,
    uint32_t* ref_slot_mask) {
  EntropyCoder ec{};
  if (!InitEntropyCoder(br, kNumPatchDictionaryContexts, false, &ec)) {
    return false;
  }

  const auto read_num = [&](size_t ctx) -> uint32_t {
    return DecodeHybridUint(br, ec, ctx);
  };

  const size_t num_ref_patch = read_num(kNumRefPatchContext);
  const size_t num_pixels = xsize_padded * ysize_padded;
  const size_t max_ref_patches = 1024 + num_pixels / 4;
  const size_t max_patches = max_ref_patches * 4;
  const size_t max_blending_infos = max_patches * 4;
  if (num_ref_patch > max_ref_patches) {
    return false;
  }

  const size_t blendings_stride = num_extra_channels + 1;
  size_t total_patches = 0;
  size_t next_size = 1;
  size_t last_pos_x = 0;
  size_t last_pos_y = 0;

  for (size_t id = 0; id < num_ref_patch; id++) {
    const uint32_t ref = read_num(kReferenceFrameContext);
    if (ref >= kNumReferenceSlots) {
      return false;
    }
    if (ref_slot_mask != nullptr) *ref_slot_mask |= 1u << ref;
    const size_t ref_w = ref_slots[ref].first;
    const size_t ref_h = ref_slots[ref].second;
    if (ref_w == 0 || ref_h == 0) {
      return false;
    }

    const size_t ref_x0 = read_num(kPatchReferencePositionContext);
    const size_t ref_y0 = read_num(kPatchReferencePositionContext);
    const size_t ref_xsize = static_cast<size_t>(read_num(kPatchSizeContext)) + 1;
    const size_t ref_ysize = static_cast<size_t>(read_num(kPatchSizeContext)) + 1;
    if (ref_x0 + ref_xsize > ref_w || ref_y0 + ref_ysize > ref_h) {
      return false;
    }

    size_t id_count = read_num(kPatchCountContext);
    if (id_count > max_patches) {
      return false;
    }
    id_count++;
    total_patches += id_count;
    if (total_patches > max_patches) {
      return false;
    }
    if (next_size < total_patches) {
      next_size *= 2;
      next_size = std::min<size_t>(next_size, max_patches);
    }
    if (next_size * blendings_stride > max_blending_infos) {
      return false;
    }

    const bool choose_alpha = (num_extra_channels > 1);
    for (size_t i = 0; i < id_count; i++) {
      size_t pos_x;
      size_t pos_y;
      if (i == 0) {
        pos_x = read_num(kPatchPositionContext);
        pos_y = read_num(kPatchPositionContext);
      } else {
        const int64_t deltax =
            UnpackSignedHybrid(read_num(kPatchOffsetContext));
        const int64_t deltay =
            UnpackSignedHybrid(read_num(kPatchOffsetContext));
        if (deltax < 0 &&
            static_cast<size_t>(-deltax) > last_pos_x) {
          return false;
        }
        if (deltay < 0 &&
            static_cast<size_t>(-deltay) > last_pos_y) {
          return false;
        }
        pos_x = last_pos_x + static_cast<size_t>(deltax);
        pos_y = last_pos_y + static_cast<size_t>(deltay);
      }
      if (pos_x + ref_xsize > xsize_padded || pos_y + ref_ysize > ysize_padded) {
        return false;
      }
      last_pos_x = pos_x;
      last_pos_y = pos_y;

      for (size_t j = 0; j < blendings_stride; j++) {
        const uint32_t blend_mode = read_num(kPatchBlendModeContext);
        if (blend_mode >= kNumPatchBlendModes) {
          return false;
        }
        if (PatchUsesAlpha(blend_mode) && choose_alpha) {
          const uint32_t alpha_ch = read_num(kPatchAlphaChannelContext);
          if (alpha_ch >= num_extra_channels) {
            return false;
          }
        }
        if (PatchUsesClamp(blend_mode)) {
          (void)read_num(kPatchClampContext);
        }
      }
    }
  }

  if (!FinalizeEntropyCoder(ec)) {
    return false;
  }
  return br.ok();
}

}  // namespace

static bool DecodeAllStartingPoints(BitReader& br, EntropyCoder& ec,
                                    size_t num_splines,
                                    std::vector<std::pair<float, float>>* points) {
  points->clear();
  points->reserve(num_splines);
  int64_t last_x = 0;
  int64_t last_y = 0;
  for (size_t i = 0; i < num_splines; i++) {
    const uint32_t dx =
        DecodeHybridUint(br, ec, kStartingPositionContext);
    const uint32_t dy =
        DecodeHybridUint(br, ec, kStartingPositionContext);
    int64_t x;
    int64_t y;
    if (i != 0) {
      x = UnpackSignedHybrid(dx) + last_x;
      y = UnpackSignedHybrid(dy) + last_y;
    } else {
      x = static_cast<int64_t>(dx);
      y = static_cast<int64_t>(dy);
    }
    constexpr int64_t kLimit = 1ll << 23;
    if (x >= kLimit || x <= -kLimit || y >= kLimit || y <= -kLimit) {
      return false;
    }
    points->emplace_back(static_cast<float>(x), static_cast<float>(y));
    last_x = x;
    last_y = y;
  }
  return br.ok();
}

static bool DecodeQuantizedSpline(BitReader& br, EntropyCoder& ec,
                                  size_t max_control_points,
                                  size_t* total_num_control_points,
                                  QuantizedSplineData* spline) {
  const uint32_t num_control_points_u =
      DecodeHybridUint(br, ec, kNumControlPointsContext);
  const size_t num_control_points = num_control_points_u;
  if (num_control_points > max_control_points) {
    return false;
  }
  *total_num_control_points += num_control_points;
  if (*total_num_control_points > max_control_points) {
    return false;
  }
  spline->control_points.resize(num_control_points);
  constexpr int64_t kDeltaLimit = 1ll << 30;
  for (auto& cp : spline->control_points) {
    cp.first = UnpackSignedHybrid(
        DecodeHybridUint(br, ec, kControlPointsContext));
    cp.second = UnpackSignedHybrid(
        DecodeHybridUint(br, ec, kControlPointsContext));
    if (cp.first >= kDeltaLimit || cp.first <= -kDeltaLimit ||
        cp.second >= kDeltaLimit || cp.second <= -kDeltaLimit) {
      return false;
    }
  }

  const auto decode_dct = [&](std::array<int32_t, 32>* dct) -> bool {
    constexpr int kWeirdNumber = std::numeric_limits<int>::min();
    for (int i = 0; i < 32; ++i) {
      const int32_t v = static_cast<int32_t>(
          UnpackSignedHybrid(DecodeHybridUint(br, ec, kDCTContext)));
      (*dct)[static_cast<size_t>(i)] = v;
      if (v == kWeirdNumber) {
        return false;
      }
    }
    return true;
  };
  for (size_t c = 0; c < 3; ++c) {
    if (!decode_dct(&spline->color_dct[c])) {
      return false;
    }
  }
  if (!decode_dct(&spline->sigma_dct)) {
    return false;
  }
  return true;
}

bool DecodeSplinesBundle(BitReader& br, size_t num_pixels,
                         LfGlobalSplines* out) {
  EntropyCoder ec{};
  if (!InitEntropyCoder(br, kNumSplineContexts, false, &ec)) {
    return false;
  }

  size_t num_splines = DecodeHybridUint(br, ec, kNumSplinesContext);
  const size_t max_control_points =
      std::min(kMaxNumControlPoints, num_pixels / kMaxNumControlPointsPerPixelRatio);
  if (num_splines > max_control_points || num_splines + 1 > max_control_points) {
    return false;
  }
  num_splines++;
  if (!DecodeAllStartingPoints(br, ec, num_splines, &out->starting_points)) {
    return false;
  }

  out->quantization_adjustment = static_cast<int32_t>(UnpackSignedHybrid(
      DecodeHybridUint(br, ec, kQuantizationAdjustmentContext)));

  out->splines.clear();
  out->splines.reserve(num_splines);
  size_t num_control_points = num_splines;
  for (size_t i = 0; i < num_splines; ++i) {
    QuantizedSplineData sp{};
    if (!DecodeQuantizedSpline(br, ec, max_control_points, &num_control_points,
                               &sp)) {
      return false;
    }
    out->splines.push_back(std::move(sp));
  }

  if (!FinalizeEntropyCoder(ec)) {
    return false;
  }
  if (out->splines.empty()) {
    return false;
  }
  return br.ok();
}

namespace {

bool ReadNoiseLut80(BitReader& br, std::array<uint8_t, 10>* bytes) {
  size_t bit_in = 0;
  bytes->fill(0);
  for (int i = 0; i < 80; ++i) {
    const uint32_t b = br.ReadBits(1);
    if (!br.ok()) {
      return false;
    }
    if (b) {
      (*bytes)[bit_in >> 3] |= static_cast<uint8_t>(1u << (bit_in & 7));
    }
    bit_in++;
  }
  return true;
}

}  // namespace

bool LfGlobalPaddedPatchCanvas(const FrameHeader& fh, const ImageMetadata& meta,
                               uint32_t canvas_w, uint32_t canvas_h,
                               size_t* xsize_padded, size_t* ysize_padded) {
  FrameTocMetrics m{};
  ComputeFrameTocMetrics(fh, meta, canvas_w, canvas_h, &m);
  if (fh.encoding == kFrameEncModular) {
    *xsize_padded = m.xsize;
    *ysize_padded = m.ysize;
  } else {
    *xsize_padded = m.xsize_blocks * 8;
    *ysize_padded = m.ysize_blocks * 8;
  }
  return *xsize_padded != 0 && *ysize_padded != 0;
}

void LfGlobalReferenceSlotSizes(const ParsedCodestream& cs, size_t frame_index,
                                const ImageMetadata& meta, uint32_t canvas_w,
                                uint32_t canvas_h,
                                std::array<std::pair<size_t, size_t>,
                                           kNumReferenceSlots>* out) {
  out->fill({0, 0});
  for (size_t j = 0; j < frame_index && j < cs.frames.size(); ++j) {
    const FrameHeader& fh = cs.frames[j].frame;
    if (!FrameCanSaveAsReference(fh)) {
      continue;
    }
    size_t pw = 0;
    size_t ph = 0;
    if (!LfGlobalPaddedPatchCanvas(fh, meta, canvas_w, canvas_h, &pw, &ph)) {
      continue;
    }
    if (fh.save_as_reference >= kNumReferenceSlots) {
      continue;
    }
    (*out)[fh.save_as_reference] = {pw, ph};
  }
}

bool ReadLfGlobalThroughNoise(
    BitReader& br, const FrameHeader& fh, const ImageMetadata& meta,
    uint32_t canvas_w, uint32_t canvas_h,
    const std::array<std::pair<size_t, size_t>, kNumReferenceSlots>& ref_slots,
    LfGlobalThroughNoise* out) {
  out->splines.reset();

  if ((fh.flags & kFrameFlagPatches) != 0) {
    size_t xp = 0;
    size_t yp = 0;
    if (!LfGlobalPaddedPatchCanvas(fh, meta, canvas_w, canvas_h, &xp, &yp)) {
      return false;
    }
    if (!SkipPatchDictionary(br, xp, yp, meta.num_extra, ref_slots,
                             &out->patch_ref_slot_mask)) {
      return false;
    }
  }

  if ((fh.flags & kFrameFlagSplines) != 0) {
    out->splines_entropy_start_bit = br.pos();
    FrameTocMetrics m{};
    ComputeFrameTocMetrics(fh, meta, canvas_w, canvas_h, &m);
    const size_t num_pixels = m.xsize * m.ysize;
    if (num_pixels == 0) {
      return false;
    }
    LfGlobalSplines spl{};
    if (!DecodeSplinesBundle(br, num_pixels, &spl)) {
      return false;
    }
    out->splines = std::move(spl);
  } else {
    out->splines_entropy_start_bit = br.pos();
  }

  out->noise_lut_start_bit = br.pos();

  if ((fh.flags & kFrameFlagNoise) != 0) {
    std::array<uint8_t, 10> noise_bytes{};
    if (!ReadNoiseLut80(br, &noise_bytes)) {
      return false;
    }
    out->noise_lut_bytes_valid = true;
    out->noise_lut_bytes = noise_bytes;
    std::array<float, kNoiseLutPoints> lut{};
    if (!DecodeNoiseLutBits(noise_bytes, &lut)) {
      return false;
    }
  }

  out->dc_quant_start_bit = br.pos();
  const uint32_t all_default_dc = br.ReadBits(1);
  if (!br.ok()) return false;
  out->dc_quant_all_default = (all_default_dc != 0);
  if (!out->dc_quant_all_default) {
    for (int c = 0; c < 3; ++c) {
      out->dc_quant_f16_bits[static_cast<size_t>(c)] =
          static_cast<uint16_t>(br.ReadF16());
      if (!br.ok()) return false;
    }
  } else {
    out->dc_quant_f16_bits.fill(0);
  }
  out->dc_quant_end_bit = br.pos();

  out->quantizer_parsed = false;
  if (fh.encoding != kFrameEncModular) {
    out->quantizer_start_bit = br.pos();
    out->quantizer_global_scale =
        ReadU32(br, U32Dist::BitsOffset(11, 1), U32Dist::BitsOffset(11, 2049),
                 U32Dist::BitsOffset(12, 4097), U32Dist::BitsOffset(16, 8193));
    out->quantizer_quant_dc =
        ReadU32(br, U32Dist::Imm(16), U32Dist::BitsOffset(5, 1),
                 U32Dist::BitsOffset(8, 1), U32Dist::BitsOffset(16, 1));
    if (!br.ok()) return false;
    out->quantizer_end_bit = br.pos();
    out->quantizer_parsed = true;
  }

  return br.ok();
}

}  // namespace jxltran
