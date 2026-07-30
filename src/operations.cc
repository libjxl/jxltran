// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "operations.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bits.h"
#include "codestream.h"
#include "entropy.h"
#include "frame_header.h"
#include "photon_noise.h"
#include "printf_macros.h"
#include "lfglobal.h"
#include "orientation_compose.h"
#include "restoration_filter.h"
#include "spline_io.h"

namespace jxltran {

namespace {

constexpr float kImplicitIntensityTargetNits = 255.f;

static bool BitDepthOkForExtra(const BitDepth& bd) {
  if (bd.float_sample) {
    if (bd.bits_per_sample < 5u || bd.bits_per_sample > 32u) return false;
    if (bd.exp_bits < 2u || bd.exp_bits > 8u) return false;
    const int mant =
        static_cast<int>(bd.bits_per_sample) - static_cast<int>(bd.exp_bits) -
        1;
    if (mant < 2 || mant > 23) return false;
    return true;
  }
  return bd.bits_per_sample >= 1u && bd.bits_per_sample <= 31u;
}

// Main image header BitDepth: integer paths allow up to 32 bits (U32 selector).
static bool BitDepthOkForMainImage(const BitDepth& bd) {
  if (bd.float_sample) {
    return BitDepthOkForExtra(bd);
  }
  return bd.bits_per_sample >= 1u && bd.bits_per_sample <= 32u;
}

static void NormalizeToneMappingBundle(ToneMapping* tm) {
  if (tm->all_default) return;
  if (tm->relative_to_max_display) return;
  if (tm->min_nits != FloatToF16Bits(0.f)) return;
  if (tm->linear_below != FloatToF16Bits(0.f)) return;
  const float it = F16BitsToFloat(tm->intensity_target);
  if (!std::isfinite(it)) return;
  if (std::fabs(it - kImplicitIntensityTargetNits) > 1e-3f) return;
  *tm = ToneMapping{};
}

}  // namespace

bool ApplyHeaderMod(ParsedCodestream* cs, const HeaderMod& mod) {
  ImageMetadata* m = &cs->image.metadata;
  if (mod.have_tone_mapping_snapshot) {
    m->tone_mapping = mod.tone_mapping_snapshot;
    NormalizeToneMappingBundle(&m->tone_mapping);
    if (mod.collapse_to_packed_all_default_image_metadata &&
        m->tone_mapping.all_default) {
      m->all_default = true;
    }
  }
  if (mod.set_orientation != 0) {
    if (mod.set_orientation < 1 || mod.set_orientation > 8) {
      fprintf(stderr,
              "jxltran: --set-orientation value %u is out of range [1,8]\n",
              mod.set_orientation);
      return false;
    }
    m->orientation = mod.set_orientation;
    if (mod.set_orientation != 1) {
      m->all_default = false;
    }
  }

  if (mod.set_bits_per_sample != 0) {
    m->bit_depth.bits_per_sample = mod.set_bits_per_sample;
    m->all_default = false;
  }
  if (mod.set_main_float_sample) {
    m->bit_depth.float_sample = mod.main_float_sample;
    m->all_default = false;
  }
  if (mod.set_main_exp_bits) {
    m->bit_depth.exp_bits = mod.main_exp_bits;
    m->all_default = false;
  }
  if (mod.set_bits_per_sample != 0 || mod.set_main_float_sample ||
      mod.set_main_exp_bits) {
    if (!BitDepthOkForMainImage(m->bit_depth)) {
      fprintf(stderr,
              "jxltran: invalid main bit depth (integer: 1..32 bits; float: "
              "same rules as extra channels: total bits 5..32, exp_bits 2..8, "
              "mantissa 2..23)\n");
      return false;
    }
  }

  if (mod.have_set_num_loops) {
    if (!m->have_animation) {
      fprintf(stderr,
              "jxltran: --set-num-loops requires an animated image "
              "(have_animation=true)\n");
      return false;
    }
    m->animation.num_loops = mod.set_num_loops;
  }

  if (mod.have_set_tps) {
    if (!m->have_animation) {
      fprintf(stderr,
              "jxltran: --set-tps requires an animated image "
              "(have_animation=true)\n");
      return false;
    }
    m->animation.tps_numerator = mod.set_tps_numerator;
    m->animation.tps_denominator = mod.set_tps_denominator;
  }

  if (mod.have_set_intensity_target) {
    if (mod.have_tone_mapping_snapshot) {
      fprintf(stderr,
              "jxltran: internal error: intensity target and tone-mapping "
              "snapshot both set\n");
      return false;
    }
    if (m->all_default) {
      if (mod.intensity_target_to_default_bundle) {
        return true;
      }
      // Expand packed image metadata in memory to explicit decoder-default
      // fields (already represented in |*m| after ReadImageMetadata skipped
      // them) so ToneMapping and other header writes can be emitted.
      m->all_default = false;
    }
    if (mod.intensity_target_to_default_bundle) {
      m->tone_mapping = ToneMapping{};
    } else {
      if (!std::isfinite(mod.intensity_target_nits) ||
          !(mod.intensity_target_nits > 0.f)) {
        fprintf(stderr,
                "jxltran: --set-intensity-target: expected a finite value > 0 "
                "(nits), or 'default'\n");
        return false;
      }
      if (m->tone_mapping.all_default) {
        m->tone_mapping.all_default = false;
        m->tone_mapping.intensity_target =
            FloatToF16Bits(mod.intensity_target_nits);
        m->tone_mapping.min_nits = FloatToF16Bits(0.f);
        m->tone_mapping.relative_to_max_display = false;
        m->tone_mapping.linear_below = FloatToF16Bits(0.f);
      } else {
        m->tone_mapping.all_default = false;
        m->tone_mapping.intensity_target =
            FloatToF16Bits(mod.intensity_target_nits);
      }
      NormalizeToneMappingBundle(&m->tone_mapping);
    }
  }

  for (const ExtraChannelHeaderPatch& p : mod.extra_channel_patches) {
    if (p.index >= m->ec_info.size()) {
      fprintf(stderr,
              "jxltran: --set-extra-channel: channel index %" PRIuS " is out of "
              "range (num_extra=%" PRIuS ")\n",
              p.index, static_cast<size_t>(m->num_extra));
      return false;
    }
    ExtraChannelInfo& ec = m->ec_info[p.index];
    if (p.set_d_alpha) {
      ec.d_alpha = p.d_alpha;
      if (ec.d_alpha) {
        ec.type = ExtraChannelType::kAlpha;
        ec.name.clear();
        ec.bit_depth = m->bit_depth;
        ec.alpha_associated = false;
        ec.spot_red = ec.spot_green = ec.spot_blue = ec.spot_solidity = 0;
        ec.cfa_channel = 1;
      }
    }
    if (!ec.d_alpha) {
      if (p.set_type) {
        ec.type = p.type;
      }
      if (p.set_ec_bits_per_sample || p.set_ec_float_sample ||
          p.set_ec_exp_bits) {
        BitDepth merged = ec.bit_depth;
        if (p.set_ec_bits_per_sample) {
          merged.bits_per_sample = p.bit_depth.bits_per_sample;
        }
        if (p.set_ec_float_sample) {
          merged.float_sample = p.bit_depth.float_sample;
        }
        if (p.set_ec_exp_bits) {
          merged.exp_bits = p.bit_depth.exp_bits;
        }
        if (!BitDepthOkForExtra(merged)) {
          fprintf(stderr,
                  "jxltran: --set-extra-channel: invalid bit depth (integer "
                  "channels: 1..31 bits; float: total bits 5..32, exp_bits "
                  "2..8, mantissa 2..23)\n");
          return false;
        }
        ec.bit_depth = merged;
      }
      if (p.set_name) {
        ec.name = p.name;
      }
      if (p.set_alpha_associated) {
        if (ec.type != ExtraChannelType::kAlpha) {
          fprintf(stderr,
                  "jxltran: --set-extra-channel: alpha_associated applies only "
                  "to type alpha\n");
          return false;
        }
        ec.alpha_associated = p.alpha_associated;
      }
      if (p.set_spot_r) ec.spot_red = FloatToF16Bits(p.spot_r);
      if (p.set_spot_g) ec.spot_green = FloatToF16Bits(p.spot_g);
      if (p.set_spot_b) ec.spot_blue = FloatToF16Bits(p.spot_b);
      if (p.set_spot_solidity) {
        ec.spot_solidity = FloatToF16Bits(p.spot_solidity);
      }
      if (p.set_cfa_channel) {
        ec.cfa_channel = p.cfa_channel;
      }
    }
    m->all_default = false;
  }

  if (mod.strip_icc) {
    m->colour_encoding.want_icc = false;
    m->all_default = false;
  }

  if (mod.set_transfer) {
    m->colour_encoding.tf.have_gamma = false;
    m->colour_encoding.tf.gamma = 0;
    m->colour_encoding.tf.transfer_function = mod.transfer_function;
    m->all_default = false;
  }


  if (mod.set_white_point) {
    m->colour_encoding.white_point = static_cast<WhitePoint>(mod.white_point);
    m->all_default = false;
  }

  if (mod.set_primaries) {
    m->colour_encoding.primaries = static_cast<Primaries>(mod.primaries);
    m->all_default = false;
  }

  if (mod.set_rendering_intent) {
    m->colour_encoding.rendering_intent = static_cast<RenderingIntent>(mod.rendering_intent);
    m->all_default = false;
  }

  return true;
}

bool RestoreExtraChannelInfoAtIndex(ParsedCodestream* cs, size_t index,
                                    const ExtraChannelInfo& snap) {
  ImageMetadata* m = &cs->image.metadata;
  if (index >= m->ec_info.size()) return false;
  m->ec_info[index] = snap;
  m->all_default = false;
  return true;
}

// Output pixel (px,py) samples input pixel (px+dx, py+dy). Equivalently,
// input (ix,iy) appears at output (ix-dx, iy-dy). The codestream canvas size
// (SizeHeader) becomes new_w x new_h; Regular / SkipProgressive frame headers
// keep the same encoded dimensions (crop_width x crop_height, or full canvas
// when !have_crop) with updated origins. The decoder only composites the
// overlap with the canvas. LF and ReferenceOnly frame headers are left
// unchanged in memory; WriteCodestream emits those frames verbatim from the
// original codestream so their bit length and TOC stay aligned with payload.

static void ClearRedundantFullCanvasCrop(FrameHeader* fh, uint32_t canvas_w,
                                         uint32_t canvas_h) {
  if (!fh->have_crop) return;
  if (fh->frame_type == kFrameTypeLF ||
      fh->frame_type == kFrameTypeReferenceOnly) {
    return;
  }
  const int32_t x0 = UnpackSigned(fh->ux0);
  const int32_t y0 = UnpackSigned(fh->uy0);
  if (x0 == 0 && y0 == 0 && fh->crop_width == canvas_w &&
      fh->crop_height == canvas_h) {
    fh->have_crop = false;
    fh->ux0 = 0;
    fh->uy0 = 0;
    fh->crop_width = 0;
    fh->crop_height = 0;
  }
}

bool ApplyCrop(ParsedCodestream* cs, int32_t dx, int32_t dy, uint32_t new_w,
               uint32_t new_h) {
  const uint32_t old_w = cs->image.size.width;
  const uint32_t old_h = cs->image.size.height;

  if (new_w == 0 || new_h == 0) {
    fprintf(stderr, "jxltran: crop dimensions must be non-zero\n");
    return false;
  }

  if (new_w == old_w && new_h == old_h && dx == 0 && dy == 0) {
    return true;
  }

  cs->image.size.width = new_w;
  cs->image.size.height = new_h;

  for (FramedUnit& fu : cs->frames) {
    FrameHeader* fh = &fu.frame;
    const uint32_t ft = fh->frame_type;

    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }

    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
      continue;
    }

    int64_t x0;
    int64_t y0;
    uint32_t fw;
    uint32_t fhh;
    if (!fh->have_crop) {
      x0 = 0;
      y0 = 0;
      fw = old_w;
      fhh = old_h;
    } else {
      x0 = UnpackSigned(fh->ux0);
      y0 = UnpackSigned(fh->uy0);
      fw = fh->crop_width;
      fhh = fh->crop_height;
    }

    const int64_t nx = x0 - static_cast<int64_t>(dx);
    const int64_t ny = y0 - static_cast<int64_t>(dy);
    if (nx < std::numeric_limits<int32_t>::min() ||
        nx > std::numeric_limits<int32_t>::max() ||
        ny < std::numeric_limits<int32_t>::min() ||
        ny > std::numeric_limits<int32_t>::max()) {
      fprintf(stderr, "jxltran: crop: frame origin overflow\n");
      return false;
    }

    fh->all_default = false;
    fh->have_crop = true;
    fh->ux0 = PackSigned(static_cast<int32_t>(nx));
    fh->uy0 = PackSigned(static_cast<int32_t>(ny));
    fh->crop_width = fw;
    fh->crop_height = fhh;

    ClearRedundantFullCanvasCrop(fh, new_w, new_h);
  }

  return true;
}

bool ApplySetFrameRegionFromDisplay(ParsedCodestream* cs, size_t frame_index,
                                    uint32_t display_space_orientation,
                                    int32_t display_x, int32_t display_y,
                                    uint32_t display_w, uint32_t display_h,
                                    bool* out_changed) {
  if (out_changed) *out_changed = false;
  if (frame_index >= cs->frames.size()) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame index %" PRIuS " out of range\n",
            frame_index);
    return false;
  }
  FramedUnit& fu = cs->frames[frame_index];
  FrameHeader* fh = &fu.frame;
  const uint32_t ft = fh->frame_type;
  if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " cannot be repositioned "
            "(LF or reference-only)\n",
            frame_index);
    return false;
  }
  if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " has unsupported type\n",
            frame_index);
    return false;
  }
  uint32_t orient = display_space_orientation;
  if (orient < 1 || orient > 8) orient = 1;
  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  int64_t sx_old = 0;
  int64_t sy_old = 0;
  uint64_t rw = canvas_w;
  uint64_t rh = canvas_h;
  if (fh->have_crop) {
    sx_old = UnpackSigned(fh->ux0);
    sy_old = UnpackSigned(fh->uy0);
    rw = fh->crop_width;
    rh = fh->crop_height;
  }

  int64_t cur_dx = 0;
  int64_t cur_dy = 0;
  uint64_t cur_dw = 0;
  uint64_t cur_dh = 0;
  StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx_old, sy_old, rw, rh,
                           &cur_dx, &cur_dy, &cur_dw, &cur_dh);

  if (!fh->have_crop) {
    if (static_cast<uint64_t>(display_w) != cur_dw ||
        static_cast<uint64_t>(display_h) != cur_dh) {
      fprintf(stderr,
              "jxltran: --set-frame-region: frame %" PRIuS " covers the full canvas; "
              "only the current WxH+offsets are allowed (reposition is not "
              "supported)\n",
              frame_index);
      return false;
    }
    if (static_cast<int64_t>(display_x) == cur_dx &&
        static_cast<int64_t>(display_y) == cur_dy) {
      return true;
    }
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " covers the full canvas; "
            "cannot change origin\n",
            frame_index);
    return false;
  }

  if (static_cast<uint64_t>(display_w) != cur_dw ||
      static_cast<uint64_t>(display_h) != cur_dh) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " display size must stay "
            "%llu x %llu (got %u x %u)\n",
            frame_index,
            static_cast<unsigned long long>(cur_dw),
            static_cast<unsigned long long>(cur_dh),
            static_cast<unsigned>(display_w), static_cast<unsigned>(display_h));
    return false;
  }

  int32_t nsx = 0;
  int32_t nsy = 0;
  if (!DisplayAabbMinToStorageCropOrigin(
          orient, canvas_w, canvas_h, static_cast<int64_t>(display_x),
          static_cast<int64_t>(display_y), rw, rh, &nsx, &nsy)) {
    return false;
  }
  if (nsx == static_cast<int32_t>(sx_old) && nsy == static_cast<int32_t>(sy_old)) {
    return true;
  }

  if (out_changed) *out_changed = true;
  fh->all_default = false;
  fh->have_crop = true;
  fh->ux0 = PackSigned(nsx);
  fh->uy0 = PackSigned(nsy);
  fh->crop_width = static_cast<uint32_t>(rw);
  fh->crop_height = static_cast<uint32_t>(rh);
  ClearRedundantFullCanvasCrop(fh, canvas_w, canvas_h);
  return true;
}

namespace {

constexpr float kGabDef1 = 1.1f * 0.104699568f;
constexpr float kGabDef2 = 1.1f * 0.055680538f;
// Bounds on s = w1+w2 per XYB channel (decoder neighbor kernel shape). Default
// sum is kGabSumDef (~0.176). Sharpen/blur shift the same amount in logit(s)
// space so equal --gab-blur=A and --gab-sharpen=A cancel; sharpen amount 1
// from the default sum moves s halfway toward the min sum (max sharp).
constexpr float kGabSumMin = -0.4f;
constexpr float kGabSumMax = 2.0f;
constexpr float kGabSumSpan = kGabSumMax - kGabSumMin;
constexpr float kGabSumDef = kGabDef1 + kGabDef2;
constexpr float kGabLogitEpsilon = 1e-7f;
// Snap within this tolerance of implicit defaults to gab_custom=false (F16
// quantization + logit round-trip).
constexpr float kGabSnapToImplicitEps = 1.5e-4f;

bool GabPairOk(float w1, float w2) {
  return std::abs(1.0f + (w1 + w2) * 4.0f) >= 1e-8f;
}

bool GabRfOk(const ParsedRestorationFilter& rf) {
  if (!rf.gab || !rf.gab_custom) return true;
  return GabPairOk(rf.gab_x_weight1, rf.gab_x_weight2) &&
         GabPairOk(rf.gab_y_weight1, rf.gab_y_weight2) &&
         GabPairOk(rf.gab_b_weight1, rf.gab_b_weight2);
}

void SetStandardGab(ParsedRestorationFilter* rf, const float w1[3],
                    const float w2[3]) {
  rf->all_default = false;
  rf->gab = true;
  rf->gab_custom = true;
  rf->gab_x_weight1 = w1[0];
  rf->gab_x_weight2 = w2[0];
  rf->gab_y_weight1 = w1[1];
  rf->gab_y_weight2 = w2[1];
  rf->gab_b_weight1 = w1[2];
  rf->gab_b_weight2 = w2[2];
}

float GabClampSum(float s) {
  return std::min(std::max(s, kGabSumMin), kGabSumMax);
}

float GabSumToU(float s) {
  s = GabClampSum(s);
  float p = (s - kGabSumMin) / kGabSumSpan;
  p = std::min(std::max(p, kGabLogitEpsilon), 1.f - kGabLogitEpsilon);
  return std::log(p / (1.f - p));
}

float GabUToSum(float u) {
  const float p = 1.f / (1.f + std::exp(-u));
  return kGabSumMin + p * kGabSumSpan;
}

float GabUnitStepU() {
  static const float kStep = GabSumToU(kGabSumDef) -
                           GabSumToU((kGabSumDef + kGabSumMin) * 0.5f);
  return kStep;
}

void GabRescaleWeightsToSum(float w1, float w2, float s_new, float* o1,
                            float* o2) {
  const float s_old = w1 + w2;
  if (std::abs(s_old) < 1e-20f) {
    const float inv = 1.f / kGabSumDef;
    *o1 = kGabDef1 * inv * s_new;
    *o2 = kGabDef2 * inv * s_new;
    return;
  }
  const float scale = s_new / s_old;
  *o1 = w1 * scale;
  *o2 = w2 * scale;
}

// If all six weights match implicit encoder defaults within tolerance, store as
// gab on + gab_custom off (avoids F16 / logit drift breaking blur↔sharpen).
void MaybeSnapGabToImplicitDefault(ParsedRestorationFilter* rf) {
  if (!rf->gab || !rf->gab_custom) return;
  const auto near = [](float a, float b) {
    return std::abs(a - b) <= kGabSnapToImplicitEps;
  };
  if (near(rf->gab_x_weight1, kGabDef1) && near(rf->gab_x_weight2, kGabDef2) &&
      near(rf->gab_y_weight1, kGabDef1) && near(rf->gab_y_weight2, kGabDef2) &&
      near(rf->gab_b_weight1, kGabDef1) && near(rf->gab_b_weight2, kGabDef2)) {
    rf->gab_custom = false;
  }
}

// Decoder-effective neighbor weights: gab off → zeros; gab on implicit default
// → libjxl defaults (kGabDef*); custom → stored values.
void EffectiveGabWeights(const ParsedRestorationFilter& rf, float w1_out[3],
                         float w2_out[3]) {
  if (!rf.gab) {
    for (int c = 0; c < 3; ++c) {
      w1_out[c] = 0.f;
      w2_out[c] = 0.f;
    }
    return;
  }
  if (!rf.gab_custom) {
    for (int c = 0; c < 3; ++c) {
      w1_out[c] = kGabDef1;
      w2_out[c] = kGabDef2;
    }
    return;
  }
  w1_out[0] = rf.gab_x_weight1;
  w2_out[0] = rf.gab_x_weight2;
  w1_out[1] = rf.gab_y_weight1;
  w2_out[1] = rf.gab_y_weight2;
  w1_out[2] = rf.gab_b_weight1;
  w2_out[2] = rf.gab_b_weight2;
}

}  // namespace

bool ApplyGabArgs(ParsedCodestream* cs, const GabArgs& args,
                  const std::vector<size_t>* only_frames) {
  if (args.kind == GabArgs::Kind::kNone) return true;
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      // Only the all_default bit was read; loop-filter fields were not parsed.
      // Expand to the same decoder defaults as GabEffectiveWeights6 / --info
      // gab_effective (DefaultRestorationFilter: gab on, implicit weights).
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    if (args.kind == GabArgs::Kind::kCustom) {
      rf->all_default = false;
      rf->gab = true;
      rf->gab_custom = true;
      rf->gab_x_weight1 = args.custom[0];
      rf->gab_x_weight2 = args.custom[1];
      rf->gab_y_weight1 = args.custom[2];
      rf->gab_y_weight2 = args.custom[3];
      rf->gab_b_weight1 = args.custom[4];
      rf->gab_b_weight2 = args.custom[5];
    } else if (args.kind == GabArgs::Kind::kBlur ||
               args.kind == GabArgs::Kind::kSharpen) {
      float cur_w1[3], cur_w2[3];
      EffectiveGabWeights(*rf, cur_w1, cur_w2);
      const float step_u = GabUnitStepU();
      const float du =
          args.amount * step_u *
          ((args.kind == GabArgs::Kind::kBlur) ? 1.f : -1.f);
      float w1[3];
      float w2[3];
      for (int c = 0; c < 3; ++c) {
        const float s = cur_w1[c] + cur_w2[c];
        float u = GabSumToU(s);
        u += du;
        const float s_new = GabClampSum(GabUToSum(u));
        GabRescaleWeightsToSum(cur_w1[c], cur_w2[c], s_new, &w1[c], &w2[c]);
      }
      SetStandardGab(rf, w1, w2);
    }
    MaybeSnapGabToImplicitDefault(rf);
    if (!GabRfOk(*rf)) {
      fprintf(stderr,
              "jxltran: Gaborish kernel is near-singular: require "
              "|1+4*(weight1+weight2)| >= 1e-8 per XYB channel (same as libjxl "
              "decode). Reduce --gab-sharpen / --gab-blur or adjust "
              "--gab-weights.\n");
      return false;
    }
    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: no regular or skip-progressive frames to edit (only LF "
            "or reference-only frames).\n");
    return false;
  }
  return true;
}

bool ApplyEpfIters(ParsedCodestream* cs, uint32_t epf_iters,
                   const std::vector<size_t>* only_frames) {
  epf_iters &= 3u;
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    rf->all_default = false;
    rf->epf_iters = epf_iters;
    if (fh->encoding == kFrameEncModular && epf_iters > 0) {
      if (rf->epf_sigma_for_modular < 1e-8f) {
        rf->epf_sigma_for_modular = 1.f;
      }
    }
    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: no regular or skip-progressive frames to edit (only LF "
            "or reference-only frames).\n");
    return false;
  }
  return true;
}

// lib/jxl/loop_filter.cc VisitFields defaults for the EPF sigma bundle.
constexpr float kEpfQuantMulImplicit = 0.46f;
constexpr float kEpfSigmaModularImplicit = 1.0f;
constexpr float kEpfPass0SigmaDefault = 0.9f;
constexpr float kEpfPass2SigmaDefault = 6.5f;
constexpr float kEpfBorderSadDefault = 2.f / 3.f;
constexpr float kEpfAmpSnapEps = 1.5e-4f;

void MaybeSnapEpfAmplitudeDefaults(bool modular, ParsedRestorationFilter* rf) {
  if (rf->epf_iters == 0) return;
  if (modular) {
    if (std::abs(rf->epf_sigma_for_modular - kEpfSigmaModularImplicit) <=
        kEpfAmpSnapEps) {
      rf->epf_sigma_for_modular = kEpfSigmaModularImplicit;
    }
    if (!rf->epf_sigma_custom) return;
    const auto near = [](float a, float b) {
      return std::abs(a - b) <= kEpfAmpSnapEps;
    };
    if (near(rf->epf_pass0_sigma_scale, kEpfPass0SigmaDefault) &&
        near(rf->epf_pass2_sigma_scale, kEpfPass2SigmaDefault) &&
        near(rf->epf_border_sad_mul, kEpfBorderSadDefault) &&
        near(rf->epf_sigma_for_modular, kEpfSigmaModularImplicit)) {
      rf->epf_sigma_custom = false;
    }
    return;
  }
  if (!rf->epf_sigma_custom) return;
  const auto near = [](float a, float b) {
    return std::abs(a - b) <= kEpfAmpSnapEps;
  };
  if (near(rf->epf_quant_mul, kEpfQuantMulImplicit) &&
      near(rf->epf_pass0_sigma_scale, kEpfPass0SigmaDefault) &&
      near(rf->epf_pass2_sigma_scale, kEpfPass2SigmaDefault) &&
      near(rf->epf_border_sad_mul, kEpfBorderSadDefault)) {
    rf->epf_sigma_custom = false;
  }
}

bool ApplyEpfAmplitudeScale(ParsedCodestream* cs, float factor,
                            const std::vector<size_t>* only_frames) {
  if (!(factor > 0.f) || !std::isfinite(factor)) {
    fprintf(stderr,
            "jxltran: --set-epf-amplitude-scale expects a positive finite "
            "float\n");
    return false;
  }
  if (factor == 1.f) return true;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    const uint32_t iters = std::min(3u, rf->epf_iters);
    if (iters == 0) continue;

    const bool modular = (fh->encoding == kFrameEncModular);
    rf->all_default = false;

    if (modular) {
      const float s = rf->epf_sigma_for_modular * factor;
      if (!(s > 1e-8f) || !std::isfinite(s)) {
        fprintf(stderr,
                "jxltran: --set-epf-amplitude-scale produced an invalid "
                "epf_sigma_for_modular for frame %" PRIuS "\n",
                fi);
        return false;
      }
      rf->epf_sigma_for_modular = s;
      MaybeSnapEpfAmplitudeDefaults(true, rf);
    } else {
      const float base =
          rf->epf_sigma_custom ? rf->epf_quant_mul : kEpfQuantMulImplicit;
      const float new_q = base * factor;
      if (!(new_q > 0.f) || !std::isfinite(new_q)) {
        fprintf(stderr,
                "jxltran: --set-epf-amplitude-scale produced an invalid "
                "epf_quant_mul for frame %" PRIuS "\n",
                fi);
        return false;
      }
      if (!rf->epf_sigma_custom) {
        rf->epf_pass0_sigma_scale = kEpfPass0SigmaDefault;
        rf->epf_pass2_sigma_scale = kEpfPass2SigmaDefault;
        rf->epf_border_sad_mul = kEpfBorderSadDefault;
      }
      rf->epf_sigma_custom = true;
      rf->epf_quant_mul = new_q;
      MaybeSnapEpfAmplitudeDefaults(false, rf);
    }

    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: --set-epf-amplitude-scale: no frames with EPF enabled "
            "(epf_iters==0 on all edited frames, or only LF / "
            "reference-only).\n");
    return false;
  }
  return true;
}

bool EpfEffectiveAmplitudeForInfo(bool modular_encoding,
                                  bool restoration_all_default,
                                  const ParsedRestorationFilter& rf,
                                  float* out_amp) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(modular_encoding);
  }
  const uint32_t it = std::min(3u, eff.epf_iters);
  if (it == 0) return false;
  if (modular_encoding) {
    *out_amp = eff.epf_sigma_for_modular;
  } else if (!eff.epf_sigma_custom) {
    *out_amp = kEpfQuantMulImplicit;
  } else {
    *out_amp = eff.epf_quant_mul;
  }
  return true;
}

static void FillEpfSharpLutMixed(float uniformity,
                                 std::array<float, kEpfSharpEntries>* out) {
  for (int i = 0; i < kEpfSharpEntries; ++i) {
    const float d = static_cast<float>(i) / 7.f;
    (*out)[static_cast<size_t>(i)] = (1.f - uniformity) * d + uniformity;
  }
}

static void MaybeSnapEpfSharpLutToImplicitRamp(ParsedRestorationFilter* rf) {
  if (!rf->epf_sharp_custom || rf->epf_iters == 0) return;
  for (int i = 0; i < kEpfSharpEntries; ++i) {
    const float d = static_cast<float>(i) / 7.f;
    if (std::abs(rf->epf_sharp_lut[static_cast<size_t>(i)] - d) > 1.5e-4f) {
      return;
    }
  }
  rf->epf_sharp_custom = false;
}

bool ApplyEpfSharpUniformity(ParsedCodestream* cs, float uniformity,
                             const std::vector<size_t>* only_frames,
                             bool* out_changed) {
  bool changed = false;
  if (out_changed != nullptr) *out_changed = false;
  if (!std::isfinite(uniformity) || uniformity < 0.f || uniformity > 1.f) {
    fprintf(stderr,
            "jxltran: --set-epf-uniformity expects a float in [0,1]\n");
    return false;
  }
  constexpr float kEps = 1e-6f;
  float u = uniformity;
  if (u <= kEps) u = 0.f;
  if (u + kEps >= 1.f) u = 1.f;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  size_t eligible = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->encoding == kFrameEncModular) {
      continue;
    }
    if (fh->restoration.all_default) {
      fh->restoration = DefaultRestorationFilter(false);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    const uint32_t iters = std::min(3u, rf->epf_iters);
    if (iters == 0) continue;
    ++eligible;

    rf->all_default = false;
    if (u == 0.f) {
      if (rf->epf_sharp_custom) {
        rf->epf_sharp_custom = false;
        fh->rf_reencode = true;
        changed = true;
      }
    } else {
      rf->epf_sharp_custom = true;
      FillEpfSharpLutMixed(u, &rf->epf_sharp_lut);
      MaybeSnapEpfSharpLutToImplicitRamp(rf);
      fh->rf_reencode = true;
      changed = true;
    }
  }

  if (out_changed != nullptr) *out_changed = changed;

  if (u > 0.f && eligible == 0) {
    fprintf(stderr,
            "jxltran: --set-epf-uniformity: no VarDCT frames with EPF "
            "(epf_iters==0 on all edited VarDCT frames, or only modular / LF / "
            "reference-only).\n");
    return false;
  }
  return true;
}

bool ApplyRestorationFilterExactRestore(ParsedCodestream* cs, size_t frame_index,
                                        const ParsedRestorationFilter& rf) {
  if (frame_index >= cs->frames.size()) return false;
  FramedUnit& fu = cs->frames[frame_index];
  const uint32_t ft = fu.frame.frame_type;
  if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) return false;
  fu.frame.restoration = rf;
  fu.frame.rf_reencode = true;
  return true;
}

float EpfSharpUniformityForInfo(bool restoration_all_default,
                                const ParsedRestorationFilter& rf) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(false);
  }
  if (eff.epf_iters == 0) return 0.f;
  if (!eff.epf_sharp_custom) return 0.f;
  const float t = eff.epf_sharp_lut[0];
  if (t < 0.f) return 0.f;
  if (t > 1.f) return 1.f;
  return t;
}

namespace {

bool LogicalSection0BodyByteOffset(const FramedUnit& fu, size_t* out_bytes) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *out_bytes = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      size_t off = 0;
      for (size_t q = 0; q < p; ++q) {
        if (q >= fu.toc_decoded_sizes.size()) return false;
        off += fu.toc_decoded_sizes[q];
      }
      *out_bytes = off;
      return true;
    }
  }
  return false;
}

// Stream-order TOC slot s carries logical section perm[s]; lf_global is logical 0.
static bool PhysicalStreamIndexForLogical0(const FramedUnit& fu, size_t* p0) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *p0 = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      *p0 = p;
      return true;
    }
  }
  return false;
}

static bool LfGlobalSection0BitLength(const FramedUnit& fu, size_t* bits) {
  if (fu.toc_decoded_sizes.empty()) return false;
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0)) return false;
  *bits = static_cast<size_t>(fu.toc_decoded_sizes[p0]) * 8u;
  return true;
}

// Bit length of LF-global physical chunk0 after a spline splice — must match
// WriteCodestream's |c0_payload_bits| (simple TOC path, no permutation).
static bool SplineEditLfChunk0SplicePayloadBits(const FramedUnit& fu,
                                               size_t* payload_bits_out) {
  const size_t frame_bit_base = fu.original_frame_byte_offset * 8;
  const size_t a0 = frame_bit_base + fu.toc_start_bit + fu.toc_bit_length;
  const int64_t delta_bits = fu.spline_edit_delta_bits;
  if (delta_bits % 8 != 0) {
    return false;
  }
  const int64_t lf_chunk0_extra_bits =
      (fu.photon_noise_edit
           ? static_cast<int64_t>(fu.photon_noise_delta_bytes) * 8
           : 0) +
      (fu.lf_global_dc_quant_edit
           ? static_cast<int64_t>(fu.lf_global_dc_quant_delta_bytes) * 8
           : 0) +
      (fu.lf_global_quantizer_edit
           ? static_cast<int64_t>(fu.lf_global_quantizer_delta_bytes) * 8
           : 0);
  const size_t old_body_bits = static_cast<size_t>(
      static_cast<int64_t>(fu.body_bit_length) - delta_bits - lf_chunk0_extra_bits);
  const size_t body_end_orig = a0 + old_body_bits;
  const size_t s0 = fu.lf_global_spline_region_abs_start_bit;
  const size_t s1 = fu.lf_global_spline_region_abs_end_bit;
  if (s0 < a0 || s1 < s0 || s1 > body_end_orig) {
    return false;
  }
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0) || p0 >= fu.toc_decoded_sizes.size()) {
    return false;
  }
  const int64_t dbytes = delta_bits / 8;
  const int64_t lf_extra_db =
      (fu.photon_noise_edit ? static_cast<int64_t>(fu.photon_noise_delta_bytes)
                            : 0) +
      (fu.lf_global_dc_quant_edit
           ? static_cast<int64_t>(fu.lf_global_dc_quant_delta_bytes)
           : 0) +
      (fu.lf_global_quantizer_edit
           ? static_cast<int64_t>(fu.lf_global_quantizer_delta_bytes)
           : 0);
  const int64_t new_c0_bytes = static_cast<int64_t>(fu.toc_decoded_sizes[p0]);
  const int64_t old_c0_bytes_i = new_c0_bytes - dbytes - lf_extra_db;
  if (old_c0_bytes_i < 0 ||
      old_c0_bytes_i > static_cast<int64_t>(UINT32_MAX)) {
    return false;
  }
  const size_t chunk0_end_abs =
      a0 + static_cast<size_t>(static_cast<uint32_t>(old_c0_bytes_i)) * 8u;
  if (s1 > chunk0_end_abs) {
    return false;
  }
  *payload_bits_out =
      (s0 - a0) + fu.spline_edit_new_mid_bits + (chunk0_end_abs - s1);
  return true;
}

// If TOC chunk0 is smaller than the splice payload, grow delta / TOC / body in
// lockstep (whole bytes) so WriteCodestream will not fail with
// "new LF-global section too small". Only for the non-permuted TOC spline path.
static bool EnsureSplineEditLfChunk0Budget(FramedUnit* fu) {
  if (fu->toc_perm.empty() == false) {
    return true;
  }
  size_t payload = 0;
  if (!SplineEditLfChunk0SplicePayloadBits(*fu, &payload)) {
    return false;
  }
  size_t budget = 0;
  if (!LfGlobalSection0BitLength(*fu, &budget)) {
    return true;
  }
  if (budget >= payload) {
    return true;
  }
  const size_t deficit = payload - budget;
  const int64_t add_bits = static_cast<int64_t>(((deficit + 7u) / 8u) * 8u);
  fu->spline_edit_delta_bits += add_bits;
  fu->body_bit_length = static_cast<size_t>(
      static_cast<int64_t>(fu->body_bit_length) + add_bits);
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(*fu, &p0)) {
    return false;
  }
  const int64_t nb =
      static_cast<int64_t>(fu->toc_decoded_sizes[p0]) + add_bits / 8;
  if (nb < 0 || nb > static_cast<int64_t>(UINT32_MAX)) {
    return false;
  }
  fu->toc_decoded_sizes[p0] = static_cast<uint32_t>(nb);
  return true;
}

enum class PhotonNoiseFrameOp { kRemove, kReplaceBytes, kInsertBytes };

// |insert_allow_empty_lut|: when true, skips NoiseLutHasAny before insert (lossless
// undo restoring verbatim prior bytes).
static bool PhotonNoiseApplyToFrame(ParsedCodestream* cs, size_t fi,
                                    PhotonNoiseFrameOp op,
                                    const std::array<uint8_t, 10>& bytes,
                                    bool insert_allow_empty_lut) {
  FramedUnit& fu = cs->frames[fi];
  fu.photon_noise_edit = false;
  fu.photon_noise_delta_bytes = 0;
  fu.photon_noise_patch_abs_bit = 0;
  fu.photon_noise_new_bytes.fill(0);
  fu.photon_noise_toc_reencode = false;

  FrameHeader* fh = &fu.frame;
  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;
  const uint32_t ft = fh->frame_type;
  if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
    return true;
  }
  if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
    return true;
  }

  size_t sec0_bytes = 0;
  if (!LogicalSection0BodyByteOffset(fu, &sec0_bytes)) {
    fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
    return false;
  }

  const size_t xsize_px = fh->have_crop ? fh->crop_width : canvas_w;
  const size_t ysize_px = fh->have_crop ? fh->crop_height : canvas_h;
  if (xsize_px == 0 || ysize_px == 0) {
    fprintf(stderr, "jxltran: photon noise: invalid frame dimensions\n");
    return false;
  }

  const size_t frame_bit0 = fu.original_frame_byte_offset * 8;
  const size_t body0 = frame_bit0 + fu.toc_start_bit + fu.toc_bit_length;

  size_t patch_abs = body0 + sec0_bytes * 8;
  if (fu.lf_global_noise_lut_abs_valid) {
    patch_abs = fu.lf_global_noise_lut_abs_bit;
  } else if ((fh->flags & kFrameFlagPatches) != 0 ||
             (fh->flags & kFrameFlagSplines) != 0) {
    fprintf(stderr,
            "jxltran: photon noise: could not parse DC-global prefix "
            "(patches/splines); noise LUT position is unknown.\n");
    return false;
  }

  const bool had_noise = (fh->flags & kFrameFlagNoise) != 0;

  size_t s0_bits = 0;
  if (!LfGlobalSection0BitLength(fu, &s0_bits)) {
    fprintf(stderr, "jxltran: photon noise: missing TOC sizes for DC global\n");
    return false;
  }
  const size_t noise_rel =
      fu.lf_global_noise_lut_abs_valid ? fu.lf_global_noise_lut_rel_bit : 0;

  if (op == PhotonNoiseFrameOp::kRemove) {
    if (!had_noise) {
      return true;
    }
    if (noise_rel + 80 > s0_bits) {
      fprintf(stderr,
              "jxltran: photon noise: noise LUT extends past DC-global section\n");
      return false;
    }
    fh->flags &= ~kFrameFlagNoise;
    fh->all_default = false;
    fu.photon_noise_edit = true;
    fu.photon_noise_delta_bytes = -10;
    fu.photon_noise_patch_abs_bit = patch_abs;
    fu.photon_noise_toc_reencode = true;
    if (!fu.toc_decoded_sizes.empty()) {
      size_t p0 = 0;
      if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
        fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
        return false;
      }
      const uint32_t s0 = fu.toc_decoded_sizes[p0];
      if (s0 < 10u) {
        fprintf(stderr,
                "jxltran: photon noise: DC global section too small to remove "
                "noise\n");
        return false;
      }
      fu.toc_decoded_sizes[p0] = s0 - 10u;
    }
    fu.body_bit_length -= 80;
    return true;
  }

  std::array<float, kNoiseLutPoints> dec{};
  if (!DecodeNoiseLutBits(bytes, &dec)) {
    fprintf(stderr, "jxltran: photon noise: invalid 80-bit LUT encoding\n");
    return false;
  }
  if (op == PhotonNoiseFrameOp::kInsertBytes) {
    if (!insert_allow_empty_lut && !NoiseLutHasAny(dec)) {
      return true;
    }
  } else if (op == PhotonNoiseFrameOp::kReplaceBytes) {
    if (!NoiseLutHasAny(dec)) {
      return PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kRemove, bytes,
                                     false);
    }
  }

  fu.photon_noise_new_bytes = bytes;

  if (op == PhotonNoiseFrameOp::kReplaceBytes) {
    if (!had_noise) {
      fprintf(stderr,
              "jxltran: photon noise: internal error (replace without prior noise)\n");
      return false;
    }
    if (noise_rel + 80 > s0_bits) {
      fprintf(stderr,
              "jxltran: photon noise: noise LUT extends past DC-global section\n");
      return false;
    }
    fh->flags |= kFrameFlagNoise;
    fh->all_default = false;
    fu.photon_noise_edit = true;
    fu.photon_noise_delta_bytes = 0;
    fu.photon_noise_patch_abs_bit = patch_abs;
    fu.photon_noise_toc_reencode = false;
    return true;
  }

  // kInsertBytes
  if (had_noise) {
    fprintf(stderr,
            "jxltran: photon noise: internal error (insert while noise flag set)\n");
    return false;
  }
  if (noise_rel > s0_bits) {
    fprintf(stderr,
            "jxltran: photon noise: noise insert point past DC-global section\n");
    return false;
  }

  fh->flags |= kFrameFlagNoise;
  fh->all_default = false;
  fu.photon_noise_edit = true;
  fu.photon_noise_delta_bytes = 10;
  fu.photon_noise_patch_abs_bit = patch_abs;
  fu.photon_noise_toc_reencode = true;
  if (!fu.toc_decoded_sizes.empty()) {
    size_t p0 = 0;
    if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
      fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
      return false;
    }
    fu.toc_decoded_sizes[p0] += 10u;
  }
  fu.body_bit_length += 80;
  return true;
}

static constexpr float kLfDcQuantWireDefault[3] = {1.0f / 32.0f, 1.0f / 4.0f,
                                                  1.0f / 2.0f};
// Decoder rejects non-positive LF dequant after ×1/128; keep a margin in wire
// space (pre-scale F16 domain).
static constexpr float kLfDcQuantWireMinPositive = 1e-12f;

static bool PackLfDcQuantNewRegion49(const float wire[3],
                                     std::vector<uint8_t>* out_bytes,
                                     size_t* out_bit_len) {
  BitWriter bw;
  bw.WriteBits(1, 0);
  for (int c = 0; c < 3; ++c) {
    bw.WriteF16(FloatToF16Bits(wire[c]));
  }
  *out_bit_len = bw.bit_pos();
  *out_bytes = bw.TakeBytes();
  return *out_bit_len == 49;
}

}  // namespace

bool ApplyPhotonNoiseIso(ParsedCodestream* cs, bool change, float iso,
                         const std::vector<size_t>* only_frames) {
  if (!change) return true;

  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    FrameHeader* fh = &fu.frame;
    const uint32_t ft = fh->frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
      continue;
    }

    size_t sec0_bytes = 0;
    if (!LogicalSection0BodyByteOffset(fu, &sec0_bytes)) {
      fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
      return false;
    }

    const size_t xsize_px = fh->have_crop ? fh->crop_width : canvas_w;
    const size_t ysize_px = fh->have_crop ? fh->crop_height : canvas_h;
    if (xsize_px == 0 || ysize_px == 0) {
      fprintf(stderr, "jxltran: photon noise: invalid frame dimensions\n");
      return false;
    }

    if (iso <= 0.f) {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kRemove,
                                   fu.photon_noise_new_bytes, false)) {
        return false;
      }
      continue;
    }

    std::array<float, kNoiseLutPoints> lut{};
    SimulatePhotonNoise(xsize_px, ysize_px, iso, &lut);
    if (!NoiseLutHasAny(lut)) {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kRemove,
                                   fu.photon_noise_new_bytes, false)) {
        return false;
      }
      continue;
    }

    std::array<uint8_t, 10> enc{};
    EncodeNoiseLutBits(lut, &enc);
    const bool had_noise = (fh->flags & kFrameFlagNoise) != 0;
    if (had_noise) {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kReplaceBytes, enc,
                                   false)) {
        return false;
      }
    } else {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kInsertBytes, enc,
                                   false)) {
        return false;
      }
    }
  }

  return true;
}

bool ApplyPhotonNoiseUndo(ParsedCodestream* cs, const PhotonNoiseUndoSpec& spec) {
  if (spec.frame_index >= cs->frames.size()) return false;
  switch (spec.forward_kind) {
    case PhotonNoiseForwardKind::kInsert: {
      std::array<uint8_t, 10> dummy{};
      return PhotonNoiseApplyToFrame(cs, spec.frame_index,
                                     PhotonNoiseFrameOp::kRemove, dummy, false);
    }
    case PhotonNoiseForwardKind::kRemove:
      return PhotonNoiseApplyToFrame(cs, spec.frame_index,
                                     PhotonNoiseFrameOp::kInsertBytes,
                                     spec.prior_bytes, true);
    case PhotonNoiseForwardKind::kReplace:
      return PhotonNoiseApplyToFrame(cs, spec.frame_index,
                                     PhotonNoiseFrameOp::kReplaceBytes,
                                     spec.prior_bytes, false);
    default:
      return false;
  }
}

static bool PhotonWeightsSpecIsHex20(const char* s) {
  if (s == nullptr) return false;
  size_t n = 0;
  for (const char* p = s; *p; ++p) {
    if (*p == ' ' || *p == '\t') continue;
    ++n;
  }
  if (n != 20) return false;
  for (const char* p = s; *p; ++p) {
    if (*p == ' ' || *p == '\t') continue;
    if (!std::isxdigit(static_cast<unsigned char>(*p))) return false;
  }
  return true;
}

static bool ParsePhotonWeightsHex20(const char* s,
                                    std::array<uint8_t, 10>* out) {
  int nibble = -1;
  size_t j = 0;
  for (const char* p = s; *p; ++p) {
    if (*p == ' ' || *p == '\t') continue;
    int v = -1;
    if (*p >= '0' && *p <= '9') v = *p - '0';
    else if (*p >= 'a' && *p <= 'f') v = 10 + (*p - 'a');
    else if (*p >= 'A' && *p <= 'F') v = 10 + (*p - 'A');
    else return false;
    if (nibble < 0) {
      nibble = v;
    } else {
      if (j >= 10) return false;
      (*out)[j++] = static_cast<uint8_t>((nibble << 4) | v);
      nibble = -1;
    }
  }
  if (nibble >= 0) return false;
  return j == 10;
}

bool ApplyPhotonNoiseWeights(ParsedCodestream* cs, bool change,
                             const char* weights_spec,
                             const std::vector<size_t>* only_frames) {
  if (!change) return true;
  if (weights_spec == nullptr || weights_spec[0] == '\0') {
    fprintf(stderr, "jxltran: --set-photon-noise-weights: empty value\n");
    return false;
  }

  std::array<uint8_t, 10> enc{};
  if (PhotonWeightsSpecIsHex20(weights_spec)) {
    if (!ParsePhotonWeightsHex20(weights_spec, &enc)) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-weights: invalid 20-hex-digit payload "
              "'%s'\n",
              weights_spec);
      return false;
    }
  } else {
    std::array<float, kNoiseLutPoints> lut{};
    const int n = std::sscanf(
        weights_spec, "%f,%f,%f,%f,%f,%f,%f,%f", &lut[0], &lut[1], &lut[2],
        &lut[3], &lut[4], &lut[5], &lut[6], &lut[7]);
    if (n != 8) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-weights expects eight comma-separated "
              "floats or 20 hex digits (verbatim 10-byte LUT), got '%s'\n",
              weights_spec);
      return false;
    }
    if (!NoiseLutHasAny(lut)) {
      const auto want_frame = [&](size_t idx) -> bool {
        if (only_frames == nullptr || only_frames->empty()) return true;
        return std::binary_search(only_frames->begin(), only_frames->end(),
                                  idx);
      };
      for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
        if (!want_frame(fi)) continue;
        const FrameHeader& fh = cs->frames[fi].frame;
        const uint32_t ft = fh.frame_type;
        if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
        if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
        if ((fh.flags & kFrameFlagNoise) == 0) continue;
        if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kRemove, enc,
                                     false)) {
          return false;
        }
      }
      return true;
    }
    EncodeNoiseLutBits(lut, &enc);
  }

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FrameHeader& fh = cs->frames[fi].frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;

    std::array<float, kNoiseLutPoints> dec{};
    if (!DecodeNoiseLutBits(enc, &dec)) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-weights: invalid LUT encoding\n");
      return false;
    }
    const bool had_noise = (fh.flags & kFrameFlagNoise) != 0;
    if (!NoiseLutHasAny(dec)) {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kRemove, enc,
                                   false)) {
        return false;
      }
      continue;
    }
    if (had_noise) {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kReplaceBytes, enc,
                                   false)) {
        return false;
      }
    } else {
      if (!PhotonNoiseApplyToFrame(cs, fi, PhotonNoiseFrameOp::kInsertBytes, enc,
                                   false)) {
        return false;
      }
    }
  }

  return true;
}

bool ApplyLfDcQuantMul(ParsedCodestream* cs, bool change, const float mul[3],
                       const std::vector<size_t>* only_frames) {
  if (!change) return true;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    fu.lf_global_dc_quant_edit = false;
    fu.lf_global_dc_quant_delta_bytes = 0;
    fu.lf_global_dc_quant_toc_reencode = false;
    fu.lf_global_dc_quant_new_region_bits.clear();
    fu.lf_global_dc_quant_new_region_len_bits = 0;

    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;

    if (!fu.lf_global_dc_quant_parsed) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: could not locate LF channel "
              "dequantization in frame %" PRIuS " (LF-global prefix parse failed "
              "or unsupported layout)\n",
              fi);
      return false;
    }
    if (fu.spline_edit) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: cannot combine with spline edits on "
              "frame %" PRIuS "\n",
              fi);
      return false;
    }
    if (fu.toc_strip_perm_reorder || !fu.toc_body_stream_shuffle.empty()) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: cannot combine with TOC strip / "
              "center-first shuffle on frame %" PRIuS "\n",
              fi);
      return false;
    }
    const size_t old_len = fu.lf_global_dc_quant_rel_region_old_len_bits;
    if (old_len != 1 && old_len != 49) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: unexpected LF dequant bundle length "
              "%" PRIuS " bits in frame %" PRIuS "\n",
              old_len, fi);
      return false;
    }

    bool any_change = false;
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(mul[c])) {
        fprintf(stderr,
                "jxltran: --lf-dc-quant-mul: each factor must be a finite float\n");
        return false;
      }
      if (mul[c] <= 0.f) {
        fprintf(stderr,
                "jxltran: --lf-dc-quant-mul: each factor must be positive (decoder "
                "rejects non-positive LF dequant); got non-positive on channel "
                "%d (frame %" PRIuS ")\n",
                c, fi);
        return false;
      }
      if (std::fabs(mul[c] - 1.0f) > 1e-5f) any_change = true;
    }
    if (!any_change) {
      continue;
    }

    float wire[3];
    if (fu.lf_global_dc_quant_all_default) {
      for (int c = 0; c < 3; ++c) {
        wire[c] = static_cast<float>(
            static_cast<double>(kLfDcQuantWireDefault[c]) *
            static_cast<double>(mul[c]));
      }
    } else {
      for (int c = 0; c < 3; ++c) {
        const double base = static_cast<double>(F16BitsToFloat(
            fu.lf_global_dc_quant_f16_bits[static_cast<size_t>(c)]));
        wire[c] = static_cast<float>(base * static_cast<double>(mul[c]));
      }
    }
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(wire[c]) || wire[c] < kLfDcQuantWireMinPositive) {
        fprintf(stderr,
                "jxltran: --lf-dc-quant-mul: resulting channel %d wire value is "
                "non-finite or not positive (frame %" PRIuS ")\n",
                c, fi);
        return false;
      }
      // Snapping: after F16×float scaling, float×inverse may land slightly off
      // the implicit encoder defaults; collapse when indistinguishable in wire
      // space so we can re-encode the compact 1-bit all-default bundle.
      const float d = kLfDcQuantWireDefault[c];
      if (d > 0.f && std::abs(mul[c] - 1.f) > 1e-6f) {
        const float ratio = wire[c] / d;
        if (std::abs(ratio - 1.f) < 2e-3f) wire[c] = d;
      }
    }

    std::vector<uint8_t> packed;
    size_t nbits = 0;
    const bool implicit_default_wire =
        FloatToF16Bits(wire[0]) == FloatToF16Bits(kLfDcQuantWireDefault[0]) &&
        FloatToF16Bits(wire[1]) == FloatToF16Bits(kLfDcQuantWireDefault[1]) &&
        FloatToF16Bits(wire[2]) == FloatToF16Bits(kLfDcQuantWireDefault[2]);
    if (implicit_default_wire) {
      // Match libjxl's compact bundle (ReadLfGlobalThroughNoise): a single 1
      // bit selects implicit defaults instead of 1 + 3×F16 explicit wire values.
      BitWriter bwb;
      bwb.WriteBits(1, 1);
      packed = bwb.TakeBytes();
      nbits = bwb.bit_pos();
    } else if (!PackLfDcQuantNewRegion49(wire, &packed, &nbits)) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: failed to encode LF dequant bundle "
              "(frame %" PRIuS ")\n",
              fi);
      return false;
    }

    const int64_t bit_delta = static_cast<int64_t>(nbits) -
                              static_cast<int64_t>(old_len);
    const int64_t delta_bytes = bit_delta / 8;
    if (bit_delta != delta_bytes * 8) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: internal bit delta error (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }

    size_t p0 = 0;
    if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: invalid TOC (frame %" PRIuS ")\n", fi);
      return false;
    }
    if (p0 >= fu.toc_decoded_sizes.size()) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: TOC index out of range (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    const int64_t new_sz =
        static_cast<int64_t>(fu.toc_decoded_sizes[p0]) + delta_bytes;
    if (new_sz < 0 || new_sz > static_cast<int64_t>(UINT32_MAX)) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: TOC section 0 size overflow (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    fu.toc_decoded_sizes[p0] = static_cast<uint32_t>(new_sz);
    const int64_t new_body =
        static_cast<int64_t>(fu.body_bit_length) + bit_delta;
    if (new_body < 0) {
      fprintf(stderr,
              "jxltran: --lf-dc-quant-mul: body length underflow (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    fu.body_bit_length = static_cast<size_t>(new_body);

    fu.lf_global_dc_quant_edit = true;
    fu.lf_global_dc_quant_delta_bytes = static_cast<int32_t>(delta_bytes);
    fu.lf_global_dc_quant_toc_reencode = (delta_bytes != 0);
    fu.lf_global_dc_quant_new_region_bits = std::move(packed);
    fu.lf_global_dc_quant_new_region_len_bits = nbits;
    fu.lf_global_dc_quant_all_default = implicit_default_wire;
    if (implicit_default_wire) {
      fu.lf_global_dc_quant_f16_bits.fill(0);
    } else {
      for (int c = 0; c < 3; ++c) {
        fu.lf_global_dc_quant_f16_bits[static_cast<size_t>(c)] =
            FloatToF16Bits(wire[c]);
      }
    }
  }

  return true;
}

// U32() allows multiple selector encodings for the same value. When the
// quantizer bundle length change is a multiple of 8, prefer that encoding so
// LF-global chunk0 needs no extra tail zero-bit padding.
static bool U32SelectorFits(uint32_t value, const U32Dist& d) {
  switch (d.kind) {
    case U32Dist::kImm:
      return value == d.p;
    case U32Dist::kBits:
      return (d.n >= 32u || value < (1u << d.n));
    case U32Dist::kBitsOffset:
      return (value >= d.p &&
              (d.n >= 32u || value - d.p < (1u << d.n)));
  }
  return false;
}

static size_t U32SelectorTotalBits(const U32Dist& d) {
  switch (d.kind) {
    case U32Dist::kImm:
      return 2;
    case U32Dist::kBits:
    case U32Dist::kBitsOffset:
      return 2u + static_cast<size_t>(d.n);
  }
  return 0;
}

static void U32WriteAtSelector(BitWriter& bw, uint32_t value, const U32Dist ds[4],
                               int sel) {
  const U32Dist& d = ds[sel];
  bw.WriteBits(2, static_cast<uint64_t>(sel));
  switch (d.kind) {
    case U32Dist::kImm:
      break;
    case U32Dist::kBits:
      bw.WriteBits(static_cast<int>(d.n), value);
      break;
    case U32Dist::kBitsOffset:
      bw.WriteBits(static_cast<int>(d.n), value - d.p);
      break;
  }
}

static bool PackQuantizerParamsBundleByteAligned(
    uint32_t global_scale, uint32_t quant_dc, size_t old_len_bits,
    std::vector<uint8_t>* out_bytes, size_t* out_bit_len) {
  static const U32Dist kGlobalScaleDists[4] = {
      U32Dist::BitsOffset(11, 1), U32Dist::BitsOffset(11, 2049),
      U32Dist::BitsOffset(12, 4097), U32Dist::BitsOffset(16, 8193)};
  static const U32Dist kQuantDcDists[4] = {
      U32Dist::Imm(16), U32Dist::BitsOffset(5, 1), U32Dist::BitsOffset(8, 1),
      U32Dist::BitsOffset(16, 1)};

  int best_sgs = -1;
  int best_sqc = -1;
  size_t best_bits = 0;
  int64_t best_abs_delta = -1;
  for (int sgs = 0; sgs < 4; ++sgs) {
    if (!U32SelectorFits(global_scale, kGlobalScaleDists[sgs])) continue;
    const size_t gs_bits = U32SelectorTotalBits(kGlobalScaleDists[sgs]);
    for (int sqc = 0; sqc < 4; ++sqc) {
      if (!U32SelectorFits(quant_dc, kQuantDcDists[sqc])) continue;
      const size_t bits = gs_bits + U32SelectorTotalBits(kQuantDcDists[sqc]);
      const int64_t bit_delta =
          static_cast<int64_t>(bits) - static_cast<int64_t>(old_len_bits);
      if (bit_delta % 8 != 0) continue;
      const int64_t ad = bit_delta >= 0 ? bit_delta : -bit_delta;
      const bool better =
          best_sgs < 0 || ad < best_abs_delta ||
          (ad == best_abs_delta && bits < best_bits);
      if (better) {
        best_sgs = sgs;
        best_sqc = sqc;
        best_bits = bits;
        best_abs_delta = ad;
      }
    }
  }
  if (best_sgs < 0) {
    return false;
  }

  BitWriter bw;
  U32WriteAtSelector(bw, global_scale, kGlobalScaleDists, best_sgs);
  U32WriteAtSelector(bw, quant_dc, kQuantDcDists, best_sqc);
  *out_bit_len = bw.bit_pos();
  *out_bytes = bw.TakeBytes();
  return true;
}

// Prefer byte-aligned bundle (no tail padding); otherwise canonical WriteU32
// first-fit encoding plus |tail_pad_bits| zero bits after chunk0 tail.
static void PackQuantizerParamsBundle(uint32_t global_scale, uint32_t quant_dc,
                                      size_t old_len_bits,
                                      std::vector<uint8_t>* out_bytes,
                                      size_t* out_bit_len, size_t* tail_pad_bits) {
  if (PackQuantizerParamsBundleByteAligned(global_scale, quant_dc, old_len_bits,
                                           out_bytes, out_bit_len)) {
    *tail_pad_bits = 0;
    return;
  }
  BitWriter bw;
  WriteU32(bw, global_scale, U32Dist::BitsOffset(11, 1),
           U32Dist::BitsOffset(11, 2049), U32Dist::BitsOffset(12, 4097),
           U32Dist::BitsOffset(16, 8193));
  WriteU32(bw, quant_dc, U32Dist::Imm(16), U32Dist::BitsOffset(5, 1),
           U32Dist::BitsOffset(8, 1), U32Dist::BitsOffset(16, 1));
  *out_bit_len = bw.bit_pos();
  *out_bytes = bw.TakeBytes();
  const int64_t d =
      static_cast<int64_t>(*out_bit_len) - static_cast<int64_t>(old_len_bits);
  int64_t r = d % 8;
  if (r < 0) r += 8;
  *tail_pad_bits = static_cast<size_t>((8 - r) % 8);
}

static bool PhysicalStreamIndexForLogical0Ops(const FramedUnit& fu,
                                              size_t* p0) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *p0 = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      *p0 = p;
      return true;
    }
  }
  return false;
}

bool ApplyLfGlobalScale(ParsedCodestream* cs, bool change, uint32_t global_scale,
                        const std::vector<size_t>* only_frames) {
  if (!change) return true;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    fu.lf_global_quantizer_edit = false;
    fu.lf_global_quantizer_delta_bytes = 0;
    fu.lf_global_quantizer_toc_reencode = false;
    fu.lf_global_quantizer_new_region_bits.clear();
    fu.lf_global_quantizer_new_region_len_bits = 0;
    fu.lf_global_quantizer_tail_pad_bits = 0;

    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    if (fh.encoding == kFrameEncModular) continue;

    if (!fu.lf_global_quantizer_parsed) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: VarDCT quantizer bundle not available "
              "in frame %" PRIuS " (LF-global prefix parse failed, modular frame, "
              "or unsupported layout)\n",
              fi);
      return false;
    }
    if (fu.spline_edit) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: cannot combine with spline edits on "
              "frame %" PRIuS "\n",
              fi);
      return false;
    }
    if (fu.toc_strip_perm_reorder || !fu.toc_body_stream_shuffle.empty()) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: cannot combine with TOC strip / "
              "center-first shuffle on frame %" PRIuS "\n",
              fi);
      return false;
    }
    if (global_scale == fu.lf_global_quantizer_global_scale) {
      continue;
    }
    if (global_scale < 1u || global_scale > 32768u) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: value %u out of range (use "
              "1..32768, frame %" PRIuS ")\n",
              static_cast<unsigned>(global_scale), fi);
      return false;
    }

    const size_t old_len = fu.lf_global_quantizer_rel_region_old_len_bits;
    if (old_len == 0) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: zero-length quantizer bundle in frame "
              "%" PRIuS "\n",
              fi);
      return false;
    }

    std::vector<uint8_t> packed;
    size_t nbits = 0;
    size_t tail_pad = 0;
    PackQuantizerParamsBundle(global_scale, fu.lf_global_quantizer_quant_dc,
                              old_len, &packed, &nbits, &tail_pad);
    if (tail_pad > 7u) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: internal tail padding error (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    fu.lf_global_quantizer_tail_pad_bits = static_cast<uint8_t>(tail_pad);

    const int64_t bit_delta = static_cast<int64_t>(nbits) -
                              static_cast<int64_t>(old_len) +
                              static_cast<int64_t>(tail_pad);
    const int64_t delta_bytes = bit_delta / 8;
    if (bit_delta != delta_bytes * 8) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: internal bit delta error (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }

    size_t p0 = 0;
    if (!PhysicalStreamIndexForLogical0Ops(fu, &p0)) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: invalid TOC (frame %" PRIuS ")\n", fi);
      return false;
    }
    if (p0 >= fu.toc_decoded_sizes.size()) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: TOC index out of range (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    const int64_t new_sz =
        static_cast<int64_t>(fu.toc_decoded_sizes[p0]) + delta_bytes;
    if (new_sz < 0 || new_sz > static_cast<int64_t>(UINT32_MAX)) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: TOC section 0 size overflow (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    fu.toc_decoded_sizes[p0] = static_cast<uint32_t>(new_sz);
    const int64_t new_body =
        static_cast<int64_t>(fu.body_bit_length) + bit_delta;
    if (new_body < 0) {
      fprintf(stderr,
              "jxltran: --lf-global-scale: body length underflow (frame %" PRIuS
              ")\n",
              fi);
      return false;
    }
    fu.body_bit_length = static_cast<size_t>(new_body);

    fu.lf_global_quantizer_edit = true;
    fu.lf_global_quantizer_delta_bytes = static_cast<int32_t>(delta_bytes);
    fu.lf_global_quantizer_toc_reencode = (delta_bytes != 0);
    fu.lf_global_quantizer_new_region_bits = std::move(packed);
    fu.lf_global_quantizer_new_region_len_bits = nbits;
    fu.lf_global_quantizer_global_scale = global_scale;
  }

  return true;
}

static bool ReadPathToString(const char* path, std::string* out) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) {
    fprintf(stderr, "jxltran: could not open '%s'\n", path);
    return false;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return false;
  }
  const long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return false;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return false;
  }
  out->assign(static_cast<size_t>(sz), '\0');
  if (sz > 0 &&
      fread(out->data(), 1, static_cast<size_t>(sz), f) !=
          static_cast<size_t>(sz)) {
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

bool ApplySplinesFromFile(ParsedCodestream* cs, const char* path,
                          const std::vector<size_t>* only_frames,
                          SplinesApplySummary* summary) {
  if (summary != nullptr) {
    summary->added_on_spline_free.clear();
    summary->replaced_existing_splines = false;
  }
  std::string text;
  if (!ReadPathToString(path, &text)) {
    return false;
  }
  std::vector<std::pair<size_t, LfGlobalSplines>> blocks;
  if (!SplinesParseText(text, &blocks)) {
    fprintf(stderr, "jxltran: failed to parse splines text file '%s'\n", path);
    return false;
  }

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  size_t applied = 0;
  for (const auto& blk : blocks) {
    const size_t fi = blk.first;
    if (!want_frame(fi)) {
      continue;
    }
    if (fi >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-splines-from: frame %" PRIuS " is out of range\n",
              fi);
      return false;
    }
    FramedUnit& fu = cs->frames[fi];
    if (!fu.lf_global_spline_region_valid) {
      fprintf(stderr,
              "jxltran: --set-splines-from: spline entropy insert/replace span "
              "unknown for frame %" PRIuS " (LF-global prefix not parsed)\n",
              fi);
      return false;
    }
    const bool had_splines = (fu.frame.flags & kFrameFlagSplines) != 0;
    if (had_splines) {
      if (summary != nullptr) {
        summary->replaced_existing_splines = true;
      }
      if (fu.lf_global_spline_region_abs_start_bit >=
          fu.lf_global_spline_region_abs_end_bit) {
        fprintf(stderr,
                "jxltran: --set-splines-from: frame %" PRIuS " has invalid spline "
                "entropy span\n",
                fi);
        return false;
      }
    } else {
      if (fu.lf_global_spline_region_abs_start_bit !=
          fu.lf_global_spline_region_abs_end_bit) {
        fprintf(stderr,
                "jxltran: --set-splines-from: frame %" PRIuS " spline insert point "
                "is inconsistent\n",
                fi);
        return false;
      }
      fu.frame.flags |= kFrameFlagSplines;
    }
    if (fu.toc_strip_perm_reorder) {
      fprintf(stderr,
              "jxltran: --set-splines-from: incompatible with stripped TOC "
              "permutation (frame %" PRIuS ")\n",
              fi);
      return false;
    }

    const uint32_t canvas_w = cs->image.size.width;
    const uint32_t canvas_h = cs->image.size.height;
    FrameTocMetrics m{};
    ComputeFrameTocMetrics(fu.frame, cs->image.metadata, canvas_w, canvas_h,
                          &m);
    const size_t num_pixels = m.xsize * m.ysize;
    LfGlobalSplines spl = blk.second;
    std::vector<uint8_t> enc;
    size_t enc_bits = 0;
    if (!EncodeSplinesBundleBits(spl, num_pixels, &enc, &enc_bits,
                                 /*pad_entropy_to_byte=*/false)) {
      fprintf(stderr,
              "jxltran: --set-splines-from: failed to re-encode splines for frame "
              "%" PRIuS " (hybrid-uint / entropy re-encoding failed)\n",
              fi);
      return false;
    }

    const int64_t old_span_bits = static_cast<int64_t>(
        fu.lf_global_spline_region_abs_end_bit -
        fu.lf_global_spline_region_abs_start_bit);
    const int64_t new_span_bits = static_cast<int64_t>(enc_bits);
    const int64_t delta_content = new_span_bits - old_span_bits;
    const int64_t old_body_bits =
        static_cast<int64_t>(fu.body_bit_length);
    const int64_t new_body_core = old_body_bits + delta_content;
    if (new_body_core < 0) {
      fprintf(stderr,
              "jxltran: --set-splines-from: frame %" PRIuS " spline edit would "
              "make negative body length\n",
              fi);
      return false;
    }
    // Byte-align the whole frame body: pad with 0..7 zero bits. We do not parse
    // the entire LF-global bundle, so trailing bits in the last body byte may
    // already have been encoder padding—we cannot subtract those. Worst case we
    // add ~one redundant byte on top of what a full decoder would need.
    int64_t phase_mod = new_body_core % 8;
    if (phase_mod < 0) {
      phase_mod += 8;
    }
    const int64_t body_tail_pad = (8 - phase_mod) % 8;
    const int64_t delta_bits = delta_content + body_tail_pad;
    fu.spline_edit = true;
    fu.spline_edit_new_mid_bytes = std::move(enc);
    fu.spline_edit_new_mid_bits = enc_bits;
    fu.spline_edit_delta_bits = delta_bits;
    fu.spline_edit_toc_reencode =
        !fu.toc_perm.empty() && fu.spline_edit_delta_bits != 0;
    fu.lf_global_splines = std::move(spl);

    fu.body_bit_length = static_cast<size_t>(
        static_cast<int64_t>(fu.body_bit_length) + delta_bits);
    if (!fu.toc_decoded_sizes.empty()) {
      size_t p0 = 0;
      if (fu.toc_perm.empty()) {
        p0 = 0;
      } else {
        bool found = false;
        for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
          if (fu.toc_perm[p] == 0) {
            p0 = p;
            found = true;
            break;
          }
        }
        if (!found) {
          fprintf(stderr, "jxltran: --set-splines-from: invalid TOC permutation\n");
          return false;
        }
      }
      const int64_t dbytes = delta_bits / 8;
      const int64_t ns0 =
          static_cast<int64_t>(fu.toc_decoded_sizes[p0]) + dbytes;
      if (ns0 < 0 || ns0 > UINT32_MAX) {
        fprintf(stderr,
                "jxltran: --set-splines-from: DC global section size invalid\n");
        return false;
      }
      fu.toc_decoded_sizes[p0] = static_cast<uint32_t>(ns0);
      if (!EnsureSplineEditLfChunk0Budget(&fu)) {
        fprintf(stderr,
                "jxltran: --set-splines-from: frame %" PRIuS " LF-global TOC budget "
                "does not match spline splice (internal)\n",
                fi);
        return false;
      }
    }
    if (summary != nullptr && !had_splines) {
      summary->added_on_spline_free.emplace_back(fi, fu.spline_edit_delta_bits);
    }
    ++applied;
  }

  if (applied == 0) {
    fprintf(stderr,
            "jxltran: --set-splines-from: no spline bundles applied (empty "
            "file, parse error, or no matching frame indices)\n");
    return false;
  }
  return true;
}

bool ApplyRemoveSplinesAddOnly(ParsedCodestream* cs, size_t fi,
                               const int64_t* exact_forward_body_delta_bits) {
  if (fi >= cs->frames.size()) {
    fprintf(stderr,
            "jxltran: --clear-splines-frames: frame %" PRIuS " is out of range\n",
            fi);
    return false;
  }
  FramedUnit& fu = cs->frames[fi];
  if (!fu.lf_global_spline_region_valid) {
    fprintf(stderr,
            "jxltran: --clear-splines-frames: spline entropy span unknown for frame "
            "%" PRIuS " (LF-global prefix not parsed)\n",
            fi);
    return false;
  }
  if (fu.toc_strip_perm_reorder) {
    fprintf(stderr,
            "jxltran: --clear-splines-frames: incompatible with stripped TOC "
            "permutation (frame %" PRIuS ")\n",
            fi);
    return false;
  }
  if ((fu.frame.flags & kFrameFlagSplines) == 0) {
    fprintf(stderr,
            "jxltran: --clear-splines-frames: frame %" PRIuS " has no splines flag\n",
            fi);
    return false;
  }
  const int64_t old_span_bits = static_cast<int64_t>(
      fu.lf_global_spline_region_abs_end_bit -
      fu.lf_global_spline_region_abs_start_bit);
  if (old_span_bits <= 0) {
    fprintf(stderr,
            "jxltran: --clear-splines-frames: frame %" PRIuS " has no spline entropy "
            "to remove\n",
            fi);
    return false;
  }
  int64_t delta_bits;
  if (exact_forward_body_delta_bits != nullptr) {
    const int64_t D = *exact_forward_body_delta_bits;
    if (D <= 0 || (D % 8) != 0) {
      fprintf(stderr,
              "jxltran: internal spline undo: invalid forward body delta (%lld) "
              "for frame %" PRIuS "\n",
              static_cast<long long>(D), fi);
      return false;
    }
    if (static_cast<int64_t>(fu.body_bit_length) < D) {
      fprintf(stderr,
              "jxltran: internal spline undo: body shorter than forward delta "
              "(%" PRIuS ")\n",
              fi);
      return false;
    }
    delta_bits = -D;
  } else {
    const int64_t new_span_bits = 0;
    const int64_t delta_content = new_span_bits - old_span_bits;
    const int64_t old_body_bits = static_cast<int64_t>(fu.body_bit_length);
    const int64_t new_body_core = old_body_bits + delta_content;
    if (new_body_core < 0) {
      fprintf(stderr,
              "jxltran: --clear-splines-frames: frame %" PRIuS " edit would make "
              "negative body length\n",
              fi);
      return false;
    }
    int64_t phase_mod = new_body_core % 8;
    if (phase_mod < 0) {
      phase_mod += 8;
    }
    const int64_t body_tail_pad = (8 - phase_mod) % 8;
    delta_bits = delta_content + body_tail_pad;
  }
  fu.spline_edit = true;
  fu.spline_edit_new_mid_bytes.clear();
  fu.spline_edit_new_mid_bits = 0;
  fu.spline_edit_delta_bits = delta_bits;
  fu.spline_edit_toc_reencode =
      !fu.toc_perm.empty() && fu.spline_edit_delta_bits != 0;
  fu.frame.flags &= ~static_cast<uint64_t>(kFrameFlagSplines);
  fu.lf_global_splines.reset();
  // Keep lf_global_spline_*_{start,end}_bit as parsed from the source codestream
  // (the spline entropy span in LF-global). WriteCodestream splices that span
  // out using empty replacement bits; collapsing end to start would make the
  // splice a no-op and leave spline entropy in the output.

  fu.body_bit_length = static_cast<size_t>(
      static_cast<int64_t>(fu.body_bit_length) + delta_bits);
  if (!fu.toc_decoded_sizes.empty()) {
    size_t p0 = 0;
    if (fu.toc_perm.empty()) {
      p0 = 0;
    } else {
      bool found = false;
      for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
        if (fu.toc_perm[p] == 0) {
          p0 = p;
          found = true;
          break;
        }
      }
      if (!found) {
        fprintf(stderr,
                "jxltran: --clear-splines-frames: invalid TOC permutation (frame "
                "%" PRIuS ")\n",
                fi);
        return false;
      }
    }
    const int64_t dbytes = delta_bits / 8;
    const int64_t ns0 =
        static_cast<int64_t>(fu.toc_decoded_sizes[p0]) + dbytes;
    if (ns0 < 0 || ns0 > UINT32_MAX) {
      fprintf(stderr,
              "jxltran: --clear-splines-frames: DC global section size invalid "
              "(frame %" PRIuS ")\n",
              fi);
      return false;
    }
    fu.toc_decoded_sizes[p0] = static_cast<uint32_t>(ns0);
    if (!EnsureSplineEditLfChunk0Budget(&fu)) {
      fprintf(stderr,
              "jxltran: spline undo: frame %" PRIuS " LF-global TOC budget does not "
              "match spline splice (internal)\n",
              fi);
      return false;
    }
  }
  return true;
}

bool ApplyClearSplinesOnFrames(ParsedCodestream* cs,
                               const std::vector<size_t>& frame_indices) {
  std::vector<size_t> uniq = frame_indices;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  for (size_t fi : uniq) {
    if (!ApplyRemoveSplinesAddOnly(cs, fi)) {
      return false;
    }
  }
  return true;
}

uint8_t ComputeLfGlobalChunk0TailPaddingTrimBytes(
    const uint8_t* long_cs, size_t long_sz, const ParsedCodestream& long_parsed,
    const uint8_t* ref_cs, size_t ref_sz, const ParsedCodestream& ref_parsed,
    size_t frame_index) {
  if (frame_index >= long_parsed.frames.size() ||
      frame_index >= ref_parsed.frames.size()) {
    return 0;
  }
  const FramedUnit& L = long_parsed.frames[frame_index];
  const FramedUnit& R = ref_parsed.frames[frame_index];
  if (L.toc_decoded_sizes.empty() || R.toc_decoded_sizes.empty()) {
    return 0;
  }
  std::vector<std::vector<uint8_t>> chunksL, chunksR;
  if (!ExtractFramedUnitBodyStreamChunks(long_cs, long_sz, L, &chunksL) ||
      !ExtractFramedUnitBodyStreamChunks(ref_cs, ref_sz, R, &chunksR)) {
    return 0;
  }
  size_t p0L = 0;
  size_t p0R = 0;
  if (!PhysicalStreamIndexForLogical0(L, &p0L) ||
      !PhysicalStreamIndexForLogical0(R, &p0R)) {
    return 0;
  }
  if (p0L >= chunksL.size() || p0R >= chunksR.size()) {
    return 0;
  }
  const std::vector<uint8_t>& cL = chunksL[p0L];
  const std::vector<uint8_t>& cR = chunksR[p0R];
  if (cL.size() < cR.size()) {
    return 0;
  }
  const size_t delta = cL.size() - cR.size();
  if (delta == 0 || delta > 255) {
    return 0;
  }
  if (memcmp(cL.data(), cR.data(), cR.size()) != 0) {
    return 0;
  }
  for (size_t i = cR.size(); i < cL.size(); ++i) {
    if (cL[i] != 0) {
      return 0;
    }
  }
  return static_cast<uint8_t>(delta);
}

bool ApplyLfGlobalPhysicalChunk0TailTrim(ParsedCodestream* cs, size_t fi,
                                         uint8_t trim_bytes) {
  if (trim_bytes == 0) {
    return true;
  }
  if (trim_bytes > 255) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: trim count %u is out of "
            "range (max 255)\n",
            static_cast<unsigned>(trim_bytes));
    return false;
  }
  if (fi >= cs->frames.size()) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " is out "
            "of range\n",
            fi);
    return false;
  }
  FramedUnit& fu = cs->frames[fi];
  const bool spline_clear_tail =
      fu.spline_edit && fu.spline_edit_new_mid_bits == 0;
  const bool quantizer_scale_tail = fu.lf_global_quantizer_edit;
  if (!spline_clear_tail && !quantizer_scale_tail) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " must "
            "follow --clear-splines-frames or --lf-global-scale on the same pass\n",
            fi);
    return false;
  }
  if (fu.toc_decoded_sizes.empty() || fu.toc_strip_perm_reorder) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " has no "
            "TOC sizes or uses stripped TOC permutation\n",
            fi);
    return false;
  }
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " has "
            "invalid TOC permutation\n",
            fi);
    return false;
  }
  if (fu.toc_decoded_sizes[p0] < trim_bytes) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " section "
            "0 too small to trim\n",
            fi);
    return false;
  }
  const int64_t bits = static_cast<int64_t>(trim_bytes) * 8;
  if (static_cast<int64_t>(fu.body_bit_length) < bits) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " body "
            "too small\n",
            fi);
    return false;
  }
  fu.toc_decoded_sizes[p0] =
      static_cast<uint32_t>(static_cast<int64_t>(fu.toc_decoded_sizes[p0]) -
                            static_cast<int64_t>(trim_bytes));
  fu.body_bit_length =
      static_cast<size_t>(static_cast<int64_t>(fu.body_bit_length) - bits);
  if (spline_clear_tail) {
    fu.spline_edit_delta_bits -= bits;
    if (!EnsureSplineEditLfChunk0Budget(&fu)) {
      return false;
    }
  } else {
    fu.lf_global_quantizer_delta_bytes -= static_cast<int32_t>(trim_bytes);
    if (fu.lf_global_quantizer_delta_bytes < 0) {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS " trim "
              "exceeds LF-global quantizer size delta\n",
              fi);
      return false;
    }
  }
  return true;
}

bool ApplyFrameBlendExactRestore(ParsedCodestream* cs, size_t frame_index,
                                 const FrameBlendingInfo& main,
                                 const std::vector<FrameBlendingInfo>& ec,
                                 uint32_t save_as_reference) {
  if (frame_index >= cs->frames.size()) {
    return false;
  }
  FrameHeader& fh = cs->frames[frame_index].frame;
  if (fh.frame_type != kFrameTypeRegular &&
      fh.frame_type != kFrameTypeSkipProgressive) {
    return false;
  }
  fh.blending_info = main;
  fh.ec_blending_info = ec;
  fh.save_as_reference = save_as_reference;
  return true;
}

void GabEffectiveWeights6(bool modular_encoding, bool restoration_all_default,
                          const ParsedRestorationFilter& rf, float out6[6]) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(modular_encoding);
  }
  float w1[3];
  float w2[3];
  EffectiveGabWeights(eff, w1, w2);
  out6[0] = w1[0];
  out6[1] = w2[0];
  out6[2] = w1[1];
  out6[3] = w2[1];
  out6[4] = w1[2];
  out6[5] = w2[2];
}

namespace {

void NormalizeBlendingInfoForMode(uint32_t mode, uint32_t num_extra,
                                  bool is_partial_frame, FrameBlendingInfo* bi) {
  bi->mode = mode;
  const bool has_alpha = num_extra > 0 && (mode == 2u || mode == 3u);
  const bool has_clamp =
      num_extra > 0 && (mode == 2u || mode == 3u || mode == 4u);
  if (!has_alpha) bi->alpha_channel = 0;
  if (!has_clamp) bi->clamp = false;
  if (mode == 0u && !is_partial_frame) {
    bi->source = 0;
  } else {
    bi->source &= 3u;
  }
}

}  // namespace

bool ApplyFrameBlendOverrides(ParsedCodestream* cs,
                              const std::vector<FrameBlendOverride>& overrides) {
  const uint32_t cw = cs->image.size.width;
  const uint32_t ch = cs->image.size.height;
  const uint32_t nx = cs->image.metadata.num_extra;
  for (const FrameBlendOverride& o : overrides) {
    if (o.frame_index >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame index %" PRIuS " out of range "
              "(codestream has %" PRIuS " frames)\n",
              o.frame_index, cs->frames.size());
      return false;
    }
    if (o.mode > 4u) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: blend mode %u invalid "
              "(expect 0–4)\n",
              static_cast<unsigned>(o.mode));
      return false;
    }
    FrameHeader& fh = cs->frames[o.frame_index].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              o.frame_index);
      return false;
    }
    const bool partial = FrameNeedsBlendingSourceField(fh, cw, ch);

    NormalizeBlendingInfoForMode(o.mode, nx, partial, &fh.blending_info);
    if (fh.ec_blending_info.size() != nx) {
      fh.ec_blending_info.assign(nx, FrameBlendingInfo{});
    }
    for (size_t e = 0; e < nx; ++e) {
      NormalizeBlendingInfoForMode(o.mode, nx, partial, &fh.ec_blending_info[e]);
    }

    auto apply_one = [&](FrameBlendingInfo* bi) {
      if (o.set_alpha_channel) {
        bi->alpha_channel = o.alpha_channel;
      }
      if (o.set_clamp) {
        bi->clamp = o.clamp;
      }
      if (o.set_source) {
        bi->source = o.source;
      }
      return true;
    };

    if (!apply_one(&fh.blending_info)) return false;
    for (size_t e = 0; e < nx; ++e) {
      if (!apply_one(&fh.ec_blending_info[e])) return false;
    }

    if (o.set_save_as_reference) {
      if (fh.is_last) {
        fprintf(stderr,
                "jxltran: --set-frame-blends: target= applies only when the "
                "frame is not the last in the codestream (is_last is false)\n");
        return false;
      }
      if (o.save_as_reference > 3u) {
        fprintf(stderr,
                "jxltran: --set-frame-blends: target= must be in range 0–3\n");
        return false;
      }
      fh.save_as_reference = o.save_as_reference & 3u;
    }
  }
  return true;
}

bool RemapFrameBlendOverridesForKeepOrder(
    const std::vector<size_t>& frames_in_order, const size_t num_frames_before,
    std::vector<FrameBlendOverride>* overrides) {
  if (overrides == nullptr) return false;
  std::vector<size_t> order;
  order.reserve(frames_in_order.size());
  for (size_t idx : frames_in_order) {
    if (idx >= num_frames_before) {
      fprintf(stderr,
              "jxltran: --set-frame-blends remap: keep-list index %" PRIuS " out "
              "of range (codestream had %" PRIuS " frames before keep)\n",
              idx, num_frames_before);
      return false;
    }
    bool seen = false;
    for (size_t prior : order) {
      if (prior == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) order.push_back(idx);
  }
  std::vector<size_t> old_to_new(num_frames_before,
                                 std::numeric_limits<size_t>::max());
  for (size_t j = 0; j < order.size(); ++j) {
    old_to_new[order[j]] = j;
  }
  for (FrameBlendOverride& o : *overrides) {
    if (o.frame_index >= num_frames_before) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame index %" PRIuS " out of range "
              "for keep remap (codestream had %" PRIuS " frames before keep)\n",
              o.frame_index, num_frames_before);
      return false;
    }
    const size_t ni = old_to_new[o.frame_index];
    if (ni == std::numeric_limits<size_t>::max()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame %" PRIuS " is not in the "
              "--keep-listed-frames keep list (cannot apply blend after "
              "reorder)\n",
              o.frame_index);
      return false;
    }
    o.frame_index = ni;
  }
  return true;
}

bool ApplyFrameDurationOverrides(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, uint32_t>>& frame_index_duration) {
  if (!cs->image.metadata.have_animation) {
    fprintf(stderr,
            "jxltran: --set-frame-durations requires an animated image "
            "(have_animation)\n");
    return false;
  }
  for (const auto& p : frame_index_duration) {
    if (p.first >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: frame index %" PRIuS " out of "
              "range (codestream has %" PRIuS " frames)\n",
              p.first, cs->frames.size());
      return false;
    }
    FrameHeader& fh = cs->frames[p.first].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              p.first);
      return false;
    }
    fh.duration = p.second;
  }
  return true;
}

namespace {

constexpr size_t kMaxFrameNameBytes = 4096;

}  // namespace

bool ApplySetFrameNames(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, std::vector<uint8_t>>>&
        frame_index_name) {
  for (const auto& p : frame_index_name) {
    if (p.first >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-names: frame index %" PRIuS " out of range "
              "(codestream has %" PRIuS " frames)\n",
              p.first, cs->frames.size());
      return false;
    }
    FrameHeader& fh = cs->frames[p.first].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-names: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              p.first);
      return false;
    }
    if (p.second.size() > kMaxFrameNameBytes) {
      fprintf(stderr,
              "jxltran: --set-frame-names: name for frame %" PRIuS " exceeds "
              "%" PRIuS " bytes\n",
              p.first, kMaxFrameNameBytes);
      return false;
    }
    fh.name = p.second;
  }
  return true;
}

bool ApplyKeepListedFrames(ParsedCodestream* cs,
                           const std::vector<size_t>& frames_in_order,
                           bool* out_changed, KeepReorderUndoSpec* undo_out) {
  if (undo_out) undo_out->clear();
  if (out_changed) *out_changed = false;
  if (frames_in_order.empty()) {
    fprintf(stderr,
            "jxltran: --keep-listed-frames requires a non-empty --keep-frames or "
            "--frames list\n");
    return false;
  }
  const size_t n_src_frames = cs->frames.size();
  std::vector<size_t> order;
  order.reserve(frames_in_order.size());
  for (size_t idx : frames_in_order) {
    if (idx >= n_src_frames) {
      fprintf(stderr,
              "jxltran: --frames/--keep-frames: index %" PRIuS " out of range "
              "(codestream has "
              "%" PRIuS " frames)\n",
              idx, n_src_frames);
      return false;
    }
    bool seen = false;
    for (size_t prior : order) {
      if (prior == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) order.push_back(idx);
  }

  bool identity_ordered = order.size() == n_src_frames;
  if (identity_ordered) {
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] != i) {
        identity_ordered = false;
        break;
      }
    }
  }

  std::vector<std::pair<bool, uint32_t>> tail_before;
  tail_before.reserve(n_src_frames);
  for (const FramedUnit& fu : cs->frames) {
    const FrameHeader& fh = fu.frame;
    if (fh.frame_type == kFrameTypeLF ||
        fh.frame_type == kFrameTypeReferenceOnly) {
      tail_before.emplace_back(false, 0);
    } else {
      tail_before.emplace_back(fh.is_last, fh.save_as_reference);
    }
  }

  if (undo_out && order.size() == n_src_frames) {
    std::vector<uint8_t> seen(n_src_frames, 0);
    bool bijective = true;
    for (size_t idx : order) {
      if (idx >= n_src_frames || seen[idx]) {
        bijective = false;
        break;
      }
      seen[idx] = 1;
    }
    if (bijective) {
      for (size_t i = 0; i < n_src_frames; ++i) {
        if (!seen[i]) {
          bijective = false;
          break;
        }
      }
    }
    if (bijective) {
      undo_out->forward_order = order;
      undo_out->tail_before = tail_before;
    }
  }

  std::vector<FramedUnit> src = std::move(cs->frames);
  std::vector<FramedUnit> out;
  out.reserve(order.size());
  for (size_t idx : order) {
    out.push_back(std::move(src[idx]));
  }
  cs->frames = std::move(out);

  const uint32_t last_ft = cs->frames.back().frame.frame_type;
  if (last_ft != kFrameTypeRegular && last_ft != kFrameTypeSkipProgressive) {
    fprintf(stderr,
            "jxltran: --keep-listed-frames: the last kept frame must be "
            "regular or skip-progressive (got frame_type=%u)\n",
            static_cast<unsigned>(last_ft));
    return false;
  }

  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  for (size_t i = 0; i < cs->frames.size(); ++i) {
    FrameHeader& fh = cs->frames[i].frame;
    if (fh.frame_type == kFrameTypeLF ||
        fh.frame_type == kFrameTypeReferenceOnly) {
      continue;
    }
    const bool last = (i + 1 == cs->frames.size());
    fh.is_last = last;
    if (last) {
      fh.save_as_reference = 0;
      continue;
    }
    const FrameHeader& nfh = cs->frames[i + 1].frame;
    uint32_t slot = fh.save_as_reference & 3u;
    if (nfh.frame_type == kFrameTypeRegular ||
        nfh.frame_type == kFrameTypeSkipProgressive) {
      const bool partial_next =
          FrameNeedsBlendingSourceField(nfh, canvas_w, canvas_h);
      if (nfh.blending_info.mode != 0u || partial_next) {
        slot = nfh.blending_info.source & 3u;
      }
    }
    fh.save_as_reference = slot;
    if (cs->image.metadata.have_animation && fh.duration != 0 &&
        fh.save_as_reference == 0) {
      fh.save_as_reference = 1u;
    }
  }

  if (out_changed) {
    if (!identity_ordered) {
      *out_changed = true;
    } else {
      bool diff = false;
      for (size_t j = 0; j < cs->frames.size(); ++j) {
        const FrameHeader& fh = cs->frames[j].frame;
        if (fh.frame_type == kFrameTypeLF ||
            fh.frame_type == kFrameTypeReferenceOnly) {
          continue;
        }
        const size_t src_i = order[j];
        if (src_i >= tail_before.size()) {
          diff = true;
          break;
        }
        if (fh.is_last != tail_before[src_i].first ||
            fh.save_as_reference != tail_before[src_i].second) {
          diff = true;
          break;
        }
      }
      *out_changed = diff;
    }
  }
  return true;
}

bool ApplyKeepListedFramesReorderInverse(ParsedCodestream* cs,
                                         const KeepReorderUndoSpec& undo) {
  if (undo.forward_order.empty()) return true;
  const size_t n = cs->frames.size();
  if (undo.forward_order.size() != n || undo.tail_before.size() != n) {
    return false;
  }
  std::vector<size_t> inv(n, n);
  for (size_t j = 0; j < n; ++j) {
    const size_t src = undo.forward_order[j];
    if (src >= n || inv[src] != n) return false;
    inv[src] = j;
  }
  for (size_t i = 0; i < n; ++i) {
    if (inv[i] == n) return false;
  }
  std::vector<FramedUnit> src = std::move(cs->frames);
  std::vector<FramedUnit> out;
  out.reserve(n);
  for (size_t orig_i = 0; orig_i < n; ++orig_i) {
    out.push_back(std::move(src[inv[orig_i]]));
  }
  cs->frames = std::move(out);
  for (size_t i = 0; i < n; ++i) {
    FrameHeader& fh = cs->frames[i].frame;
    fh.is_last = undo.tail_before[i].first;
    fh.save_as_reference = undo.tail_before[i].second;
  }
  return true;
}

namespace {

bool ImageHeadersAppendCompatible(const ImageHeader& a, const ImageHeader& b,
                                  std::string* err) {
  const ImageMetadata& ma = a.metadata;
  const ImageMetadata& mb = b.metadata;
  if (ma.num_extra != mb.num_extra) {
    *err = "extra channel count mismatch";
    return false;
  }
  if (ma.xyb_encoded != mb.xyb_encoded) {
    *err = "xyb_encoded mismatch (need both XYB or both non-XYB)";
    return false;
  }
  if (!ma.xyb_encoded) {
    if (ma.bit_depth.bits_per_sample != mb.bit_depth.bits_per_sample ||
        ma.bit_depth.float_sample != mb.bit_depth.float_sample ||
        ma.bit_depth.exp_bits != mb.bit_depth.exp_bits) {
      *err = "main bit_depth mismatch (required when not XYB)";
      return false;
    }
  }
  return true;
}

// When the merged image header canvas is larger than |file_canvas| in either
// dimension, regular / skip-progressive frames that still use implicit
// full-canvas framing (!have_crop) must get an explicit crop so the encoded
// extent stays WxH at origin (0,0) on the larger canvas.
static void AppendMaterializeDefaultFrameCrops(std::vector<FramedUnit>* frames,
                                               uint32_t file_canvas_w,
                                               uint32_t file_canvas_h,
                                               uint32_t merged_w,
                                               uint32_t merged_h) {
  if (file_canvas_w >= merged_w && file_canvas_h >= merged_h) return;
  for (FramedUnit& fu : *frames) {
    FrameHeader* fh = &fu.frame;
    if (fh->frame_type != kFrameTypeRegular &&
        fh->frame_type != kFrameTypeSkipProgressive) {
      continue;
    }
    if (fh->have_crop) continue;
    fh->have_crop = true;
    fh->ux0 = PackSigned(0);
    fh->uy0 = PackSigned(0);
    fh->crop_width = file_canvas_w;
    fh->crop_height = file_canvas_h;
  }
}

void ShiftFramedUnitForAppend(FramedUnit* fu, int64_t byte_delta) {
  if (byte_delta < 0) {
    return;
  }
  const size_t bit_delta = static_cast<size_t>(byte_delta) * 8u;
  fu->original_frame_byte_offset =
      static_cast<size_t>(static_cast<int64_t>(fu->original_frame_byte_offset) +
                          byte_delta);
  if (fu->lf_global_noise_lut_abs_valid) {
    fu->lf_global_noise_lut_abs_bit += bit_delta;
  }
  if (fu->lf_global_spline_region_valid) {
    fu->lf_global_spline_region_abs_start_bit += bit_delta;
    fu->lf_global_spline_region_abs_end_bit += bit_delta;
  }
  if (fu->photon_noise_edit) {
    fu->photon_noise_patch_abs_bit += bit_delta;
  }
}

}  // namespace

// 42×3 modular zeros, single group; bare codestream. Base64 "/woQAFIASAgGAQAMAEsgGA==".
static constexpr uint8_t kBuiltinDummyTailCodestream[] = {
    0xff, 0x0a, 0x10, 0x00, 0x52, 0x00, 0x48, 0x08,
    0x06, 0x01, 0x00, 0x0c, 0x00, 0x4b, 0x20, 0x18,
};

static bool IsBuiltinDummyTailCodestreamBytes(const std::vector<uint8_t>& cs) {
  return cs.size() == sizeof(kBuiltinDummyTailCodestream) &&
         std::memcmp(cs.data(), kBuiltinDummyTailCodestream,
                     sizeof(kBuiltinDummyTailCodestream)) == 0;
}

std::vector<uint8_t> BuiltinAppendDummyTailCodestream() {
  return std::vector<uint8_t>(
      kBuiltinDummyTailCodestream,
      kBuiltinDummyTailCodestream + sizeof(kBuiltinDummyTailCodestream));
}

bool AppendCodestreamMerge(const std::vector<uint8_t>& primary_cs,
                            const std::vector<uint8_t>& append_cs,
                            std::vector<uint8_t>* merged_out, std::string* err,
                            bool skip_header_compat) {
  merged_out->clear();
  ParsedCodestream a;
  ParsedCodestream b;
  if (!ReadCodestream(primary_cs.data(), primary_cs.size(), &a)) {
    *err = "failed to parse primary codestream";
    return false;
  }
  if (!ReadCodestream(append_cs.data(), append_cs.size(), &b)) {
    *err = "failed to parse append codestream";
    return false;
  }
  if (a.frames.empty() || b.frames.empty()) {
    *err = "empty frame list";
    return false;
  }
  if (!skip_header_compat) {
    if (!ImageHeadersAppendCompatible(a.image, b.image, err)) {
      return false;
    }
  }
  const uint32_t aw = a.image.size.width;
  const uint32_t ah = a.image.size.height;
  const uint32_t bw = b.image.size.width;
  const uint32_t bh = b.image.size.height;
  const uint32_t merged_w = std::max(aw, bw);
  const uint32_t merged_h = std::max(ah, bh);
  AppendMaterializeDefaultFrameCrops(&a.frames, aw, ah, merged_w, merged_h);
  AppendMaterializeDefaultFrameCrops(&b.frames, bw, bh, merged_w, merged_h);

  const uint32_t ft_last = a.frames.back().frame.frame_type;
  if (ft_last == kFrameTypeLF || ft_last == kFrameTypeReferenceOnly) {
    *err =
        "primary codestream's last frame must be regular or skip-progressive";
    return false;
  }
  if (!b.frames.back().frame.is_last) {
    *err = "append codestream has no terminal is_last frame";
    return false;
  }
  const size_t cut_b = b.frames[0].original_frame_byte_offset;
  if (cut_b > append_cs.size()) {
    *err = "invalid first-frame offset in append codestream";
    return false;
  }
  const int64_t byte_delta =
      static_cast<int64_t>(primary_cs.size()) - static_cast<int64_t>(cut_b);
  if (byte_delta < 0) {
    *err = "invalid append layout (negative offset delta)";
    return false;
  }

  std::vector<uint8_t> combined;
  combined.reserve(primary_cs.size() + (append_cs.size() - cut_b));
  combined.insert(combined.end(), primary_cs.begin(), primary_cs.end());
  combined.insert(combined.end(), append_cs.begin() + cut_b,
                  append_cs.end());

  ParsedCodestream merged;
  merged.image = a.image;
  merged.image.size.width = merged_w;
  merged.image.size.height = merged_h;
  a.frames.back().frame.is_last = false;
  merged.frames = std::move(a.frames);
  merged.frames.reserve(merged.frames.size() + b.frames.size());
  for (size_t i = 0; i < b.frames.size(); ++i) {
    FramedUnit fu = b.frames[i];
    ShiftFramedUnitForAppend(&fu, byte_delta);
    merged.frames.push_back(std::move(fu));
  }
  // Standalone dummy is full-canvas replace (no source bits). After merge the
  // frame becomes a partial overlay on a larger canvas. kAdd with source 1
  // adds the zero patch to reference slot 1; that slot must hold the composited
  // image from the previous frame, so force the former primary tail to
  // save_as_reference=1 when it was is_last (otherwise nothing valid is in slot 1).
  if (IsBuiltinDummyTailCodestreamBytes(append_cs)) {
    if (merged.frames.size() < 2) {
      *err = "internal error: builtin dummy append without prior frame";
      return false;
    }
    FramedUnit& prev = merged.frames[merged.frames.size() - 2];
    FrameHeader* pfh = &prev.frame;
    if (pfh->frame_type == kFrameTypeRegular ||
        pfh->frame_type == kFrameTypeSkipProgressive) {
      pfh->save_as_reference = 1;
    }
    FrameHeader* fh = &merged.frames.back().frame;
    if (fh->frame_type == kFrameTypeRegular ||
        fh->frame_type == kFrameTypeSkipProgressive) {
      fh->blending_info.mode = 1;  // kAdd
      fh->blending_info.source = 1;
    }
  }
  if (!merged.frames.back().frame.is_last) {
    *err = "internal error: merged last frame is not is_last";
    return false;
  }
  if (!WriteCodestream(merged, combined.data(), merged_out)) {
    *err = "failed to rewrite merged codestream";
    return false;
  }
  return true;
}

}  // namespace jxltran
