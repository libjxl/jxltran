// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "reversible_undo.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <vector>

#include "bits.h"
#include "codestream.h"
#include "frame_header.h"
#include "operations.h"
#include "orientation_compose.h"
#include "opsin_adjust.h"
#include "printf_macros.h"
#include "restoration_filter.h"

namespace jxltran {

namespace {

static const char* ExtraChannelTypeToCliSlug(ExtraChannelType t) {
  switch (t) {
    case ExtraChannelType::kAlpha:
      return "alpha";
    case ExtraChannelType::kDepth:
      return "depth";
    case ExtraChannelType::kSpotColour:
      return "spot";
    case ExtraChannelType::kSelectionMask:
      return "selection";
    case ExtraChannelType::kBlack:
      return "black";
    case ExtraChannelType::kCFA:
      return "cfa";
    case ExtraChannelType::kThermal:
      return "thermal";
    case ExtraChannelType::kNonOptional:
      return "nonoptional";
    case ExtraChannelType::kOptional:
      return "optional";
    default:
      return nullptr;
  }
}

// Appends a single ` --set-extra-channel=INDEX:...` that restores |ec|.
static bool AppendSetExtraChannelRestoreToCmd(size_t index,
                                               const ExtraChannelInfo& ec,
                                               std::string* cmd) {
  std::string spec;
  char ib[48];
  snprintf(ib, sizeof(ib), "%" PRIuS ":", index);
  spec += ib;
  if (ec.d_alpha) {
    spec += "d_alpha=1";
    cmd->append(" --set-extra-channel=");
    cmd->append(spec);
    return true;
  }
  spec += "d_alpha=0";
  const char* slug = ExtraChannelTypeToCliSlug(ec.type);
  if (!slug) return false;
  char bitsbuf[160];
  snprintf(bitsbuf, sizeof(bitsbuf), ",type=%s,bits=%" PRIu32 ",float=%d",
           slug, ec.bit_depth.bits_per_sample, ec.bit_depth.float_sample ? 1 : 0);
  spec += bitsbuf;
  if (ec.bit_depth.float_sample) {
    char expbuf[40];
    snprintf(expbuf, sizeof(expbuf), ",exp=%" PRIu32, ec.bit_depth.exp_bits);
    spec += expbuf;
  }
  if (ec.type == ExtraChannelType::kAlpha) {
    char abuf[32];
    snprintf(abuf, sizeof(abuf), ",assoc=%d", ec.alpha_associated ? 1 : 0);
    spec += abuf;
  }
  if (!ec.name.empty()) {
    spec += ",name_hex=";
    for (uint8_t c : ec.name) {
      char hx[4];
      snprintf(hx, sizeof(hx), "%02x", static_cast<unsigned>(c));
      spec += hx;
    }
  }
  if (ec.type == ExtraChannelType::kSpotColour) {
    char sp[384];
    snprintf(sp, sizeof(sp),
             ",spot_r=%.17g,spot_g=%.17g,spot_b=%.17g,spot_solidity=%.17g",
             static_cast<double>(F16BitsToFloat(ec.spot_red)),
             static_cast<double>(F16BitsToFloat(ec.spot_green)),
             static_cast<double>(F16BitsToFloat(ec.spot_blue)),
             static_cast<double>(F16BitsToFloat(ec.spot_solidity)));
    spec += sp;
  }
  if (ec.type == ExtraChannelType::kCFA) {
    char cf[48];
    snprintf(cf, sizeof(cf), ",cfa=%" PRIu32, ec.cfa_channel);
    spec += cf;
  }
  cmd->append(" --set-extra-channel=");
  cmd->append(spec);
  return true;
}

std::string ShellQuotePath(const char* s) {
  std::string out;
  out.push_back('\'');
  for (const char* p = s; *p; ++p) {
    if (*p == '\'') {
      out += "'\\''";
    } else {
      out.push_back(*p);
    }
  }
  out.push_back('\'');
  return out;
}

std::string HexLowerFromUtf8(const std::vector<uint8_t>& bytes) {
  static const char kHex[] = "0123456789abcdef";
  std::string h;
  h.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    h.push_back(kHex[b >> 4]);
    h.push_back(kHex[b & 15]);
  }
  return h;
}

// Even-length all-hex-ASCII names would parse as hex if emitted literally
// (e.g. "6809" vs UTF-8 for U+6809); force hex for those.
bool NameBytesAmbiguousEvenHexAscii(const std::vector<uint8_t>& name) {
  if (name.empty() || (name.size() % 2) != 0) return false;
  for (uint8_t b : name) {
    if (!std::isxdigit(static_cast<unsigned char>(b))) return false;
  }
  return true;
}

// Printable ASCII (33–126) safe as an unquoted INDEX:NAME token; comma/colon
// break pair parsing; space requires "" (see AppendFrameNameTokenForUndo).
bool NameBytesEmitAsUndoUnquotedLiteral(const std::vector<uint8_t>& name) {
  if (name.empty()) return false;
  for (uint8_t b : name) {
    if (b < 33 || b > 126) return false;
    if (b == ',' || b == ':') return false;
  }
  return !NameBytesAmbiguousEvenHexAscii(name);
}

bool NameBytesEmitAsUndoQuotedAscii(const std::vector<uint8_t>& name) {
  if (name.empty()) return false;
  for (uint8_t b : name) {
    if (b < 32 || b > 126) return false;
  }
  return !NameBytesEmitAsUndoUnquotedLiteral(name);
}

// |ApplyFrameBlendOverrides| mirrors the main row onto every extra channel, so a
// single |--set-frame-blends| argv cannot invert a prior state where EC rows
// differed from main. |VerifyRoundtripCodestream| still restores exactly via
// |ApplyFrameBlendExactRestore|.
bool ExtraChannelBlendingMatchesMain(
    const FrameBlendingInfo& main,
    const std::vector<FrameBlendingInfo>& ec) {
  for (const FrameBlendingInfo& bi : ec) {
    if (bi.mode != main.mode || bi.alpha_channel != main.alpha_channel ||
        bi.clamp != main.clamp || bi.source != main.source) {
      return false;
    }
  }
  return true;
}

void AppendFrameNameTokenForUndo(std::string* cmd,
                                 const std::vector<uint8_t>& name) {
  if (name.empty()) return;
  if (NameBytesEmitAsUndoUnquotedLiteral(name)) {
    for (uint8_t b : name) {
      cmd->push_back(static_cast<char>(b));
    }
  } else if (NameBytesEmitAsUndoQuotedAscii(name)) {
    cmd->push_back('"');
    for (uint8_t b : name) {
      if (b == '\\' || b == '"') {
        cmd->push_back('\\');
      }
      cmd->push_back(static_cast<char>(b));
    }
    cmd->push_back('"');
  } else {
    *cmd += HexLowerFromUtf8(name);
  }
}

static const char* BlendModeSlug(uint32_t m) {
  switch (m) {
    case 0:
      return "replace";
    case 1:
      return "add";
    case 2:
      return "blend";
    case 3:
      return "alpha_weighted_add";
    case 4:
      return "mul";
    default:
      return "replace";
  }
}

bool FrameDisplayAabb(const ParsedCodestream& cs, size_t frame_index,
                      uint32_t orient, int64_t* out_dx, int64_t* out_dy,
                      uint64_t* out_dw, uint64_t* out_dh) {
  if (frame_index >= cs.frames.size()) return false;
  const FrameHeader& fh = cs.frames[frame_index].frame;
  const uint32_t canvas_w = cs.image.size.width;
  const uint32_t canvas_h = cs.image.size.height;
  int64_t sx_old = 0;
  int64_t sy_old = 0;
  uint64_t rw = canvas_w;
  uint64_t rh = canvas_h;
  if (fh.have_crop) {
    sx_old = UnpackSigned(fh.ux0);
    sy_old = UnpackSigned(fh.uy0);
    rw = fh.crop_width;
    rh = fh.crop_height;
  }
  StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx_old, sy_old, rw, rh,
                           out_dx, out_dy, out_dw, out_dh);
  return true;
}

bool LogicalSection0BodyByteOffsetRu(const FramedUnit& fu, size_t* out_bytes) {
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

std::string HexLowerPhoton10(const std::array<uint8_t, 10>& b) {
  std::vector<uint8_t> v(b.begin(), b.end());
  return HexLowerFromUtf8(v);
}

// Replace bits [del_start_bit, del_end_bit_excl) in |buf| with
// ins_src[ins_start_bit .. +ins_num_bits). Used so --check_reversible can restore
// the exact pre-transform frame-header encoding (e.g. packed all-default) after
// EPF edits expanded the loop-filter bundle in the working codestream.
bool SpliceCodestreamReplaceBitSpan(std::vector<uint8_t>* buf,
                                    size_t del_start_bit,
                                    size_t del_end_bit_excl,
                                    const uint8_t* ins_src, size_t ins_src_size_bytes,
                                    size_t ins_start_bit, size_t ins_num_bits) {
  if (del_end_bit_excl < del_start_bit) return false;
  const size_t old_bytes = buf->size();
  if (old_bytes > SIZE_MAX / 8u) return false;
  const size_t old_total_bits = old_bytes * 8;
  if (del_end_bit_excl > old_total_bits) return false;
  if (ins_num_bits != 0) {
    if (ins_src_size_bytes > SIZE_MAX / 8u) return false;
    const size_t ins_total_bits = ins_src_size_bytes * 8;
    if (ins_start_bit > ins_total_bits || ins_num_bits > ins_total_bits - ins_start_bit) {
      return false;
    }
  }
  BitWriter bw;
  bw.AppendBitsRange(buf->data(), 0, del_start_bit);
  bw.AppendBitsRange(ins_src, ins_start_bit, ins_num_bits);
  bw.AppendBitsRange(buf->data(), del_end_bit_excl,
                     old_total_bits - del_end_bit_excl);
  bw.ZeroPadToByte();
  *buf = bw.TakeBytes();
  return true;
}

// Frames whose frame headers may need a verbatim splice from the expected
// codestream after --gab-* (same rf_reencode / packed-header issue as EPF).
void CollectGabHeaderSpliceFrameIndices(const ParsedCodestream& exp_cs,
                                       const std::vector<size_t>& selective_frames,
                                       std::vector<size_t>* out) {
  out->clear();
  const auto push_if_editable = [&](size_t fi) {
    if (fi >= exp_cs.frames.size()) return;
    const uint32_t ft = exp_cs.frames[fi].frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) return;
    out->push_back(fi);
  };
  if (!selective_frames.empty()) {
    for (size_t fi : selective_frames) push_if_editable(fi);
  } else {
    for (size_t fi = 0; fi < exp_cs.frames.size(); ++fi) push_if_editable(fi);
  }
  std::sort(out->begin(), out->end());
  out->erase(std::unique(out->begin(), out->end()), out->end());
}

}  // namespace

void UndoRecorder::Reset() {
  *this = UndoRecorder();
}

void UndoRecorder::SetBoxPipelineNotReversible(const char* reason) {
  box_pipeline_ok_ = false;
  if (!block_reason_) block_reason_ = reason;
}

void UndoRecorder::SetCodestreamNotReversible(const char* reason) {
  codestream_ok_ = false;
  if (!block_reason_) block_reason_ = reason;
}

void UndoRecorder::BlockCs(const char* reason) {
  SetCodestreamNotReversible(reason);
}

void UndoRecorder::CapturePreHeader(const ParsedCodestream& parsed) {
  const ImageMetadata& m = parsed.image.metadata;
  orient_before_ = (m.orientation >= 1 && m.orientation <= 8) ? m.orientation : 1;
  // Snapshot main image BitDepth for --set-bits-per-sample / --set-main-float /
  // --set-main-exp undo on any colour encoding (XYB and non-XYB).
  main_bit_depth_before_ = m.bit_depth;
  if (m.have_animation) {
    have_animation_before_ = true;
    num_loops_before_ = m.animation.num_loops;
    tps_num_before_ = m.animation.tps_numerator;
    tps_den_before_ = m.animation.tps_denominator;
  } else {
    have_animation_before_ = false;
  }
  tone_mapping_before_ = m.tone_mapping;
  image_metadata_all_default_before_ = m.all_default;
}

void UndoRecorder::NoteExtraChannelHeaderBeforeApply(
    const ParsedCodestream& pre,
    const std::vector<ExtraChannelHeaderPatch>& patches) {
  if (patches.empty()) return;
  for (const ExtraChannelHeaderPatch& p : patches) {
    if (p.index >= pre.image.metadata.ec_info.size()) continue;
    bool dup = false;
    for (const auto& prev : extra_channel_undo_) {
      if (prev.first == p.index) {
        dup = true;
        break;
      }
    }
    if (dup) continue;
    extra_channel_undo_.emplace_back(
        p.index, pre.image.metadata.ec_info[p.index]);
  }
  if (!extra_channel_undo_.empty()) {
    undo_extra_channels_ = true;
    has_any_step_ = true;
    std::sort(extra_channel_undo_.begin(), extra_channel_undo_.end(),
              [](const std::pair<size_t, ExtraChannelInfo>& a,
                 const std::pair<size_t, ExtraChannelInfo>& b) {
                return a.first < b.first;
              });
  }
}

void UndoRecorder::NoteCropApplied(uint32_t old_canvas_w, uint32_t old_canvas_h,
                                   int32_t storage_dx, int32_t storage_dy,
                                   uint32_t new_canvas_w, uint32_t new_canvas_h) {
  crop_undo_ = true;
  crop_old_w_ = old_canvas_w;
  crop_old_h_ = old_canvas_h;
  crop_storage_dx_ = storage_dx;
  crop_storage_dy_ = storage_dy;
  crop_new_w_ = new_canvas_w;
  crop_new_h_ = new_canvas_h;
  has_any_step_ = true;
}

void UndoRecorder::NoteGabBlur(float amount) {
  gab_undo_ = true;
  gab_undo_kind_ = GabUndoKind::kSharpen;
  gab_undo_amount_ = amount;
  has_any_step_ = true;
}

void UndoRecorder::NoteGabSharpen(float amount) {
  gab_undo_ = true;
  gab_undo_kind_ = GabUndoKind::kBlur;
  gab_undo_amount_ = amount;
  has_any_step_ = true;
}

void UndoRecorder::NoteEpfAmplitudeScale(float factor) {
  if (factor == 1.f || !std::isfinite(factor)) return;
  epf_amp_undo_ = true;
  epf_amp_forward_factor_ = factor;
  has_any_step_ = true;
}

void UndoRecorder::CaptureRestorationBeforeEpfOps(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  epf_rf_before_valid_.assign(cs.frames.size(), false);
  epf_rf_snap_modular_.assign(cs.frames.size(), false);
  epf_rf_before_.resize(cs.frames.size());
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    epf_rf_before_valid_[fi] = true;
    epf_rf_snap_modular_[fi] = (fu.frame.encoding == kFrameEncModular);
    epf_rf_before_[fi] = fu.frame.restoration;
  }
}

void UndoRecorder::FinalizeEpfRestorationUndo(uint32_t mask) {
  epf_undo_mask_ = mask;
  epf_restore_verify_ = (mask != 0u);
  if (!epf_restore_verify_) {
    epf_rf_before_valid_.clear();
    epf_rf_snap_modular_.clear();
    epf_rf_before_.clear();
  }
  if (epf_restore_verify_) {
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteOpsinAdjust(bool exp, bool temp, bool tint, bool hue,
                                   float ev, float temperature, float tint_v,
                                   float hue_v) {
  opsin_undo_ = true;
  opsin_set_exp_ = exp;
  opsin_set_temp_ = temp;
  opsin_set_tint_ = tint;
  opsin_set_hue_ = hue;
  opsin_ev_ = ev;
  opsin_temp_ = temperature;
  opsin_tint_ = tint_v;
  opsin_hue_ = hue_v;
  has_any_step_ = true;
}

void UndoRecorder::NoteOpsinBiasAdjust(bool sx, bool sy, bool sb, float vx,
                                       float vy, float vb) {
  if (!sx && !sy && !sb) return;
  opsin_bias_undo_ = true;
  opsin_bias_set_[0] = sx;
  opsin_bias_set_[1] = sy;
  opsin_bias_set_[2] = sb;
  opsin_bias_v_[0] = vx;
  opsin_bias_v_[1] = vy;
  opsin_bias_v_[2] = vb;
  has_any_step_ = true;
}

void UndoRecorder::NoteOpsinQuantBiasAdjust(bool s0, bool s1, bool s2, bool s3,
                                             float v0, float v1, float v2,
                                             float v3) {
  if (!s0 && !s1 && !s2 && !s3) return;
  opsin_quant_bias_undo_ = true;
  opsin_quant_bias_set_[0] = s0;
  opsin_quant_bias_set_[1] = s1;
  opsin_quant_bias_set_[2] = s2;
  opsin_quant_bias_set_[3] = s3;
  opsin_quant_bias_v_[0] = v0;
  opsin_quant_bias_v_[1] = v1;
  opsin_quant_bias_v_[2] = v2;
  opsin_quant_bias_v_[3] = v3;
  has_any_step_ = true;
}

bool UndoRecorder::CaptureFrameRegionBeforeMove(
    const ParsedCodestream& parsed, size_t frame_index,
    uint32_t orient_before_header) {
  int64_t dx = 0;
  int64_t dy = 0;
  uint64_t dw = 0;
  uint64_t dh = 0;
  if (!FrameDisplayAabb(parsed, frame_index, orient_before_header, &dx, &dy,
                         &dw, &dh)) {
    return false;
  }
  FrameRegionUndo u;
  u.frame_index = frame_index;
  u.orient_before_header = orient_before_header;
  u.old_display_x = dx;
  u.old_display_y = dy;
  u.display_w = dw;
  u.display_h = dh;
  frame_region_undos_.push_back(std::move(u));
  has_any_step_ = true;
  return true;
}

void UndoRecorder::NoteFrameNameBeforeChange(const ParsedCodestream& parsed,
                                             size_t frame_index) {
  if (frame_index >= parsed.frames.size()) return;
  const FrameHeader& fh = parsed.frames[frame_index].frame;
  FrameNameUndo u;
  u.frame_index = frame_index;
  u.old_name_utf8 = fh.name;
  frame_name_undos_.push_back(std::move(u));
  has_any_step_ = true;
}

namespace {

const char* FrameTypeSlugForUndo(uint32_t t) {
  switch (t) {
    case kFrameTypeLF:
      return "lf";
    case kFrameTypeReferenceOnly:
      return "ref_only";
    case kFrameTypeSkipProgressive:
      return "skip_prog";
    default:
      return "regular";
  }
}

}  // namespace

void UndoRecorder::NoteFrameTypeBeforeChange(const ParsedCodestream& parsed,
                                             size_t frame_index) {
  if (frame_index >= parsed.frames.size()) return;
  const FrameHeader& fh = parsed.frames[frame_index].frame;
  FrameTypeUndo u;
  u.frame_index = frame_index;
  u.old_type = fh.frame_type;
  u.old_is_last = fh.is_last;
  u.old_save_before_ct = fh.save_before_ct;
  u.old_blend_mode = fh.blending_info.mode;
  u.old_blend_source = fh.blending_info.source;
  u.old_alpha_channel = fh.blending_info.alpha_channel;
  u.old_clamp = fh.blending_info.clamp;
  u.old_duration = fh.duration;
  u.old_have_crop = fh.have_crop;
  u.old_x0 = UnpackSigned(fh.ux0);
  u.old_y0 = UnpackSigned(fh.uy0);
  frame_type_undos_.push_back(u);
  has_any_step_ = true;
}

void UndoRecorder::NoteSplinesAddedOnFrames(
    const std::vector<std::pair<size_t, int64_t>>& entries) {
  if (entries.empty()) return;
  for (const auto& pr : entries) {
    SplineAddUndoEntry e;
    e.frame_index = pr.first;
    e.forward_body_delta_bits = pr.second;
    e.lf_chunk0_tail_trim_bytes = 0;
    spline_add_undos_.push_back(e);
  }
  has_any_step_ = true;
}

void UndoRecorder::FinalizeSplineLfGlobalChunk0TailTrim(
    const uint8_t* ref_cs, size_t ref_n, const uint8_t* out_cs, size_t out_n) {
  if (spline_add_undos_.empty() && lf_global_scale_undos_.empty()) return;
  ParsedCodestream ref_parsed;
  ParsedCodestream out_parsed;
  if (!ReadCodestream(ref_cs, ref_n, &ref_parsed) ||
      !ReadCodestream(out_cs, out_n, &out_parsed)) {
    return;
  }
  std::vector<size_t> frames;
  for (const SplineAddUndoEntry& e : spline_add_undos_) {
    frames.push_back(e.frame_index);
  }
  for (const LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
    frames.push_back(e.frame_index);
  }
  std::sort(frames.begin(), frames.end());
  frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
  std::map<size_t, uint8_t> trim;
  for (size_t fi : frames) {
    trim[fi] = ComputeLfGlobalChunk0TailPaddingTrimBytes(
        out_cs, out_n, out_parsed, ref_cs, ref_n, ref_parsed, fi);
  }
  for (SplineAddUndoEntry& e : spline_add_undos_) {
    auto it = trim.find(e.frame_index);
    if (it != trim.end()) {
      e.lf_chunk0_tail_trim_bytes = it->second;
    }
  }
  for (LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
    auto it = trim.find(e.frame_index);
    if (it != trim.end()) {
      e.lf_chunk0_tail_trim_bytes = it->second;
    }
  }
}

void UndoRecorder::CaptureLfGlobalScaleBeforeApply(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames,
    uint32_t forward_new_scale) {
  lf_global_scale_undos_.clear();
  const auto want = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  std::optional<uint32_t> common_old;
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    if (fh.encoding == kFrameEncModular) continue;
    if (!fu.lf_global_quantizer_parsed) continue;
    if (forward_new_scale == fu.lf_global_quantizer_global_scale) continue;
    LfGlobalScaleUndoEntry e;
    e.frame_index = fi;
    e.old_scale = fu.lf_global_quantizer_global_scale;
    if (common_old.has_value() && *common_old != e.old_scale) {
      SetCodestreamNotReversible(
          "--lf-global-scale: edited frames had different prior global_scale "
          "(undo not emitted)");
      lf_global_scale_undos_.clear();
      return;
    }
    common_old = e.old_scale;
    lf_global_scale_undos_.push_back(std::move(e));
  }
  if (!lf_global_scale_undos_.empty()) {
    has_any_step_ = true;
  }
}

void UndoRecorder::CapturePhotonNoiseBeforeApply(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  photon_before_by_frame_.clear();
  photon_before_by_frame_.resize(cs.frames.size());
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    size_t sec0 = 0;
    if (!LogicalSection0BodyByteOffsetRu(fu, &sec0)) continue;
    if (fu.body_bit_length == 0) continue;
    PhotonBeforeSnap& s = photon_before_by_frame_[fi];
    s.had_noise = (fh.flags & kFrameFlagNoise) != 0;
    if (s.had_noise && fu.lf_global_noise_raw_valid) {
      s.raw_valid = true;
      s.raw_bytes = fu.lf_global_noise_raw_bytes;
    }
  }
}

void UndoRecorder::FinalizePhotonNoiseAfterApply(const ParsedCodestream& cs) {
  photon_undos_.clear();
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    const FramedUnit& fu = cs.frames[fi];
    if (!fu.photon_noise_edit) continue;
    PhotonNoiseUndoSpec u;
    u.frame_index = fi;
    if (fi >= photon_before_by_frame_.size()) {
      SetCodestreamNotReversible("photon noise: internal undo snapshot mismatch");
      return;
    }
    const PhotonBeforeSnap& snap = photon_before_by_frame_[fi];
    if (fu.photon_noise_delta_bytes == 10) {
      u.forward_kind = PhotonNoiseForwardKind::kInsert;
    } else if (fu.photon_noise_delta_bytes == -10) {
      u.forward_kind = PhotonNoiseForwardKind::kRemove;
      if (!snap.had_noise || !snap.raw_valid) {
        SetCodestreamNotReversible(
            "photon noise: internal undo state for removed LUT");
        return;
      }
      u.prior_bytes = snap.raw_bytes;
    } else if (fu.photon_noise_delta_bytes == 0) {
      u.forward_kind = PhotonNoiseForwardKind::kReplace;
      if (!snap.had_noise || !snap.raw_valid) {
        SetCodestreamNotReversible(
            "photon noise: internal undo state for replaced LUT");
        return;
      }
      u.prior_bytes = snap.raw_bytes;
    } else {
      SetCodestreamNotReversible("photon noise: unknown splice delta for undo");
      return;
    }
    photon_undos_.push_back(u);
  }
  photon_before_by_frame_.clear();
  if (!photon_undos_.empty()) {
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteLfDcQuantMulForward(const std::vector<size_t>& splice_frames,
                                           float mx, float my, float mb) {
  lf_dc_quant_mul_undo_ = true;
  lf_dc_splice_frames_ = splice_frames;
  std::sort(lf_dc_splice_frames_.begin(), lf_dc_splice_frames_.end());
  lf_dc_splice_frames_.erase(
      std::unique(lf_dc_splice_frames_.begin(), lf_dc_splice_frames_.end()),
      lf_dc_splice_frames_.end());
  lf_dc_quant_mul_forward_[0] = mx;
  lf_dc_quant_mul_forward_[1] = my;
  lf_dc_quant_mul_forward_[2] = mb;
  has_any_step_ = true;
}

void UndoRecorder::NoteFrameBlendBeforeOverrides(
    const ParsedCodestream& parsed,
    const std::vector<FrameBlendOverride>& overrides) {
  if (overrides.empty()) return;
  std::vector<bool> seen(parsed.frames.size(), false);
  const uint32_t num_extra = parsed.image.metadata.num_extra;
  for (const FrameBlendOverride& o : overrides) {
    if (o.frame_index >= parsed.frames.size()) continue;
    if (seen[o.frame_index]) continue;
    seen[o.frame_index] = true;
    const FrameHeader& fh = parsed.frames[o.frame_index].frame;
    BlendUndo u;
    u.frame_index = o.frame_index;
    u.num_extra = num_extra;
    u.is_last = fh.is_last;
    u.main = fh.blending_info;
    u.ec = fh.ec_blending_info;
    u.save_as_reference = fh.save_as_reference;
    blend_undos_.push_back(std::move(u));
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteKeepReorder(KeepReorderUndoSpec spec) {
  if (spec.forward_order.empty()) return;
  keep_reorder_undo_ = true;
  keep_reorder_spec_ = std::move(spec);
  has_any_step_ = true;
}

void UndoRecorder::NoteHeaderChanges(bool set_abs_orient_nonzero,
                                     bool set_rel_nonzero, uint32_t orient_after,
                                     bool set_main_bit_depth_nonzero,
                                     bool set_loops, bool set_tps,
                                     bool set_intensity_target) {
  if (set_abs_orient_nonzero || set_rel_nonzero) {
    if (orient_after != orient_before_) {
      undo_orientation_ = true;
      has_any_step_ = true;
    }
  }
  if (set_main_bit_depth_nonzero) {
    undo_main_bit_depth_ = true;
    has_any_step_ = true;
  }
  if (set_loops) {
    undo_loops_ = true;
    has_any_step_ = true;
  }
  if (set_tps) {
    undo_tps_ = true;
    has_any_step_ = true;
  }
  if (set_intensity_target) {
    undo_intensity_ = true;
    has_any_step_ = true;
  }
}

void UndoRecorder::SetSelectiveFramesCopy(const std::vector<size_t>& indices) {
  selective_frames_ = indices;
}

void UndoRecorder::FinalizePipelineCompatibility() {
  if (crop_undo_ && !frame_region_undos_.empty()) {
    SetCodestreamNotReversible(
        "--crop combined with --set-frame-region (multi-pass undo not emitted)");
  }
  if (crop_undo_ && (gab_undo_ || epf_amp_undo_ || epf_restore_verify_)) {
    SetCodestreamNotReversible(
        "--crop combined with --gab-* or --set-epf-*");
  }
  if (crop_undo_ && !spline_add_undos_.empty()) {
    SetCodestreamNotReversible(
        "--crop combined with --set-splines-from (add-only undo not emitted)");
  }
  if (crop_undo_ && !lf_global_scale_undos_.empty()) {
    SetCodestreamNotReversible(
        "--crop combined with --lf-global-scale (undo not emitted)");
  }
  if (!frame_region_undos_.empty() && !spline_add_undos_.empty()) {
    SetCodestreamNotReversible(
        "--set-frame-region combined with --set-splines-from (add-only undo not "
        "emitted)");
  }
}

void UndoRecorder::CaptureTocBeforeGroupOrder(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  if (!CaptureTocSnapshotsBeforeGroupOrder(cs, only_frames, &toc_snaps_)) {
    toc_snaps_.clear();
  }
}

void UndoRecorder::FinalizeTocGroupOrderAfterApply(bool toc_changed) {
  if (!toc_changed) {
    toc_snaps_.clear();
    return;
  }
  toc_undo_ = true;
  has_any_step_ = true;
}

bool UndoRecorder::BuildUndoCommandLine(const char* argv0, const char* out_jxl,
                                        const char* restored_jxl,
                                        std::string* out_line) const {
  out_line->clear();
  if (!box_pipeline_ok_ || !codestream_ok_ || !has_any_step_) return false;
  if (toc_undo_) {
    // TOC layout is restored from an internal snapshot for --check_reversible;
    // there is no single argv inverse that round-trips arbitrary permutations.
    return false;
  }
  for (const BlendUndo& b : blend_undos_) {
    if (!ExtraChannelBlendingMatchesMain(b.main, b.ec)) {
      return false;
    }
  }
  std::string cmd;
  cmd += ShellQuotePath(argv0);
  cmd += ' ';
  cmd += ShellQuotePath(out_jxl);

  if (keep_reorder_undo_) {
    cmd += " --keep-listed-frames --keep-frames=";
    const std::vector<size_t>& fw = keep_reorder_spec_.forward_order;
    const size_t n = fw.size();
    std::vector<size_t> inv(n);
    for (size_t j = 0; j < n; ++j) {
      inv[fw[j]] = j;
    }
    for (size_t i = 0; i < n; ++i) {
      if (i != 0) cmd += ',';
      char b[32];
      snprintf(b, sizeof(b), "%" PRIuS, inv[i]);
      cmd += b;
    }
  }

  if (undo_orientation_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-orientation=%" PRIu32, orient_before_);
    cmd += buf;
  }
  if (undo_main_bit_depth_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-bits-per-sample=%" PRIu32,
             main_bit_depth_before_.bits_per_sample);
    cmd += buf;
    if (main_bit_depth_before_.float_sample) {
      cmd += " --set-main-float=1";
      char ebuf[48];
      snprintf(ebuf, sizeof(ebuf), " --set-main-exp=%" PRIu32,
               main_bit_depth_before_.exp_bits);
      cmd += ebuf;
    } else {
      cmd += " --set-main-float=0";
    }
  }
  if (undo_loops_ && have_animation_before_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-num-loops=%" PRIu32, num_loops_before_);
    cmd += buf;
  }
  if (undo_tps_ && have_animation_before_) {
    char buf[80];
    snprintf(buf, sizeof(buf), " --set-tps=%" PRIu32 "/%" PRIu32, tps_num_before_,
             tps_den_before_);
    cmd += buf;
  }

  if (undo_intensity_) {
    if (tone_mapping_before_.all_default) {
      cmd += " --set-intensity-target=default";
    } else if (!tone_mapping_before_.relative_to_max_display &&
               tone_mapping_before_.min_nits == FloatToF16Bits(0.f) &&
               tone_mapping_before_.linear_below == FloatToF16Bits(0.f)) {
      char buf[96];
      snprintf(buf, sizeof(buf), " --set-intensity-target=%.9g",
               static_cast<double>(
                   F16BitsToFloat(tone_mapping_before_.intensity_target)));
      cmd += buf;
    }
  }

  if (undo_extra_channels_) {
    for (const auto& pr : extra_channel_undo_) {
      if (!AppendSetExtraChannelRestoreToCmd(pr.first, pr.second, &cmd)) {
        return false;
      }
    }
  }

  // Forward pass applies opsin before crop, then gab/epf/splines/photon, then
  // lf_dc, lf_global, toc, blend (jxltran.cc). VerifyRoundtripCodestream applies
  // inverses in reverse composition order on a ParsedCodestream; the emitted
  // reversible_undo argv still follows jxltran's fixed per-invocation order, so
  // a single manual run may not match Verify unless operations commute.
  if (!lf_global_scale_undos_.empty()) {
    std::vector<size_t> ord;
    ord.reserve(lf_global_scale_undos_.size());
    for (size_t i = 0; i < lf_global_scale_undos_.size(); ++i) {
      ord.push_back(i);
    }
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return lf_global_scale_undos_[a].frame_index <
             lf_global_scale_undos_[b].frame_index;
    });
    cmd += " --frames=";
    for (size_t k = 0; k < ord.size(); ++k) {
      if (k != 0) cmd += ',';
      char b[32];
      snprintf(b, sizeof(b), "%" PRIuS,
               lf_global_scale_undos_[ord[k]].frame_index);
      cmd += b;
    }
    char gbuf[48];
    snprintf(gbuf, sizeof(gbuf), " --lf-global-scale=%" PRIu32,
             lf_global_scale_undos_[ord[0]].old_scale);
    cmd += gbuf;
  }
  if (!lf_global_scale_undos_.empty() && spline_add_undos_.empty()) {
    bool any_trim_lf = false;
    std::map<size_t, uint8_t> trim_lf;
    for (const LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
      if (e.lf_chunk0_tail_trim_bytes == 0) continue;
      any_trim_lf = true;
      trim_lf[e.frame_index] = e.lf_chunk0_tail_trim_bytes;
    }
    if (any_trim_lf) {
      cmd += " --lf-global-chunk0-tail-trim-bytes=";
      bool first_tp = true;
      for (const auto& pr : trim_lf) {
        if (!first_tp) cmd += ',';
        first_tp = false;
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), "%" PRIuS ":%u", pr.first,
                 static_cast<unsigned>(pr.second));
        cmd += tbuf;
      }
    }
  }

  if (lf_dc_quant_mul_undo_) {
    char qbuf[200];
    const double ix =
        static_cast<double>(1.0f / lf_dc_quant_mul_forward_[0]);
    const double iy =
        static_cast<double>(1.0f / lf_dc_quant_mul_forward_[1]);
    const double iz =
        static_cast<double>(1.0f / lf_dc_quant_mul_forward_[2]);
    // %.9g truncates reciprocals so a manual undo argv may not parse back to
    // the exact float inverse of the forward factors (breaks round-trip).
    snprintf(qbuf, sizeof(qbuf), " --lf-dc-quant-mul=%.17g,%.17g,%.17g", ix, iy, iz);
    cmd += qbuf;
    if (!selective_frames_.empty()) {
      cmd += " --frames=";
      for (size_t i = 0; i < selective_frames_.size(); ++i) {
        if (i != 0) cmd += ',';
        char b[32];
        snprintf(b, sizeof(b), "%" PRIuS, selective_frames_[i]);
        cmd += b;
      }
    }
  }

  for (size_t fri = 0; fri < frame_region_undos_.size(); ++fri) {
    const FrameRegionUndo& fr = frame_region_undos_[fri];
    cmd += " --set-frame-region=";
    char buf[160];
    snprintf(buf, sizeof(buf),
             "%" PRIuS ":%" PRIu64 "x%" PRIu64 "%+" PRId64 "%+" PRId64,
             fr.frame_index, fr.display_w, fr.display_h, fr.old_display_x,
             fr.old_display_y);
    cmd += buf;
  }

  if (crop_undo_) {
    char buf[128];
    snprintf(buf, sizeof(buf), " --crop=%" PRIu32 "x%" PRIu32 "%+" PRId32 "%+" PRId32,
             crop_old_w_, crop_old_h_, -crop_storage_dx_, -crop_storage_dy_);
    cmd += buf;
  }

  if (opsin_quant_bias_undo_) {
    char buf[112];
    for (int qi = 0; qi < 4; ++qi) {
      if (!opsin_quant_bias_set_[qi]) continue;
      snprintf(buf, sizeof(buf), " --opsin-quant-bias-%d=%.9g", qi,
               opsin_quant_bias_v_[qi]);
      cmd += buf;
    }
  }
  if (opsin_bias_undo_) {
    char buf[96];
    static const char* kBiasFlag[3] = {"--opsin-bias-x=", "--opsin-bias-y=",
                                       "--opsin-bias-b="};
    for (int bi = 0; bi < 3; ++bi) {
      if (!opsin_bias_set_[bi]) continue;
      snprintf(buf, sizeof(buf), " %s%.9g", kBiasFlag[bi], opsin_bias_v_[bi]);
      cmd += buf;
    }
  }
  if (opsin_undo_) {
    cmd += " --opsin-inverse";
    char buf[128];
    if (opsin_set_exp_) {
      snprintf(buf, sizeof(buf), " --opsin-exposure=%.9g", opsin_ev_);
      cmd += buf;
    }
    if (opsin_set_temp_) {
      snprintf(buf, sizeof(buf), " --opsin-temperature=%.9g", opsin_temp_);
      cmd += buf;
    }
    if (opsin_set_tint_) {
      snprintf(buf, sizeof(buf), " --opsin-tint=%.9g", opsin_tint_);
      cmd += buf;
    }
    if (opsin_set_hue_) {
      snprintf(buf, sizeof(buf), " --opsin-hue=%.9g", opsin_hue_);
      cmd += buf;
    }
  }

  if (gab_undo_) {
    char buf[64];
    if (gab_undo_kind_ == GabUndoKind::kSharpen) {
      snprintf(buf, sizeof(buf), " --gab-sharpen=%.9g", gab_undo_amount_);
    } else {
      snprintf(buf, sizeof(buf), " --gab-blur=%.9g", gab_undo_amount_);
    }
    cmd += buf;
  }

  if (epf_restore_verify_) {
    if (epf_undo_mask_ == 2u && epf_amp_undo_) {
      char buf[80];
      const float inv = 1.0f / epf_amp_forward_factor_;
      snprintf(buf, sizeof(buf), " --set-epf-amplitude-scale=%.9g", inv);
      cmd += buf;
    } else if (epf_undo_mask_ == 1u) {
      uint32_t prior_iters = 0;
      bool seen_any = false;
      for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
        if (!epf_rf_before_valid_[fi]) continue;
        const ParsedRestorationFilter& s = epf_rf_before_[fi];
        const bool modular = epf_rf_snap_modular_[fi];
        const uint32_t it = std::min(
            3u, s.all_default ? DefaultRestorationFilter(modular).epf_iters
                              : s.epf_iters);
        if (!seen_any) {
          prior_iters = it;
          seen_any = true;
        } else if (prior_iters != it) {
          return false;
        }
      }
      if (!seen_any) return false;
      char buf[64];
      snprintf(buf, sizeof(buf), " --set-epf-iters=%" PRIu32, prior_iters);
      cmd += buf;
      if (!selective_frames_.empty()) {
        cmd += " --frames=";
        for (size_t i = 0; i < selective_frames_.size(); ++i) {
          if (i != 0) cmd += ',';
          char fbuf[32];
          snprintf(fbuf, sizeof(fbuf), "%" PRIuS, selective_frames_[i]);
          cmd += fbuf;
        }
      }
    } else {
      return false;
    }
  }

  if (!frame_name_undos_.empty()) {
    cmd += " --set-frame-names=";
    for (ssize_t i = static_cast<ssize_t>(frame_name_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameNameUndo& fn = frame_name_undos_[static_cast<size_t>(i)];
      if (i != static_cast<ssize_t>(frame_name_undos_.size()) - 1) {
        cmd += ',';
      }
      char idxbuf[32];
      snprintf(idxbuf, sizeof(idxbuf), "%" PRIuS ":", fn.frame_index);
      cmd += idxbuf;
      AppendFrameNameTokenForUndo(&cmd, fn.old_name_utf8);
    }
  }

  if (!frame_type_undos_.empty()) {
    cmd += " --set-frame-types=";
    for (ssize_t i = static_cast<ssize_t>(frame_type_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameTypeUndo& ft = frame_type_undos_[static_cast<size_t>(i)];
      if (i != static_cast<ssize_t>(frame_type_undos_.size()) - 1) cmd += ',';
      char buf[256];
      snprintf(buf, sizeof(buf), "%" PRIuS ":%s", ft.frame_index,
               FrameTypeSlugForUndo(ft.old_type));
      cmd += buf;
      if (ft.old_type == kFrameTypeReferenceOnly) {
        // The only field a reference-only header carries that the other types
        // cannot always express.
        snprintf(buf, sizeof(buf), ":sbct=%d", ft.old_save_before_ct ? 1 : 0);
        cmd += buf;
      } else {
        snprintf(buf, sizeof(buf),
                 ":last=%d:blend=%u:src=%u:alpha=%u:clamp=%d:dur=%u",
                 ft.old_is_last ? 1 : 0, ft.old_blend_mode, ft.old_blend_source,
                 ft.old_alpha_channel, ft.old_clamp ? 1 : 0, ft.old_duration);
        cmd += buf;
        if (ft.old_have_crop) {
          snprintf(buf, sizeof(buf), ":x0=%d:y0=%d", ft.old_x0, ft.old_y0);
          cmd += buf;
        }
      }
    }
  }

  if (!blend_undos_.empty()) {
    std::vector<size_t> ord(blend_undos_.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return blend_undos_[a].frame_index < blend_undos_[b].frame_index;
    });
    cmd += " --set-frame-blends=";
    for (size_t k = 0; k < ord.size(); ++k) {
      const BlendUndo& b = blend_undos_[ord[k]];
      if (k != 0) cmd += ',';
      char idxbuf[32];
      snprintf(idxbuf, sizeof(idxbuf), "%" PRIuS ":", b.frame_index);
      cmd += idxbuf;
      cmd += BlendModeSlug(b.main.mode);
      if (b.num_extra > 0) {
        char fbuf[160];
        snprintf(fbuf, sizeof(fbuf), ",alpha=%" PRIu32 ",clamp=%s,source=%" PRIu32,
                 b.main.alpha_channel, b.main.clamp ? "true" : "false",
                 b.main.source);
        cmd += fbuf;
      }
      if (!b.is_last) {
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), ",target=%" PRIu32,
                 b.save_as_reference & 3u);
        cmd += tbuf;
      }
    }
  }

  if (!spline_add_undos_.empty()) {
    std::vector<size_t> u;
    u.reserve(spline_add_undos_.size());
    for (const SplineAddUndoEntry& e : spline_add_undos_) {
      u.push_back(e.frame_index);
    }
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    cmd += " --clear-splines-frames=";
    for (size_t i = 0; i < u.size(); ++i) {
      if (i != 0) cmd += ',';
      char buf[32];
      snprintf(buf, sizeof(buf), "%" PRIuS, u[i]);
      cmd += buf;
    }
    bool any_trim = false;
    std::map<size_t, uint8_t> trim_by_frame;
    for (const SplineAddUndoEntry& e : spline_add_undos_) {
      if (e.lf_chunk0_tail_trim_bytes == 0) continue;
      any_trim = true;
      auto it = trim_by_frame.find(e.frame_index);
      if (it != trim_by_frame.end() && it->second != e.lf_chunk0_tail_trim_bytes) {
        return false;
      }
      trim_by_frame[e.frame_index] = e.lf_chunk0_tail_trim_bytes;
    }
    for (const LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
      if (e.lf_chunk0_tail_trim_bytes == 0) continue;
      any_trim = true;
      auto it = trim_by_frame.find(e.frame_index);
      if (it != trim_by_frame.end() && it->second != e.lf_chunk0_tail_trim_bytes) {
        return false;
      }
      trim_by_frame[e.frame_index] = e.lf_chunk0_tail_trim_bytes;
    }
    if (any_trim) {
      cmd += " --lf-global-chunk0-tail-trim-bytes=";
      bool first_tp = true;
      for (const auto& pr : trim_by_frame) {
        if (pr.second == 0) continue;
        if (!first_tp) {
          cmd += ',';
        }
        first_tp = false;
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), "%" PRIuS ":%u", pr.first,
                 static_cast<unsigned>(pr.second));
        cmd += tbuf;
      }
    }
  }

  if (!photon_undos_.empty()) {
    std::vector<size_t> fidx;
    fidx.reserve(photon_undos_.size());
    for (const PhotonNoiseUndoSpec& u : photon_undos_) {
      fidx.push_back(u.frame_index);
    }
    std::sort(fidx.begin(), fidx.end());
    fidx.erase(std::unique(fidx.begin(), fidx.end()), fidx.end());
    auto append_photon_frames = [&]() {
      cmd += " --frames=";
      for (size_t i = 0; i < fidx.size(); ++i) {
        if (i != 0) cmd += ',';
        char b[32];
        snprintf(b, sizeof(b), "%" PRIuS, fidx[i]);
        cmd += b;
      }
    };
    const bool all_insert_forward =
        std::all_of(photon_undos_.begin(), photon_undos_.end(),
                    [](const PhotonNoiseUndoSpec& u) {
                      return u.forward_kind == PhotonNoiseForwardKind::kInsert;
                    });
    if (all_insert_forward) {
      cmd += " --set-photon-noise-iso=0";
      append_photon_frames();
    } else {
      const PhotonNoiseForwardKind k0 = photon_undos_[0].forward_kind;
      if (k0 != PhotonNoiseForwardKind::kRemove &&
          k0 != PhotonNoiseForwardKind::kReplace) {
        return false;
      }
      for (const PhotonNoiseUndoSpec& u : photon_undos_) {
        if (u.forward_kind != k0) return false;
      }
      const std::array<uint8_t, 10>& ref = photon_undos_[0].prior_bytes;
      for (const PhotonNoiseUndoSpec& u : photon_undos_) {
        if (u.prior_bytes != ref) return false;
      }
      cmd += " --set-photon-noise-weights=";
      cmd += HexLowerPhoton10(ref);
      append_photon_frames();
    }
  }

  cmd += ' ';
  cmd += ShellQuotePath(restored_jxl);
  *out_line = std::move(cmd);
  return true;
}

ReversibleCheckStatus UndoRecorder::VerifyRoundtripCodestream(
    const std::vector<uint8_t>& expected_codestream,
    const std::vector<uint8_t>& out_codestream) const {
  if (!codestream_ok_ || !box_pipeline_ok_) {
    return ReversibleCheckStatus::kNotReversibleSkipped;
  }
  if (!has_any_step_) {
    if (expected_codestream.size() != out_codestream.size()) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (memcmp(expected_codestream.data(), out_codestream.data(),
               expected_codestream.size()) != 0) {
      return ReversibleCheckStatus::kMismatch;
    }
    return ReversibleCheckStatus::kOk;
  }
  jxltran::ParsedCodestream parsed;
  if (!jxltran::ReadCodestream(out_codestream.data(), out_codestream.size(),
                               &parsed)) {
    return ReversibleCheckStatus::kMismatch;
  }

  if (keep_reorder_undo_) {
    if (!ApplyKeepListedFramesReorderInverse(&parsed, keep_reorder_spec_)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (undo_orientation_ || undo_main_bit_depth_ || undo_loops_ || undo_tps_ ||
      undo_intensity_) {
    HeaderMod mod;
    mod.set_orientation = 0;
    if (undo_orientation_) {
      mod.set_orientation = orient_before_;
    }
    mod.set_bits_per_sample =
        undo_main_bit_depth_ ? main_bit_depth_before_.bits_per_sample : 0;
    if (undo_main_bit_depth_) {
      mod.set_main_float_sample = true;
      mod.main_float_sample = main_bit_depth_before_.float_sample;
      if (main_bit_depth_before_.float_sample) {
        mod.set_main_exp_bits = true;
        mod.main_exp_bits = main_bit_depth_before_.exp_bits;
      }
    }
    mod.have_set_num_loops = undo_loops_ && have_animation_before_;
    mod.set_num_loops = num_loops_before_;
    mod.have_set_tps = undo_tps_ && have_animation_before_;
    mod.set_tps_numerator = tps_num_before_;
    mod.set_tps_denominator = tps_den_before_;
    if (undo_intensity_) {
      mod.have_tone_mapping_snapshot = true;
      mod.tone_mapping_snapshot = tone_mapping_before_;
      mod.collapse_to_packed_all_default_image_metadata =
          image_metadata_all_default_before_;
    }
    if (!ApplyHeaderMod(&parsed, mod)) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (undo_main_bit_depth_) {
      parsed.image.metadata.bit_depth = main_bit_depth_before_;
    }
  }

  if (undo_extra_channels_) {
    for (const auto& pr : extra_channel_undo_) {
      if (!RestoreExtraChannelInfoAtIndex(&parsed, pr.first, pr.second)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  for (ssize_t fri = static_cast<ssize_t>(frame_region_undos_.size()) - 1;
       fri >= 0; --fri) {
    const FrameRegionUndo& fr = frame_region_undos_[static_cast<size_t>(fri)];
    bool ch = false;
    if (!ApplySetFrameRegionFromDisplay(
            &parsed, fr.frame_index, fr.orient_before_header,
            static_cast<int32_t>(fr.old_display_x),
            static_cast<int32_t>(fr.old_display_y),
            static_cast<uint32_t>(fr.display_w), static_cast<uint32_t>(fr.display_h),
            &ch)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)ch;
  }

  if (crop_undo_) {
    if (!ApplyCrop(&parsed, -crop_storage_dx_, -crop_storage_dy_, crop_old_w_,
                   crop_old_h_)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  // Reverse of jxltran.cc transform_codestream after crop: gab, epf,
  // set-splines-from (add-only), photon, lf_dc, lf_global, toc, then blend /
  // keep+blend (when not emitted as separate undo), etc.
  const std::vector<size_t>* sel =
      selective_frames_.empty() ? nullptr : &selective_frames_;

  if (gab_undo_) {
    GabArgs ga;
    if (gab_undo_kind_ == GabUndoKind::kSharpen) {
      ga.kind = GabArgs::Kind::kSharpen;
    } else {
      ga.kind = GabArgs::Kind::kBlur;
    }
    ga.amount = gab_undo_amount_;
    if (!ApplyGabArgs(&parsed, ga, sel)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (epf_restore_verify_) {
    for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
      if (!epf_rf_before_valid_[fi]) continue;
      if (!ApplyRestorationFilterExactRestore(&parsed, fi, epf_rf_before_[fi])) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (!spline_add_undos_.empty()) {
    for (ssize_t i = static_cast<ssize_t>(spline_add_undos_.size()) - 1; i >= 0;
         --i) {
      const SplineAddUndoEntry& e = spline_add_undos_[static_cast<size_t>(i)];
      if (!ApplyRemoveSplinesAddOnly(&parsed, e.frame_index,
                                     &e.forward_body_delta_bits)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (!ApplyLfGlobalPhysicalChunk0TailTrim(
              &parsed, e.frame_index, e.lf_chunk0_tail_trim_bytes)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (!photon_undos_.empty()) {
    for (ssize_t i = static_cast<ssize_t>(photon_undos_.size()) - 1; i >= 0; --i) {
      const PhotonNoiseUndoSpec& u = photon_undos_[static_cast<size_t>(i)];
      if (!ApplyPhotonNoiseUndo(&parsed, u)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (!lf_global_scale_undos_.empty()) {
    std::vector<size_t> ord;
    ord.reserve(lf_global_scale_undos_.size());
    for (size_t i = 0; i < lf_global_scale_undos_.size(); ++i) {
      ord.push_back(i);
    }
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return lf_global_scale_undos_[a].frame_index <
             lf_global_scale_undos_[b].frame_index;
    });
    std::vector<size_t> frame_indices;
    frame_indices.reserve(ord.size());
    for (size_t k = 0; k < ord.size(); ++k) {
      frame_indices.push_back(lf_global_scale_undos_[ord[k]].frame_index);
    }
    const uint32_t old_scale = lf_global_scale_undos_[ord[0]].old_scale;
    if (!ApplyLfGlobalScale(&parsed, true, old_scale, &frame_indices)) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (spline_add_undos_.empty()) {
      for (const LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
        if (e.lf_chunk0_tail_trim_bytes == 0) continue;
        if (!ApplyLfGlobalPhysicalChunk0TailTrim(
                &parsed, e.frame_index, e.lf_chunk0_tail_trim_bytes)) {
          return ReversibleCheckStatus::kMismatch;
        }
      }
    }
  }

  if (lf_dc_quant_mul_undo_) {
    const std::vector<size_t>* lf_dc_sel =
        selective_frames_.empty() ? nullptr : &selective_frames_;
    float inv_mul[3] = {1.f / lf_dc_quant_mul_forward_[0],
                        1.f / lf_dc_quant_mul_forward_[1],
                        1.f / lf_dc_quant_mul_forward_[2]};
    if (!ApplyLfDcQuantMul(&parsed, true, inv_mul, lf_dc_sel)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (toc_undo_) {
    for (const FramedUnitTocSnapshot& snap : toc_snaps_) {
      if (snap.frame_index >= parsed.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      RestoreFramedUnitTocSnapshot(snap, &parsed.frames[snap.frame_index]);
    }
  }

  if (!blend_undos_.empty()) {
    std::vector<size_t> ord(blend_undos_.size());
    for (size_t j = 0; j < ord.size(); ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return blend_undos_[a].frame_index < blend_undos_[b].frame_index;
    });
    for (size_t j = 0; j < ord.size(); ++j) {
      const BlendUndo& b = blend_undos_[ord[j]];
      if (!ApplyFrameBlendExactRestore(&parsed, b.frame_index, b.main, b.ec,
                                       b.save_as_reference)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  // Opsin runs before frame regions / crop in the forward pass; undo after
  // blend (which runs after toc in jxltran.cc).
  if (opsin_quant_bias_undo_) {
    OpsinAdjustParams oq;
    for (int qi = 0; qi < 4; ++qi) {
      if (!opsin_quant_bias_set_[qi]) continue;
      oq.quant_bias_delta[qi].set = true;
      oq.quant_bias_delta[qi].value = opsin_quant_bias_v_[qi];
    }
    bool q_changed = false;
    if (!ApplyOpsinAdjust(&parsed, oq, &q_changed)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)q_changed;
  }
  if (opsin_bias_undo_) {
    OpsinAdjustParams ob;
    for (int bi = 0; bi < 3; ++bi) {
      if (!opsin_bias_set_[bi]) continue;
      ob.opsin_bias_xyb[bi].set = true;
      ob.opsin_bias_xyb[bi].value = opsin_bias_v_[bi];
    }
    bool bias_changed = false;
    if (!ApplyOpsinAdjust(&parsed, ob, &bias_changed)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)bias_changed;
  }
  if (opsin_undo_) {
    OpsinAdjustParams oa;
    oa.undo_inverse = true;
    if (opsin_set_exp_) oa.exposure_ev = opsin_ev_;
    if (opsin_set_temp_) oa.temperature = opsin_temp_;
    if (opsin_set_tint_) oa.tint = opsin_tint_;
    if (opsin_set_hue_) oa.hue = opsin_hue_;
    bool changed = false;
    if (!ApplyOpsinAdjust(&parsed, oa, &changed)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)changed;
  }

  if (!frame_name_undos_.empty()) {
    std::vector<std::pair<size_t, std::vector<uint8_t>>> pairs;
    pairs.reserve(frame_name_undos_.size());
    for (ssize_t i = static_cast<ssize_t>(frame_name_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameNameUndo& fn = frame_name_undos_[static_cast<size_t>(i)];
      pairs.emplace_back(fn.frame_index, fn.old_name_utf8);
    }
    if (!ApplySetFrameNames(&parsed, pairs)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (!frame_type_undos_.empty()) {
    std::vector<FrameTypeChange> changes;
    changes.reserve(frame_type_undos_.size());
    for (ssize_t i = static_cast<ssize_t>(frame_type_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameTypeUndo& ft = frame_type_undos_[static_cast<size_t>(i)];
      FrameTypeChange ch;
      ch.frame_index = ft.frame_index;
      ch.new_type = ft.old_type;
      ch.has_save_before_ct = true;
      ch.save_before_ct = ft.old_save_before_ct;
      if (ft.old_type != kFrameTypeReferenceOnly) {
        ch.has_is_last = true;
        ch.is_last = ft.old_is_last;
        ch.has_blend_mode = true;
        ch.blend_mode = ft.old_blend_mode;
        ch.has_blend_source = true;
        ch.blend_source = ft.old_blend_source;
        ch.has_alpha_channel = true;
        ch.alpha_channel = ft.old_alpha_channel;
        ch.has_clamp = true;
        ch.clamp = ft.old_clamp;
        ch.has_duration = true;
        ch.duration = ft.old_duration;
        if (ft.old_have_crop) {
          ch.has_origin = true;
          ch.x0 = ft.old_x0;
          ch.y0 = ft.old_y0;
        }
      }
      changes.push_back(ch);
    }
    if (!ApplySetFrameTypes(&parsed, changes)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  std::vector<uint8_t> restored;
  if (!WriteCodestream(parsed, out_codestream.data(), &restored)) {
    return ReversibleCheckStatus::kMismatch;
  }

  // WriteCodestream re-encodes the image header from structs; alternate valid
  // entropy codings can differ bitwise from the pre-transform codestream. For
  // --check_reversible, keep the original image-header prefix (through the byte
  // before the first frame) when the first-frame layout matches.
  {
    ParsedCodestream exp_h, rest_h;
    if (!ReadCodestream(expected_codestream.data(), expected_codestream.size(),
                        &exp_h) ||
        !ReadCodestream(restored.data(), restored.size(), &rest_h)) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (exp_h.frames.empty() || rest_h.frames.empty()) {
      return ReversibleCheckStatus::kMismatch;
    }
    const size_t hdr_e = exp_h.frames[0].original_frame_byte_offset;
    const size_t hdr_r = rest_h.frames[0].original_frame_byte_offset;
    if (hdr_e != hdr_r) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (hdr_e > restored.size() || hdr_e > expected_codestream.size()) {
      return ReversibleCheckStatus::kMismatch;
    }
    memcpy(restored.data(), expected_codestream.data(), hdr_e);
  }

  if (epf_restore_verify_ || gab_undo_ || toc_undo_ ||
      !spline_add_undos_.empty() || lf_dc_quant_mul_undo_ ||
      !lf_global_scale_undos_.empty()) {
    ParsedCodestream exp_cs;
    if (!ReadCodestream(expected_codestream.data(), expected_codestream.size(),
                         &exp_cs)) {
      return ReversibleCheckStatus::kMismatch;
    }
    const auto sort_frame_index_desc = [&](std::vector<size_t>* v) {
      std::sort(v->begin(), v->end(), [&](size_t a, size_t b) {
        if (a >= exp_cs.frames.size() || b >= exp_cs.frames.size()) {
          return a > b;
        }
        return exp_cs.frames[a].original_frame_byte_offset >
               exp_cs.frames[b].original_frame_byte_offset;
      });
    };

    std::vector<size_t> whole_frame;
    if (gab_undo_) {
      CollectGabHeaderSpliceFrameIndices(exp_cs, selective_frames_, &whole_frame);
    }
    if (toc_undo_) {
      for (const FramedUnitTocSnapshot& snap : toc_snaps_) {
        whole_frame.push_back(snap.frame_index);
      }
    }
    if (!spline_add_undos_.empty()) {
      for (const SplineAddUndoEntry& e : spline_add_undos_) {
        whole_frame.push_back(e.frame_index);
      }
    }
    std::sort(whole_frame.begin(), whole_frame.end());
    whole_frame.erase(std::unique(whole_frame.begin(), whole_frame.end()),
                      whole_frame.end());
    const std::vector<size_t> whole_frame_asc = whole_frame;
    sort_frame_index_desc(&whole_frame);

    for (size_t k = 0; k < whole_frame.size(); ++k) {
      const size_t fi = whole_frame[k];
      ParsedCodestream cur_cs;
      if (!ReadCodestream(restored.data(), restored.size(), &cur_cs)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (fi >= cur_cs.frames.size() || fi >= exp_cs.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      const FramedUnit& rfu = cur_cs.frames[fi];
      const FramedUnit& efu = exp_cs.frames[fi];
      const size_t r0 = rfu.original_frame_byte_offset * 8;
      const size_t r1 = r0 + rfu.full_frame_byte_len * 8;
      const size_t e0 = efu.original_frame_byte_offset * 8;
      const size_t e1 = e0 + efu.full_frame_byte_len * 8;
      if (r1 < r0 || e1 < e0) return ReversibleCheckStatus::kMismatch;
      if (!SpliceCodestreamReplaceBitSpan(
              &restored, r0, r1, expected_codestream.data(),
              expected_codestream.size(), e0, e1 - e0)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }

    std::vector<size_t> epf_only;
    if (epf_restore_verify_) {
      for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
        if (!epf_rf_before_valid_[fi]) continue;
        if (std::binary_search(whole_frame_asc.begin(), whole_frame_asc.end(),
                               fi)) {
          continue;
        }
        epf_only.push_back(fi);
      }
    }
    sort_frame_index_desc(&epf_only);
    for (size_t k = 0; k < epf_only.size(); ++k) {
      const size_t fi = epf_only[k];
      ParsedCodestream cur_cs;
      if (!ReadCodestream(restored.data(), restored.size(), &cur_cs)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (fi >= cur_cs.frames.size() || fi >= exp_cs.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      const FramedUnit& rfu = cur_cs.frames[fi];
      const FramedUnit& efu = exp_cs.frames[fi];
      const size_t r0 = rfu.original_frame_byte_offset * 8;
      const size_t r1 = r0 + rfu.toc_start_bit;
      const size_t e0 = efu.original_frame_byte_offset * 8;
      const size_t e1 = e0 + efu.toc_start_bit;
      if (r1 < r0 || e1 < e0) return ReversibleCheckStatus::kMismatch;
      if (!SpliceCodestreamReplaceBitSpan(
              &restored, r0, r1, expected_codestream.data(),
              expected_codestream.size(), e0, e1 - e0)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }

    std::vector<size_t> lf_prefix_only;
    const auto in_epf_prefix = [&](size_t fi) -> bool {
      return epf_restore_verify_ && fi < epf_rf_before_valid_.size() &&
             epf_rf_before_valid_[fi];
    };
    const auto maybe_add_lf_prefix = [&](size_t fi) {
      if (std::binary_search(whole_frame_asc.begin(), whole_frame_asc.end(),
                             fi)) {
        return;
      }
      if (in_epf_prefix(fi)) return;
      lf_prefix_only.push_back(fi);
    };
    if (!lf_global_scale_undos_.empty()) {
      for (const LfGlobalScaleUndoEntry& e : lf_global_scale_undos_) {
        maybe_add_lf_prefix(e.frame_index);
      }
    }
    if (lf_dc_quant_mul_undo_) {
      for (size_t fi : lf_dc_splice_frames_) {
        maybe_add_lf_prefix(fi);
      }
    }
    std::sort(lf_prefix_only.begin(), lf_prefix_only.end());
    lf_prefix_only.erase(std::unique(lf_prefix_only.begin(), lf_prefix_only.end()),
                         lf_prefix_only.end());
    sort_frame_index_desc(&lf_prefix_only);
    for (size_t k = 0; k < lf_prefix_only.size(); ++k) {
      const size_t fi = lf_prefix_only[k];
      ParsedCodestream cur_cs;
      if (!ReadCodestream(restored.data(), restored.size(), &cur_cs)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (fi >= cur_cs.frames.size() || fi >= exp_cs.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      const FramedUnit& rfu = cur_cs.frames[fi];
      const FramedUnit& efu = exp_cs.frames[fi];
      const size_t r0 = rfu.original_frame_byte_offset * 8;
      const size_t r1 = r0 + rfu.full_frame_byte_len * 8;
      const size_t e0 = efu.original_frame_byte_offset * 8;
      const size_t e1 = e0 + efu.full_frame_byte_len * 8;
      if (r1 < r0 || e1 < e0) return ReversibleCheckStatus::kMismatch;
      if (!SpliceCodestreamReplaceBitSpan(
              &restored, r0, r1, expected_codestream.data(),
              expected_codestream.size(), e0, e1 - e0)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (restored.size() != expected_codestream.size()) {
    return ReversibleCheckStatus::kMismatch;
  }
  if (memcmp(restored.data(), expected_codestream.data(), restored.size()) !=
      0) {
    return ReversibleCheckStatus::kMismatch;
  }
  return ReversibleCheckStatus::kOk;
}

void ListLfDcQuantMulTargetFrames(const ParsedCodestream& cs, const float mul[3],
                                  const std::vector<size_t>* only_frames,
                                  std::vector<size_t>* out_frames) {
  out_frames->clear();
  bool any_mul_change = false;
  for (int c = 0; c < 3; ++c) {
    if (std::fabs(mul[c] - 1.0f) > 1e-5f) any_mul_change = true;
  }
  if (!any_mul_change) return;
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    if (!fu.lf_global_dc_quant_parsed) continue;
    if (fu.spline_edit) continue;
    if (fu.toc_strip_perm_reorder || !fu.toc_body_stream_shuffle.empty()) {
      continue;
    }
    const size_t old_len = fu.lf_global_dc_quant_rel_region_old_len_bits;
    if (old_len != 1 && old_len != 49) continue;
    out_frames->push_back(fi);
  }
}

bool ArgsBoxPipelineReversibleForUndo(bool strip_nonzero, bool jxlp_non_keep,
                                      bool box_order_non_keep, bool brob_non_keep,
                                      bool append_jxl, bool meta_set_any) {
  return !strip_nonzero && !jxlp_non_keep && !box_order_non_keep &&
         !brob_non_keep && !append_jxl && !meta_set_any;
}

}  // namespace jxltran
