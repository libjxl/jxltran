// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JXLTRAN_SRC_REVERSIBLE_UNDO_H_
#define TOOLS_JXLTRAN_SRC_REVERSIBLE_UNDO_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "frame_header.h"
#include "operations.h"
#include "toc_group_order.h"

namespace jxltran {

struct ParsedCodestream;
struct FrameBlendOverride;

enum class ReversibleCheckStatus {
  kNotReversibleSkipped,
  kOk,
  kMismatch,
};

// Collects inverse codestream steps for a single jxltran forward run.
// Box-level options that drop or rewrite non-codestream container bytes (strip,
// jxlp remux, box order, brob, append, metadata set) disable the undo line and
// skip --check_reversible; pure --container remux does not.
class UndoRecorder {
 public:
  void Reset();

  void SetBoxPipelineNotReversible(const char* reason);
  void SetCodestreamNotReversible(const char* reason);

  bool box_pipeline_reversible() const { return box_pipeline_ok_; }
  bool codestream_undo_supported() const { return codestream_ok_; }
  const char* block_reason() const { return block_reason_; }

  // --- Capture helpers (call from jxltran transform in forward order) ---

  void CapturePreHeader(const ParsedCodestream& parsed);
  void NoteHeaderChanges(bool set_abs_orient_nonzero, bool set_rel_nonzero,
                         uint32_t orient_after, bool set_main_bit_depth_nonzero,
                         bool set_loops, bool set_tps, bool set_intensity_target);
  void SetSelectiveFramesCopy(const std::vector<size_t>& indices);

  // Call immediately before ApplyHeaderMod when |patches| is non-empty; records
  // per-index ExtraChannelInfo snapshots from the pre-apply codestream.
  void NoteExtraChannelHeaderBeforeApply(
      const ParsedCodestream& pre,
      const std::vector<ExtraChannelHeaderPatch>& patches);

  void NoteCropApplied(uint32_t old_canvas_w, uint32_t old_canvas_h,
                       int32_t storage_dx, int32_t storage_dy,
                       uint32_t new_canvas_w, uint32_t new_canvas_h);
  void NoteGabBlur(float amount);
  void NoteGabSharpen(float amount);
  void NoteEpfAmplitudeScale(float factor);
  void CaptureRestorationBeforeEpfOps(const ParsedCodestream& cs,
                                      const std::vector<size_t>* only_frames);
  // |mask|: bit 1 = --set-epf-iters, 2 = --set-epf-amplitude-scale, 4 =
  // --set-epf-uniformity (used for undo argv; verify always restores snapshots).
  void FinalizeEpfRestorationUndo(uint32_t mask);
  void NoteOpsinAdjust(bool exp, bool temp, bool tint, bool hue, float ev,
                       float temperature, float tint_v, float hue_v);
  void NoteOpsinBiasAdjust(bool sx, bool sy, bool sb, float vx, float vy,
                           float vb);
  void NoteOpsinQuantBiasAdjust(bool s0, bool s1, bool s2, bool s3, float v0,
                                float v1, float v2, float v3);

  // |orient_before_header| is the EXIF tag before --set-orientation in this
  // run (same basis as --crop / --set-frame-region).
  bool CaptureFrameRegionBeforeMove(const ParsedCodestream& parsed,
                                    size_t frame_index,
                                    uint32_t orient_before_header);
  void NoteFrameNameBeforeChange(const ParsedCodestream& parsed,
                                 size_t frame_index);
  // Records the header fields --set-frame-types would drop, so the undo can
  // pass them back. Call before ApplySetFrameTypes.
  void NoteFrameTypeBeforeChange(const ParsedCodestream& parsed,
                                 size_t frame_index);

  // |entries| lists (codestream frame index, forward |spline_edit_delta_bits|)
  // for each insert, in the same order as |ApplySplinesFromFile| applied blocks.
  void NoteSplinesAddedOnFrames(
      const std::vector<std::pair<size_t, int64_t>>& entries);

  // After OUTPUT is written, compares LF-global physical chunk0 vs the
  // pre-transform codestream and fills per-entry tail trim counts (0–255)
  // for spline-add and --lf-global-scale undos.
  void FinalizeSplineLfGlobalChunk0TailTrim(const uint8_t* ref_codestream,
                                            size_t ref_size,
                                            const uint8_t* out_codestream,
                                            size_t out_size);

  // Call immediately before ApplyLfGlobalScale when |file_out| is set.
  void CaptureLfGlobalScaleBeforeApply(const ParsedCodestream& cs,
                                       const std::vector<size_t>* only_frames,
                                       uint32_t forward_new_scale);
  void CapturePhotonNoiseBeforeApply(const ParsedCodestream& cs,
                                     const std::vector<size_t>* only_frames);
  // Call immediately after a successful photon pass.
  void FinalizePhotonNoiseAfterApply(const ParsedCodestream& cs);

  // Call after a successful forward --lf-dc-quant-mul (same mx,my,mb as CLI).
  // |splice_frames| from ListLfDcQuantMulTargetFrames(cs, mul, only, ...) on
  // the pre-apply codestream.
  void NoteLfDcQuantMulForward(const std::vector<size_t>& splice_frames, float mx,
                               float my, float mb);

  // Captures blending fields for each frame mentioned in |overrides| (first
  // mention per frame index only), before |ApplyFrameBlendOverrides|.
  void NoteFrameBlendBeforeOverrides(
      const ParsedCodestream& parsed,
      const std::vector<FrameBlendOverride>& overrides);

  // Full bijective --keep-listed-frames reorder (every frame index kept once).
  void NoteKeepReorder(KeepReorderUndoSpec spec);

  bool has_any_undo_step() const { return has_any_step_; }

  // argv0: program path as invoked; out_jxl / restored_jxl paths are shell-quoted.
  bool BuildUndoCommandLine(const char* argv0, const char* out_jxl,
                            const char* restored_jxl,
                            std::string* out_line) const;

  // Applies inverse ops to |out_codestream| (bytes of the written codestream)
  // and compares the result to |expected_codestream| (byte-for-byte).
  ReversibleCheckStatus VerifyRoundtripCodestream(
      const std::vector<uint8_t>& expected_codestream,
      const std::vector<uint8_t>& out_codestream) const;

  void FinalizePipelineCompatibility();

  // Snapshots TOC-bearing fields before ApplyTocGroupOrder; call
  // FinalizeTocGroupOrderAfterApply(toc_changed) immediately after.
  void CaptureTocBeforeGroupOrder(const ParsedCodestream& cs,
                                  const std::vector<size_t>* only_frames);
  void FinalizeTocGroupOrderAfterApply(bool toc_changed);

 private:
  bool box_pipeline_ok_ = true;
  bool codestream_ok_ = true;
  const char* block_reason_ = nullptr;

  bool has_any_step_ = false;

  bool toc_undo_ = false;
  std::vector<FramedUnitTocSnapshot> toc_snaps_;

  // Pre-header snapshot (defaults if absent in bitstream).
  uint32_t orient_before_ = 1;
  BitDepth main_bit_depth_before_{};
  bool have_animation_before_ = false;
  uint32_t num_loops_before_ = 0;
  uint32_t tps_num_before_ = 1;
  uint32_t tps_den_before_ = 1;

  bool crop_undo_ = false;
  uint32_t crop_old_w_ = 0;
  uint32_t crop_old_h_ = 0;
  int32_t crop_storage_dx_ = 0;
  int32_t crop_storage_dy_ = 0;
  uint32_t crop_new_w_ = 0;
  uint32_t crop_new_h_ = 0;

  bool gab_undo_ = false;
  enum class GabUndoKind { kBlur, kSharpen };
  GabUndoKind gab_undo_kind_ = GabUndoKind::kBlur;
  float gab_undo_amount_ = 0.f;

  bool epf_amp_undo_ = false;
  float epf_amp_forward_factor_ = 1.f;

  std::vector<bool> epf_rf_before_valid_;
  std::vector<bool> epf_rf_snap_modular_;
  std::vector<ParsedRestorationFilter> epf_rf_before_;
  bool epf_restore_verify_ = false;
  uint32_t epf_undo_mask_ = 0;

  bool opsin_undo_ = false;
  bool opsin_set_exp_ = false;
  bool opsin_set_temp_ = false;
  bool opsin_set_tint_ = false;
  bool opsin_set_hue_ = false;
  float opsin_ev_ = 0.f;
  float opsin_temp_ = 0.f;
  float opsin_tint_ = 0.f;
  float opsin_hue_ = 0.f;

  bool opsin_bias_undo_ = false;
  bool opsin_bias_set_[3] = {false, false, false};
  float opsin_bias_v_[3] = {0.f, 0.f, 0.f};

  bool opsin_quant_bias_undo_ = false;
  bool opsin_quant_bias_set_[4] = {false, false, false, false};
  float opsin_quant_bias_v_[4] = {0.f, 0.f, 0.f, 0.f};

  struct FrameRegionUndo {
    size_t frame_index = 0;
    uint32_t orient_before_header = 1;
    int64_t old_display_x = 0;
    int64_t old_display_y = 0;
    uint64_t display_w = 0;
    uint64_t display_h = 0;
  };
  std::vector<FrameRegionUndo> frame_region_undos_;

  struct FrameNameUndo {
    size_t frame_index = 0;
    std::vector<uint8_t> old_name_utf8;
  };
  std::vector<FrameNameUndo> frame_name_undos_;

  struct FrameTypeUndo {
    size_t frame_index = 0;
    uint32_t old_type = 0;
    bool old_is_last = false;
    bool old_save_before_ct = false;
    uint32_t old_blend_mode = 0;
    uint32_t old_blend_source = 0;
    uint32_t old_alpha_channel = 0;
    bool old_clamp = false;
    uint32_t old_duration = 0;
    bool old_have_crop = false;
    int32_t old_x0 = 0;
    int32_t old_y0 = 0;
  };
  std::vector<FrameTypeUndo> frame_type_undos_;

  bool undo_orientation_ = false;
  bool undo_main_bit_depth_ = false;
  bool undo_loops_ = false;
  bool undo_tps_ = false;
  bool undo_intensity_ = false;
  ToneMapping tone_mapping_before_{};
  bool image_metadata_all_default_before_ = false;

  bool undo_extra_channels_ = false;
  std::vector<std::pair<size_t, ExtraChannelInfo>> extra_channel_undo_;

  std::vector<size_t> selective_frames_;

  struct LfGlobalScaleUndoEntry {
    size_t frame_index = 0;
    uint32_t old_scale = 0;
    uint8_t lf_chunk0_tail_trim_bytes = 0;
  };
  std::vector<LfGlobalScaleUndoEntry> lf_global_scale_undos_;

  struct SplineAddUndoEntry {
    size_t frame_index = 0;
    int64_t forward_body_delta_bits = 0;
    uint8_t lf_chunk0_tail_trim_bytes = 0;
  };
  std::vector<SplineAddUndoEntry> spline_add_undos_;

  struct PhotonBeforeSnap {
    bool had_noise = false;
    bool raw_valid = false;
    std::array<uint8_t, 10> raw_bytes{};
  };
  std::vector<PhotonBeforeSnap> photon_before_by_frame_;
  std::vector<PhotonNoiseUndoSpec> photon_undos_;

  bool lf_dc_quant_mul_undo_ = false;
  float lf_dc_quant_mul_forward_[3] = {1.f, 1.f, 1.f};
  std::vector<size_t> lf_dc_splice_frames_;

  struct BlendUndo {
    size_t frame_index = 0;
    uint32_t num_extra = 0;
    bool is_last = true;
    FrameBlendingInfo main{};
    std::vector<FrameBlendingInfo> ec;
    uint32_t save_as_reference = 0;
  };
  std::vector<BlendUndo> blend_undos_;

  bool keep_reorder_undo_ = false;
  KeepReorderUndoSpec keep_reorder_spec_;

  void BlockCs(const char* reason);
};

// True when box-level options do not drop or rewrite container bytes in a way
// that --check_reversible cannot validate from primary codestream equality alone.
// (--container no/yes/if-needed alone is OK: primary codestream bytes match.)
// --append-dummy-tail is excluded: it only extends the primary codestream before
// transforms, same as other codestream edits. --append-jxl is not: it merges
// arbitrary external codestream bytes.
bool ArgsBoxPipelineReversibleForUndo(bool strip_nonzero, bool jxlp_non_keep,
                                      bool box_order_non_keep, bool brob_non_keep,
                                      bool append_jxl, bool meta_set_any);

// Frames |ApplyLfDcQuantMul| would touch (call on pre-apply |cs|).
void ListLfDcQuantMulTargetFrames(const ParsedCodestream& cs, const float mul[3],
                                  const std::vector<size_t>* only_frames,
                                  std::vector<size_t>* out_frames);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_SRC_REVERSIBLE_UNDO_H_
