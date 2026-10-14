// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc_group_order.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include "codestream.h"
#include "frame_header.h"
#include "printf_macros.h"

namespace jxltran {

namespace {

static size_t AcGroupIndex(size_t pass, size_t group, size_t num_groups,
                           size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

// Stream-order slot index carrying logical TOC section 0 (LF-global).
static bool StreamSlotOfLogicalLfGlobal(const FramedUnit& fu, size_t* p0) {
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

// |post_edit| matches |original| body layout except LF-global stream slot |p0|,
// which |fu| may have grown/shrunk for pending photon or spline edits (same-run
// metadata updates before WriteCodestream rewrites the body).
static bool ShuffleBodyStreamSizesAsInOriginal(const FramedUnit& fu,
                                               const std::vector<uint32_t>& post_edit,
                                               std::vector<uint32_t>* out) {
  *out = post_edit;
  if (out->empty()) return true;
  size_t p0 = 0;
  if (!StreamSlotOfLogicalLfGlobal(fu, &p0) || p0 >= out->size()) {
    return false;
  }
  int64_t pending = 0;
  if (fu.photon_noise_edit) {
    pending += static_cast<int64_t>(fu.photon_noise_delta_bytes);
  }
  if (fu.lf_global_dc_quant_edit) {
    pending += static_cast<int64_t>(fu.lf_global_dc_quant_delta_bytes);
  }
  if (fu.lf_global_quantizer_edit) {
    pending += static_cast<int64_t>(fu.lf_global_quantizer_delta_bytes);
  }
  if (fu.spline_edit && fu.spline_edit_delta_bits != 0) {
    if (fu.spline_edit_delta_bits % 8 != 0) return false;
    pending += static_cast<int64_t>(fu.spline_edit_delta_bits / 8);
  }
  if (pending == 0) return true;
  const int64_t v = static_cast<int64_t>((*out)[p0]) - pending;
  if (v < 0 || v > static_cast<int64_t>(UINT32_MAX)) return false;
  (*out)[p0] = static_cast<uint32_t>(v);
  return true;
}

static bool InvertStreamPermutation(const std::vector<uint32_t>& perm,
                                    std::vector<size_t>* inv) {
  const size_t n = perm.size();
  inv->assign(n, n);
  for (size_t s = 0; s < n; ++s) {
    if (perm[s] >= n) return false;
    const size_t L = perm[s];
    if ((*inv)[L] != n) return false;
    (*inv)[L] = s;
  }
  for (size_t L = 0; L < n; ++L) {
    if ((*inv)[L] == n) return false;
  }
  return true;
}

static bool StripTocPermutationOnFrame(FramedUnit* fu) {
  fu->toc_body_stream_shuffle.clear();
  fu->toc_body_shuffle_src_sizes.clear();
  if (fu->toc_perm.empty()) {
    return true;
  }
  const size_t n = fu->toc_perm.size();
  if (fu->toc_decoded_sizes.size() != n) {
    fprintf(stderr, "jxltran: TOC permutation/size count mismatch\n");
    return false;
  }
  fu->toc_strip_stream_sizes = fu->toc_decoded_sizes;
  std::vector<size_t> inv;
  if (!InvertStreamPermutation(fu->toc_perm, &inv)) {
    fprintf(stderr, "jxltran: invalid TOC permutation\n");
    return false;
  }
  std::vector<uint32_t> logical_sizes(n);
  for (size_t L = 0; L < n; ++L) {
    logical_sizes[L] = fu->toc_decoded_sizes[inv[L]];
  }
  fu->toc_decoded_sizes = std::move(logical_sizes);
  fu->toc_strip_logical_to_stream = std::move(inv);
  fu->toc_perm.clear();
  fu->toc_bits_before_sizes = 0;
  fu->toc_strip_perm_reorder = true;
  return true;
}

static constexpr double kPi = 3.1415926535897932384626433832795;

// Permutation of stream indices 0..n-1: chunk at old linear index |src| moves
// to stream position perm_move[src] (same convention as libjxl PermuteGroups).
static bool BuildLibjxlMovePermutation(const FrameTocMetrics& tm, int64_t center_x,
                                     int64_t center_y,
                                     const std::vector<size_t>* custom_ac_order,
                                     size_t frame_index,
                                     std::vector<size_t>* perm_move) {
  perm_move->clear();
  const size_t ndc = tm.num_dc_groups;
  const size_t ng = tm.num_groups;
  const size_t np = tm.num_passes;
  if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
    return true;
  }
  if (ng == 1 && np == 1) {
    return true;
  }
  const size_t group_dim = tm.group_dim;
  if (group_dim == 0) {
    return false;
  }

  int64_t imag_cx =
      (center_x >= 0 && static_cast<size_t>(center_x) < tm.xsize)
          ? center_x
          : static_cast<int64_t>(tm.xsize / 2);
  int64_t imag_cy =
      (center_y >= 0 && static_cast<size_t>(center_y) < tm.ysize)
          ? center_y
          : static_cast<int64_t>(tm.ysize / 2);

  const int64_t cx =
      (imag_cx / static_cast<int64_t>(group_dim)) *
          static_cast<int64_t>(group_dim) +
      static_cast<int64_t>(group_dim / 2);
  const int64_t cy =
      (imag_cy / static_cast<int64_t>(group_dim)) *
          static_cast<int64_t>(group_dim) +
      static_cast<int64_t>(group_dim / 2);

  const double direction =
      -std::atan2(static_cast<double>(imag_cy - cy),
                  static_cast<double>(imag_cx - cx));
  const double side_d =
      std::fmod((direction + 5 * kPi / 4), 2 * kPi) * 2 / kPi;
  const int64_t side = static_cast<int64_t>(side_d);

  auto group_center = [&](size_t gid, int64_t* gcx, int64_t* gcy) {
    const size_t gx = gid % tm.xsize_groups;
    const size_t gy = gid / tm.xsize_groups;
    const int64_t x0 = static_cast<int64_t>(gx * group_dim);
    const int64_t y0 = static_cast<int64_t>(gy * group_dim);
    *gcx = x0 + static_cast<int64_t>(group_dim / 2);
    *gcy = y0 + static_cast<int64_t>(group_dim / 2);
  };

  auto distance_key = [&](size_t gid) {
    int64_t gcx, gcy;
    group_center(gid, &gcx, &gcy);
    const int64_t dx = gcx - cx;
    const int64_t dy = gcy - cy;
    const int64_t adx = dx >= 0 ? dx : -dx;
    const int64_t ady = dy >= 0 ? dy : -dy;
    const double angle =
        std::remainder(std::atan2(static_cast<double>(dy), static_cast<double>(dx)) +
                           kPi / 4 + static_cast<double>(side) * (kPi / 2),
                       2 * kPi);
    return std::make_pair(std::max(adx, ady), angle);
  };

  std::vector<size_t> ac_order(ng);
  std::iota(ac_order.begin(), ac_order.end(), 0);
  std::sort(ac_order.begin(), ac_order.end(),
            [&](size_t a, size_t b) { return distance_key(a) < distance_key(b); });

  if (custom_ac_order != nullptr) {
    std::vector<bool> listed(ng, false);
    std::vector<size_t> custom;
    custom.reserve(ng);
    for (size_t gid : *custom_ac_order) {
      if (gid >= ng) {
        fprintf(stderr,
                "jxltran: --group_order_list: group %" PRIuS
                " is out of range for frame %" PRIuS " (%" PRIuS " AC groups, "
                "%" PRIuS "x%" PRIuS "); use --frames=N to reorder only the "
                "frame the list was made for\n",
                gid, frame_index, ng, tm.xsize_groups,
                ng / (tm.xsize_groups ? tm.xsize_groups : 1));
        return false;
      }
      if (listed[gid]) {
        fprintf(stderr,
                "jxltran: --group_order_list: group %" PRIuS " is listed twice\n",
                gid);
        return false;
      }
      listed[gid] = true;
      custom.push_back(gid);
    }
    for (size_t gid : ac_order) {
      if (!listed[gid]) custom.push_back(gid);
    }
    ac_order = std::move(custom);
  }

  std::vector<size_t> inv_ac(ng, 0);
  for (size_t i = 0; i < ac_order.size(); ++i) {
    inv_ac[ac_order[i]] = i;
  }

  perm_move->reserve(ndc + 2 + ng * np);
  for (size_t i = 0; i < ndc + 2; ++i) {
    perm_move->push_back(i);
  }
  for (size_t pass = 0; pass < np; ++pass) {
    const size_t pass_start = perm_move->size();
    for (size_t gid = 0; gid < ng; ++gid) {
      perm_move->push_back(pass_start + inv_ac[gid]);
    }
  }
  if (perm_move->size() != tm.num_toc_entries) {
    fprintf(stderr,
            "jxltran: internal error: center-first permutation size %" PRIuS
            " != toc_entries %" PRIuS "\n",
            perm_move->size(), tm.num_toc_entries);
    return false;
  }
  return true;
}

static bool ApplyCenterFirstOnFrame(FramedUnit* fu, const FrameTocMetrics& tm,
                                    int64_t center_x, int64_t center_y,
                                    const std::vector<size_t>* custom_ac_order,
                                    size_t frame_index, bool* mutated) {
  fu->toc_body_stream_shuffle.clear();
  fu->toc_body_shuffle_src_sizes.clear();

  if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
    return true;
  }
  const size_t n = fu->toc_decoded_sizes.size();
  if (n == 0 || n != tm.num_toc_entries) {
    fprintf(stderr,
            "jxltran: --group_order: TOC size mismatch (got %" PRIuS
            ", expected %" PRIuS ")\n",
            fu->toc_decoded_sizes.size(), tm.num_toc_entries);
    return false;
  }
  if (fu->toc_strip_perm_reorder) {
    fprintf(stderr,
            "jxltran: --group_order cannot combine with another TOC strip on "
            "the same frame.\n");
    return false;
  }

  std::vector<size_t> perm_move;
  if (!BuildLibjxlMovePermutation(tm, center_x, center_y, custom_ac_order,
                                  frame_index, &perm_move)) {
    return false;
  }
  if (perm_move.empty()) {
    return true;
  }

  // |perm_move[L]| = target stream slot for canonical logical section L (LF
  // global, LF groups, HF global, then AC groups in center-first order per
  // pass). Invert: |logical_at_target_stream[new_s]| = L.
  std::vector<size_t> logical_at_target_stream(n, n);
  for (size_t logical = 0; logical < n; ++logical) {
    const size_t d = perm_move[logical];
    if (d >= n || logical_at_target_stream[d] != n) {
      fprintf(stderr, "jxltran: --group_order: invalid move permutation\n");
      return false;
    }
    logical_at_target_stream[d] = logical;
  }

  const std::vector<uint32_t> src_sizes = fu->toc_decoded_sizes;
  const std::vector<uint32_t> old_perm = fu->toc_perm;
  auto old_stream_for_logical = [&](size_t logical) -> size_t {
    if (old_perm.empty()) {
      return logical;
    }
    for (size_t s = 0; s < old_perm.size(); ++s) {
      if (static_cast<size_t>(old_perm[s]) == logical) {
        return s;
      }
    }
    return n;
  };

  std::vector<size_t> shuffle(n, n);
  bool identity_shuffle = true;
  for (size_t new_s = 0; new_s < n; ++new_s) {
    const size_t logical = logical_at_target_stream[new_s];
    if (logical >= n) {
      fprintf(stderr, "jxltran: --group_order: invalid move permutation\n");
      return false;
    }
    const size_t old_s = old_stream_for_logical(logical);
    if (old_s >= n) {
      fprintf(stderr,
              "jxltran: --group_order: could not map logical %" PRIuS
              " to a stream slot\n",
              logical);
      return false;
    }
    shuffle[new_s] = old_s;
    if (old_s != new_s) {
      identity_shuffle = false;
    }
  }
  if (identity_shuffle) {
    return true;
  }

  std::vector<uint32_t> new_perm(n, 0);
  std::vector<uint32_t> new_sizes(n, 0);
  for (size_t new_s = 0; new_s < n; ++new_s) {
    const size_t logical = logical_at_target_stream[new_s];
    const size_t old_s = shuffle[new_s];
    new_sizes[new_s] = src_sizes[old_s];
    new_perm[new_s] = static_cast<uint32_t>(logical);
  }

  fu->toc_decoded_sizes = std::move(new_sizes);
  fu->toc_perm = std::move(new_perm);
  fu->toc_bits_before_sizes = 0;
  fu->toc_strip_perm_reorder = false;
  fu->toc_strip_stream_sizes.clear();
  fu->toc_strip_logical_to_stream.clear();

  if (!ShuffleBodyStreamSizesAsInOriginal(*fu, src_sizes,
                                           &fu->toc_body_shuffle_src_sizes)) {
    fprintf(stderr,
            "jxltran: --group_order: could not derive stream chunk sizes from "
            "source codestream (pending photon/spline LF-global edit?)\n");
    return false;
  }
  fu->toc_body_stream_shuffle = std::move(shuffle);
  if (mutated != nullptr) {
    *mutated = true;
  }
  return true;
}

}  // namespace

bool TocStreamOrderLooksProgressive(const FramedUnit& fu,
                                    const FrameTocMetrics& tm) {
  if (fu.toc_decoded_sizes.empty()) return true;
  if (tm.num_toc_entries == 0) return true;
  if (fu.toc_decoded_sizes.size() != tm.num_toc_entries) return true;
  const size_t n = fu.toc_decoded_sizes.size();
  if (!fu.toc_perm.empty() && fu.toc_perm.size() != n) return true;

  const size_t ndc = tm.num_dc_groups;
  if (ndc >= 63) return true;
  const size_t first_hf_spatial =
      AcGroupIndex(0, 0, tm.num_groups, ndc);
  const uint64_t need_mask = (1ull << (ndc + 1)) - 1ull;
  uint64_t seen = 0;
  for (size_t s = 0; s < n; ++s) {
    const size_t L =
        fu.toc_perm.empty()
            ? s
            : (s < fu.toc_perm.size() ? static_cast<size_t>(fu.toc_perm[s]) : s);
    if (L <= ndc) {
      seen |= 1ull << L;
    } else if (L >= first_hf_spatial) {
      if ((seen & need_mask) != need_mask) {
        return false;
      }
    }
  }
  return true;
}

bool ApplyTocGroupOrder(ParsedCodestream* cs, TocGroupOrderCli order,
                        int64_t center_x, int64_t center_y,
                        const std::vector<size_t>* custom_ac_order,
                        const std::vector<size_t>* only_frames,
                        bool* any_mutation) {
  if (any_mutation != nullptr) *any_mutation = false;
  if (order == TocGroupOrderCli::kKeep) return true;

  const uint32_t cw = cs->image.size.width;
  const uint32_t ch = cs->image.size.height;

  const auto want = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;

    FrameTocMetrics tm;
    ComputeFrameTocMetrics(fu.frame, cs->image.metadata, cw, ch, &tm);
    if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
      continue;
    }

    if (order == TocGroupOrderCli::kCenterFirst ||
        order == TocGroupOrderCli::kCustom) {
      const std::vector<size_t>* custom =
          order == TocGroupOrderCli::kCustom ? custom_ac_order : nullptr;
      bool m = false;
      if (!ApplyCenterFirstOnFrame(&fu, tm, center_x, center_y, custom, fi,
                                   &m)) {
        return false;
      }
      if (m && any_mutation != nullptr) {
        *any_mutation = true;
      }
      continue;
    }

    if (order == TocGroupOrderCli::kCanonical) {
      if (!fu.toc_perm.empty() && any_mutation != nullptr) {
        *any_mutation = true;
      }
      if (!StripTocPermutationOnFrame(&fu)) return false;
      continue;
    }
    if (order == TocGroupOrderCli::kProgressive) {
      if (!TocStreamOrderLooksProgressive(fu, tm)) {
        if (!fu.toc_perm.empty() && any_mutation != nullptr) {
          *any_mutation = true;
        }
        if (!StripTocPermutationOnFrame(&fu)) return false;
      }
    }
  }
  return true;
}

void CopyFramedUnitTocSnapshot(const FramedUnit& fu, FramedUnitTocSnapshot* snap) {
  snap->toc_decoded_sizes = fu.toc_decoded_sizes;
  snap->toc_perm = fu.toc_perm;
  snap->toc_bits_before_sizes = fu.toc_bits_before_sizes;
  snap->toc_strip_perm_reorder = fu.toc_strip_perm_reorder;
  snap->toc_strip_stream_sizes = fu.toc_strip_stream_sizes;
  snap->toc_strip_logical_to_stream = fu.toc_strip_logical_to_stream;
  snap->toc_body_stream_shuffle = fu.toc_body_stream_shuffle;
  snap->toc_body_shuffle_src_sizes = fu.toc_body_shuffle_src_sizes;
}

void RestoreFramedUnitTocSnapshot(const FramedUnitTocSnapshot& snap, FramedUnit* fu) {
  fu->toc_decoded_sizes = snap.toc_decoded_sizes;
  fu->toc_perm = snap.toc_perm;
  fu->toc_bits_before_sizes = snap.toc_bits_before_sizes;
  fu->toc_strip_perm_reorder = snap.toc_strip_perm_reorder;
  fu->toc_strip_stream_sizes = snap.toc_strip_stream_sizes;
  fu->toc_strip_logical_to_stream = snap.toc_strip_logical_to_stream;
  fu->toc_body_stream_shuffle = snap.toc_body_stream_shuffle;
  fu->toc_body_shuffle_src_sizes = snap.toc_body_shuffle_src_sizes;
}

bool CaptureTocSnapshotsBeforeGroupOrder(const ParsedCodestream& cs,
                                         const std::vector<size_t>* only_frames,
                                         std::vector<FramedUnitTocSnapshot>* out_snaps) {
  out_snaps->clear();
  const uint32_t cw = cs.image.size.width;
  const uint32_t ch = cs.image.size.height;
  const auto want = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    FrameTocMetrics tm;
    ComputeFrameTocMetrics(fu.frame, cs.image.metadata, cw, ch, &tm);
    if (tm.all_in_one_section || tm.num_toc_entries <= 1) continue;
    FramedUnitTocSnapshot snap;
    snap.frame_index = fi;
    CopyFramedUnitTocSnapshot(fu, &snap);
    out_snaps->push_back(std::move(snap));
  }
  return true;
}

}  // namespace jxltran
