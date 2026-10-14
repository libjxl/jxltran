// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Frame-level TOC permutation / group-order helpers (mirrors a subset of
// cjxl --group_order / --center_x / --center_y).

#ifndef TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_
#define TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

struct ParsedCodestream;
struct FramedUnit;
struct FrameTocMetrics;

enum class TocGroupOrderCli {
  kKeep,
  kCanonical,    // 0: strip TOC permutation (logical stream order)
  kCenterFirst,  // 1: center-first AC group order (matches cjxl --group_order=1)
  kProgressive,  // normalize only when HF data appears before all LF groups
  kCustom,       // caller-supplied AC group order; the rest follow center-first
};

bool TocStreamOrderLooksProgressive(const FramedUnit& fu,
                                    const FrameTocMetrics& tm);

struct FramedUnitTocSnapshot {
  size_t frame_index = 0;
  std::vector<uint32_t> toc_decoded_sizes;
  std::vector<uint32_t> toc_perm;
  size_t toc_bits_before_sizes = 0;
  bool toc_strip_perm_reorder = false;
  std::vector<uint32_t> toc_strip_stream_sizes;
  std::vector<size_t> toc_strip_logical_to_stream;
  std::vector<size_t> toc_body_stream_shuffle;
  std::vector<uint32_t> toc_body_shuffle_src_sizes;
};

void CopyFramedUnitTocSnapshot(const FramedUnit& fu, FramedUnitTocSnapshot* snap);
void RestoreFramedUnitTocSnapshot(const FramedUnitTocSnapshot& snap, FramedUnit* fu);

// Eligible VarDCT / skip-progressive frames with multi-entry TOC (same filter as
// ApplyTocGroupOrder). Populates |out_snaps| in codestream frame order.
bool CaptureTocSnapshotsBeforeGroupOrder(const ParsedCodestream& cs,
                                         const std::vector<size_t>* only_frames,
                                         std::vector<FramedUnitTocSnapshot>* out_snaps);

// Applies |order| to eligible frames (optionally restricted by |only_frames|).
// kCanonical strips permutation; kProgressive strips only when not
// TocStreamOrderLooksProgressive. kCenterFirst reorders stream-order TOC sections
// to libjxl center-first layout (see |center_x| / |center_y|, default image center).
// kCustom puts the AC groups listed in |custom_ac_order| (raster group ids,
// same order for every pass) first and the remaining ones after them in
// center-first order; LF global, LF groups and HF global keep their canonical
// place up front so the frame stays progressively decodable.
// When |any_mutation| is non-null, sets *any_mutation true if any frame was
// modified.
bool ApplyTocGroupOrder(ParsedCodestream* cs, TocGroupOrderCli order,
                        int64_t center_x, int64_t center_y,
                        const std::vector<size_t>* custom_ac_order,
                        const std::vector<size_t>* only_frames,
                        bool* any_mutation);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_
