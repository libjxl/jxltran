// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "codestream.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bits.h"
#include "frame_header.h"
#include "lfglobal.h"
#include "printf_macros.h"
#include "toc.h"
#include "trace.h"

namespace jxltran {

namespace {

void AppendNoise80Bits(BitWriter* bw, const std::array<uint8_t, 10>& nb) {
  for (size_t i = 0; i < 80; ++i) {
    bw->WriteBits(1, (nb[i >> 3] >> (i & 7)) & 1);
  }
}

// Splice photon-noise 80 bits inside the LF-global section bytes. |noise_rel| is
// the bit index within |chunk| of the LUT (or insertion point when adding).
// |db| is photon_noise_delta_bytes (-10 remove, 0 replace, +10 insert).
static bool SpliceLfPhotonNoise(const std::vector<uint8_t>& chunk,
                                size_t noise_rel, int64_t db,
                                const std::array<uint8_t, 10>& new_lut,
                                std::vector<uint8_t>* out) {
  const size_t chunk_bits = chunk.size() * 8;
  if (db < 0) {
    if (noise_rel + 80 > chunk_bits) return false;
  } else if (db == 0) {
    if (noise_rel + 80 > chunk_bits) return false;
  } else {
    if (noise_rel > chunk_bits) return false;
  }
  BitWriter bw;
  bw.AppendBitsRange(chunk.data(), 0, noise_rel);
  if (db < 0) {
    bw.AppendBitsRange(chunk.data(), noise_rel + 80,
                       chunk_bits - noise_rel - 80);
  } else if (db == 0) {
    AppendNoise80Bits(&bw, new_lut);
    bw.AppendBitsRange(chunk.data(), noise_rel + 80,
                       chunk_bits - noise_rel - 80);
  } else {
    AppendNoise80Bits(&bw, new_lut);
    bw.AppendBitsRange(chunk.data(), noise_rel, chunk_bits - noise_rel);
  }
  if ((bw.bit_pos() & 7u) != 0) return false;
  *out = bw.TakeBytes();
  return true;
}

static size_t CodestreamUsedBytes(const ParsedCodestream& cs) {
  size_t n = 0;
  for (const FramedUnit& fu : cs.frames) {
    const size_t e = fu.original_frame_byte_offset + fu.full_frame_byte_len;
    if (e > n) n = e;
  }
  return n;
}

// Byte offset of logical LF/DC-global section within the frame body, and its
// byte length (first TOC entry).
static bool Section0BodySpan(const FramedUnit& fu, size_t* sec0_off,
                             uint32_t* s0_bytes) {
  if (fu.toc_decoded_sizes.empty()) return false;
  size_t p0 = 0;
  if (!fu.toc_perm.empty()) {
    bool found = false;
    for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
      if (fu.toc_perm[p] == 0) {
        p0 = p;
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  if (p0 >= fu.toc_decoded_sizes.size()) return false;
  size_t off = 0;
  for (size_t q = 0; q < p0; ++q) {
    off += fu.toc_decoded_sizes[q];
  }
  *sec0_off = off;
  *s0_bytes = fu.toc_decoded_sizes[p0];
  return true;
}

static size_t PhotonNoiseRelInLfSection(const FramedUnit& fu,
                                      size_t body_src_abs_bit) {
  if (fu.lf_global_noise_lut_abs_valid) {
    size_t sec0_off = 0;
    uint32_t s0 = 0;
    if (Section0BodySpan(fu, &sec0_off, &s0)) {
      const size_t lf0 = body_src_abs_bit + sec0_off * 8;
      if (fu.lf_global_noise_lut_abs_bit >= lf0) {
        return fu.lf_global_noise_lut_abs_bit - lf0;
      }
    }
    return fu.lf_global_noise_lut_rel_bit;
  }
  return fu.lf_global_noise_lut_rel_bit;
}

// |noise_rel|, |spline_start|, |spline_end_excl| share one coordinate system
// (LF-bundle-relative bits for permuted spline rebuild; chunk0-relative for the
// simple-TOC inline path). Noise must lie entirely before the spline span or
// entirely at/after spline_end_excl (noise starts where the parsed bundle ends).
static bool NoiseRelAfterSplineSplice(size_t noise_rel, size_t spline_start,
                                      size_t spline_end_excl, size_t new_mid_bits,
                                      size_t* out_rel) {
  if (noise_rel + 80 <= spline_start) {
    *out_rel = noise_rel;
    return true;
  }
  if (noise_rel >= spline_end_excl) {
    *out_rel =
        noise_rel + new_mid_bits - (spline_end_excl - spline_start);
    return true;
  }
  return false;
}

static int64_t PhotonTocChunk0AdjustBytes(const FramedUnit& fu) {
  return fu.photon_noise_edit ? static_cast<int64_t>(fu.photon_noise_delta_bytes)
                              : 0;
}

static int64_t DcQuantTocChunk0AdjustBytes(const FramedUnit& fu) {
  return fu.lf_global_dc_quant_edit
             ? static_cast<int64_t>(fu.lf_global_dc_quant_delta_bytes)
             : 0;
}

static int64_t QuantizerTocChunk0AdjustBytes(const FramedUnit& fu) {
  return fu.lf_global_quantizer_edit
             ? static_cast<int64_t>(fu.lf_global_quantizer_delta_bytes)
             : 0;
}

static int64_t LfChunk0PhotonDcExtraAdjustBytes(const FramedUnit& fu) {
  return PhotonTocChunk0AdjustBytes(fu) + DcQuantTocChunk0AdjustBytes(fu) +
         QuantizerTocChunk0AdjustBytes(fu);
}

// Replace |old_len_bits| at chunk bit |rel_start| with |new_len_bits| from
// |new_bits| (LSB-first packing). When |append_tail_zero_bits| > 0, writes that
// many 0 bits after the preserved chunk tail so the result is byte-aligned
// (used for VarDCT quantizer bundle when the bit-length change is not a
// multiple of 8).
static bool SpliceLfDcQuantRegion(const std::vector<uint8_t>& chunk,
                                  size_t rel_start, size_t old_len_bits,
                                  const std::vector<uint8_t>& new_bits,
                                  size_t new_len_bits,
                                  size_t append_tail_zero_bits,
                                  std::vector<uint8_t>* out) {
  const size_t chunk_bits = chunk.size() * 8;
  if (rel_start + old_len_bits > chunk_bits) return false;
  if (new_len_bits > new_bits.size() * 8u) return false;
  BitWriter bw;
  bw.AppendBitsRange(chunk.data(), 0, rel_start);
  for (size_t i = 0; i < new_len_bits; ++i) {
    bw.WriteBits(1, (new_bits[i >> 3] >> (i & 7)) & 1);
  }
  bw.AppendBitsRange(chunk.data(), rel_start + old_len_bits,
                     chunk_bits - rel_start - old_len_bits);
  for (size_t z = 0; z < append_tail_zero_bits; ++z) {
    bw.WriteBits(1, 0);
  }
  if ((bw.bit_pos() & 7u) != 0u) return false;
  *out = bw.TakeBytes();
  return true;
}

// |dc_rel| is chunk0-relative (before any photon splice on this chunk).
static size_t DcQuantRelAfterPhotonSplice(const FramedUnit& fu, size_t dc_rel) {
  if (!fu.photon_noise_edit) return dc_rel;
  const int64_t pdb_b = static_cast<int64_t>(fu.photon_noise_delta_bytes);
  const size_t nr = fu.lf_global_noise_lut_rel_bit;
  if (pdb_b == 10) {
    return (dc_rel >= nr) ? dc_rel + 80u : dc_rel;
  }
  if (pdb_b == -10) {
    if (dc_rel >= nr + 80u) return dc_rel - 80u;
    return dc_rel;
  }
  return dc_rel;
}

// perm[s] = logical TOC index carried in stream-order slot s.
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

static bool InversePermutation(const std::vector<uint32_t>& perm,
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

static bool ExtractStreamOrderBodyChunks(const uint8_t* orig, size_t orig_size,
                                         size_t body_src_abs_bit,
                                         const std::vector<uint32_t>& old_stream_sizes,
                                         std::vector<std::vector<uint8_t>>* chunks_out) {
  BitReader br(orig, orig_size);
  if (body_src_abs_bit > br.size_bits()) return false;
  br.SkipBits(body_src_abs_bit);
  chunks_out->clear();
  for (uint32_t sz : old_stream_sizes) {
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    for (size_t bi = 0; bi < static_cast<size_t>(sz) * 8u; ++bi) {
      if (!br.ok()) return false;
      const uint32_t bit = br.ReadBits(1);
      buf[bi >> 3] |= static_cast<uint8_t>(bit << (bi & 7));
    }
    chunks_out->push_back(std::move(buf));
  }
  return br.ok();
}

static bool BuildLfChunk0PhotonDcBody(const FramedUnit& fu, const uint8_t* orig,
                                      size_t orig_size, size_t body_src_abs_bit,
                                      std::vector<uint8_t>* out) {
  if (fu.toc_decoded_sizes.empty()) return false;
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0)) return false;
  const size_t n = fu.toc_decoded_sizes.size();
  const int64_t photon_db = PhotonTocChunk0AdjustBytes(fu);
  const int64_t sub = LfChunk0PhotonDcExtraAdjustBytes(fu);
  std::vector<uint32_t> old_stream = fu.toc_decoded_sizes;
  const int64_t old_p0 = static_cast<int64_t>(old_stream[p0]) - sub;
  if (old_p0 < 0 || old_p0 > static_cast<int64_t>(UINT32_MAX)) return false;
  old_stream[p0] = static_cast<uint32_t>(old_p0);
  uint64_t sum_old = 0;
  for (uint32_t sz : old_stream) {
    sum_old += static_cast<uint64_t>(sz);
  }
  const int64_t new_bits = static_cast<int64_t>(fu.body_bit_length);
  if (new_bits < sub * 8) return false;
  const uint64_t old_body_bytes =
      static_cast<uint64_t>((new_bits - sub * 8) / 8);
  if (sum_old != old_body_bytes) return false;
  std::vector<std::vector<uint8_t>> chunks;
  if (!ExtractStreamOrderBodyChunks(orig, orig_size, body_src_abs_bit, old_stream,
                                    &chunks)) {
    fprintf(stderr,
            "jxltran: LF DC quant / photon: failed to read original frame body\n");
    return false;
  }
  if (chunks.size() != n) return false;
  std::vector<uint8_t> c0 = chunks[p0];
  if (fu.photon_noise_edit) {
    const size_t noise_rel = PhotonNoiseRelInLfSection(fu, body_src_abs_bit);
    std::vector<uint8_t> patched;
    if (!SpliceLfPhotonNoise(c0, noise_rel, photon_db, fu.photon_noise_new_bytes,
                             &patched)) {
      fprintf(stderr,
              "jxltran: LF DC quant / photon: photon noise splice in chunk0 "
              "failed\n");
      return false;
    }
    c0 = std::move(patched);
  }
  if (fu.lf_global_dc_quant_edit) {
    const size_t dc_rel =
        DcQuantRelAfterPhotonSplice(fu, fu.lf_global_dc_quant_rel_region_start);
    std::vector<uint8_t> patched2;
    if (!SpliceLfDcQuantRegion(
            c0, dc_rel, fu.lf_global_dc_quant_rel_region_old_len_bits,
            fu.lf_global_dc_quant_new_region_bits,
            fu.lf_global_dc_quant_new_region_len_bits, 0, &patched2)) {
      fprintf(stderr,
              "jxltran: LF DC quant: splice in LF-global chunk0 failed\n");
      return false;
    }
    c0 = std::move(patched2);
  }
  if (fu.lf_global_quantizer_edit) {
    const size_t dc_st =
        DcQuantRelAfterPhotonSplice(fu, fu.lf_global_dc_quant_rel_region_start);
    const size_t dc_len_bits =
        fu.lf_global_dc_quant_edit ? fu.lf_global_dc_quant_new_region_len_bits
                                   : fu.lf_global_dc_quant_rel_region_old_len_bits;
    const size_t q_rel = dc_st + dc_len_bits;
    std::vector<uint8_t> patched3;
    if (!SpliceLfDcQuantRegion(
            c0, q_rel, fu.lf_global_quantizer_rel_region_old_len_bits,
            fu.lf_global_quantizer_new_region_bits,
            fu.lf_global_quantizer_new_region_len_bits,
            fu.lf_global_quantizer_tail_pad_bits, &patched3)) {
      fprintf(stderr,
              "jxltran: LF global scale: splice in LF-global chunk0 failed\n");
      return false;
    }
    c0 = std::move(patched3);
  }
  chunks[p0] = std::move(c0);
  out->clear();
  for (const auto& ch : chunks) {
    out->insert(out->end(), ch.begin(), ch.end());
  }
  if (out->size() * 8u != fu.body_bit_length) {
    fprintf(stderr,
            "jxltran: LF DC quant / photon: output body %" PRIuS " bytes != "
            "expected %" PRIuS "\n",
            out->size(), static_cast<size_t>(fu.body_bit_length / 8u));
    return false;
  }
  return true;
}

static bool SpliceLfSplineEntropy(const std::vector<uint8_t>& chunk,
                                  size_t rel_start_bit, size_t rel_end_bit_excl,
                                  const std::vector<uint8_t>& new_mid,
                                  size_t new_mid_bits,
                                  size_t new_chunk_total_bits,
                                  std::vector<uint8_t>* out) {
  const size_t chunk_bits = chunk.size() * 8;
  if (rel_start_bit > rel_end_bit_excl || rel_end_bit_excl > chunk_bits) {
    return false;
  }
  if (new_mid_bits > new_mid.size() * 8u) {
    return false;
  }
  BitWriter bw;
  bw.AppendBitsRange(chunk.data(), 0, rel_start_bit);
  for (size_t i = 0; i < new_mid_bits; ++i) {
    bw.WriteBits(1, (new_mid[i >> 3] >> (i & 7)) & 1);
  }
  bw.AppendBitsRange(chunk.data(), rel_end_bit_excl,
                     chunk_bits - rel_end_bit_excl);
  const size_t written = bw.bit_pos();
  if (new_chunk_total_bits < written) {
    return false;
  }
  for (size_t z = written; z < new_chunk_total_bits; ++z) {
    bw.WriteBits(1, 0);
  }
  *out = bw.TakeBytes();
  return true;
}

static bool BuildSplineStreamOrderBody(const FramedUnit& fu, const uint8_t* orig,
                                       size_t orig_size, size_t body_src_abs_bit,
                                       std::vector<uint8_t>* out) {
  if (!fu.spline_edit) {
    return false;
  }
  if (fu.toc_strip_perm_reorder) {
    return false;
  }
  if (fu.toc_decoded_sizes.empty()) {
    return false;
  }
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
    return false;
  }
  const size_t n = fu.toc_decoded_sizes.size();
  if (!fu.toc_perm.empty() && fu.toc_perm.size() != n) {
    return false;
  }
  const int64_t delta_bits = fu.spline_edit_delta_bits;
  const int64_t dbytes = delta_bits / 8;
  if (delta_bits != dbytes * 8) {
    return false;
  }
  std::vector<uint32_t> old_stream = fu.toc_decoded_sizes;
  const int64_t lf_extra_adj = LfChunk0PhotonDcExtraAdjustBytes(fu);
  const int64_t stream_slot_adj = dbytes + lf_extra_adj;
  if (stream_slot_adj != 0) {
    const int64_t v =
        static_cast<int64_t>(old_stream[p0]) - stream_slot_adj;
    if (v < 0 || v > static_cast<int64_t>(UINT32_MAX)) {
      return false;
    }
    old_stream[p0] = static_cast<uint32_t>(v);
  }
  uint64_t sum_old = 0;
  for (uint32_t sz : old_stream) {
    sum_old += static_cast<uint64_t>(sz);
  }
  const int64_t new_bits = static_cast<int64_t>(fu.body_bit_length);
  const int64_t lf_extra_bits = lf_extra_adj * 8;
  if (new_bits < delta_bits + lf_extra_bits) {
    return false;
  }
  const uint64_t old_body_bytes =
      static_cast<uint64_t>((new_bits - delta_bits - lf_extra_bits) / 8);
  if (sum_old != old_body_bytes) {
    return false;
  }
  std::vector<std::vector<uint8_t>> chunks;
  if (!ExtractStreamOrderBodyChunks(orig, orig_size, body_src_abs_bit, old_stream,
                                    &chunks)) {
    return false;
  }
  if (chunks.size() != n) {
    return false;
  }
  const size_t rel_start = fu.lf_global_spline_rel_start_bit;
  const size_t rel_end = fu.lf_global_spline_rel_end_bit;
  const int64_t spline_only_c0_bytes =
      static_cast<int64_t>(fu.toc_decoded_sizes[p0]) - lf_extra_adj;
  if (spline_only_c0_bytes < 0 ||
      spline_only_c0_bytes > static_cast<int64_t>(UINT32_MAX)) {
    return false;
  }
  const size_t spline_only_c0_bits =
      static_cast<size_t>(static_cast<uint32_t>(spline_only_c0_bytes)) * 8u;
  std::vector<uint8_t> new_p0;
  if (!SpliceLfSplineEntropy(
          chunks[p0], rel_start, rel_end, fu.spline_edit_new_mid_bytes,
          fu.spline_edit_new_mid_bits, spline_only_c0_bits, &new_p0)) {
    return false;
  }
  chunks[p0] = std::move(new_p0);
  if (fu.photon_noise_edit) {
    const size_t noise_rel0 = PhotonNoiseRelInLfSection(fu, body_src_abs_bit);
    size_t noise_adj = 0;
    if (!NoiseRelAfterSplineSplice(noise_rel0, rel_start, rel_end,
                                   fu.spline_edit_new_mid_bits, &noise_adj)) {
      fprintf(stderr,
              "jxltran: spline+photon: noise LUT position overlaps spline entropy "
              "(TOC permutation path)\n");
      return false;
    }
    if (!SpliceLfPhotonNoise(chunks[p0], noise_adj, PhotonTocChunk0AdjustBytes(fu),
                             fu.photon_noise_new_bytes, &chunks[p0])) {
      fprintf(stderr,
              "jxltran: spline+photon: LF-global photon noise splice failed (TOC "
              "permutation path)\n");
      return false;
    }
  }
  out->clear();
  for (size_t s = 0; s < n; ++s) {
    const std::vector<uint8_t>& ch = chunks[s];
    out->insert(out->end(), ch.begin(), ch.end());
  }
  if (out->size() * 8u != fu.body_bit_length) {
    return false;
  }
  return true;
}

static bool ConcatLogicalOrderBody(const std::vector<std::vector<uint8_t>>& chunks,
                                   const std::vector<size_t>& inv, size_t n,
                                   std::vector<uint8_t>* out) {
  out->clear();
  for (size_t L = 0; L < n; ++L) {
    if (L >= inv.size()) return false;
    const size_t s = inv[L];
    if (s >= chunks.size()) return false;
    const std::vector<uint8_t>& c = chunks[s];
    out->insert(out->end(), c.begin(), c.end());
  }
  return true;
}

static bool BuildTocShuffleBody(const FramedUnit& fu, const uint8_t* orig,
                                size_t orig_size, size_t body_src_abs_bit,
                                std::vector<uint8_t>* out) {
  const auto& shuffle = fu.toc_body_stream_shuffle;
  const auto& src_sz = fu.toc_body_shuffle_src_sizes;
  const size_t n = fu.toc_decoded_sizes.size();
  if (shuffle.size() != n || src_sz.size() != n) {
    return false;
  }
  std::vector<std::vector<uint8_t>> chunks;
  if (!ExtractStreamOrderBodyChunks(orig, orig_size, body_src_abs_bit, src_sz,
                                    &chunks)) {
    fprintf(stderr, "jxltran: TOC shuffle: failed to read original body\n");
    return false;
  }
  if (chunks.size() != n) {
    return false;
  }
  std::vector<std::vector<uint8_t>> new_chunks;
  new_chunks.reserve(n);
  for (size_t new_s = 0; new_s < n; ++new_s) {
    const size_t old_s = shuffle[new_s];
    if (old_s >= chunks.size()) {
      return false;
    }
    new_chunks.emplace_back(chunks[old_s]);
  }
  if (fu.photon_noise_edit) {
    size_t p0 = 0;
    if (!PhysicalStreamIndexForLogical0(fu, &p0) || p0 >= new_chunks.size()) {
      fprintf(stderr,
              "jxltran: TOC shuffle: invalid TOC for photon noise splice\n");
      return false;
    }
    const size_t noise_rel = PhotonNoiseRelInLfSection(fu, body_src_abs_bit);
    const int64_t db = fu.photon_noise_delta_bytes;
    std::vector<uint8_t> patched;
    if (!SpliceLfPhotonNoise(new_chunks[p0], noise_rel, db,
                             fu.photon_noise_new_bytes, &patched)) {
      fprintf(stderr,
              "jxltran: TOC shuffle: photon noise LF-global splice failed\n");
      return false;
    }
    new_chunks[p0] = std::move(patched);
  }
  out->clear();
  for (const auto& ch : new_chunks) {
    out->insert(out->end(), ch.begin(), ch.end());
  }
  if (out->size() * 8u != fu.body_bit_length) {
    fprintf(stderr,
            "jxltran: TOC shuffle: output body %" PRIuS " bytes != expected %" PRIuS
            "\n",
            out->size(), static_cast<size_t>(fu.body_bit_length / 8u));
    return false;
  }
  return true;
}

// Rebuild frame body in logical TOC order and optionally splice photon-noise
// 80 bits inside the LF-global section (arbitrary bit phase).
static bool BuildStripPermPhotonBody(const FramedUnit& fu, const uint8_t* orig,
                                     size_t orig_size, size_t body_src_abs_bit,
                                     std::vector<uint8_t>* out) {
  if (!fu.toc_strip_perm_reorder) return false;
  const size_t n = fu.toc_decoded_sizes.size();
  if (n == 0) {
    fprintf(stderr, "jxltran: TOC strip: empty TOC\n");
    return false;
  }
  if (fu.toc_strip_stream_sizes.size() != n ||
      fu.toc_strip_logical_to_stream.size() != n) {
    fprintf(stderr,
            "jxltran: TOC strip: missing stream-order snapshot (internal "
            "error)\n");
    return false;
  }
  const std::vector<uint32_t>& stream_sz = fu.toc_strip_stream_sizes;
  const std::vector<size_t>& inv = fu.toc_strip_logical_to_stream;
  const int64_t db = fu.photon_noise_delta_bytes;

  uint64_t sum_stream = 0;
  for (uint32_t sz : stream_sz) sum_stream += static_cast<uint64_t>(sz);
  const int64_t delta_bits = static_cast<int64_t>(db) * 8;
  const int64_t new_bits = static_cast<int64_t>(fu.body_bit_length);
  if (new_bits < delta_bits) return false;
  const uint64_t old_body_bytes =
      static_cast<uint64_t>((new_bits - delta_bits) / 8);
  if (sum_stream != old_body_bytes) {
    fprintf(stderr,
            "jxltran: TOC strip: TOC byte sum %" PRIuS " != original body %" PRIuS
            "\n",
            static_cast<size_t>(sum_stream), static_cast<size_t>(old_body_bytes));
    return false;
  }
  std::vector<std::vector<uint8_t>> chunks;
  if (!ExtractStreamOrderBodyChunks(orig, orig_size, body_src_abs_bit, stream_sz,
                                    &chunks)) {
    fprintf(stderr, "jxltran: TOC strip: failed to read original body\n");
    return false;
  }
  if (chunks.size() != n) return false;
  std::vector<uint8_t> body;
  if (!ConcatLogicalOrderBody(chunks, inv, n, &body)) return false;
  if (fu.photon_noise_edit) {
    const size_t noise_rel = PhotonNoiseRelInLfSection(fu, body_src_abs_bit);
    if (inv[0] >= chunks.size()) return false;
    const size_t l0_bytes = chunks[inv[0]].size();
    if (body.size() < l0_bytes) return false;
    std::vector<uint8_t> l0(body.begin(), body.begin() + l0_bytes);
    if (!SpliceLfPhotonNoise(l0, noise_rel, db, fu.photon_noise_new_bytes, &l0)) {
      fprintf(stderr,
              "jxltran: TOC strip: photon noise LF-global bit splice failed\n");
      return false;
    }
    body.erase(body.begin(), body.begin() + l0_bytes);
    body.insert(body.begin(), l0.begin(), l0.end());
  }
  if (body.size() * 8u != fu.body_bit_length) {
    fprintf(stderr,
            "jxltran: TOC strip: output body %" PRIuS " bytes != expected %" PRIuS
            "\n",
            body.size(), static_cast<size_t>(fu.body_bit_length / 8u));
    return false;
  }
  *out = std::move(body);
  return true;
}

}  // namespace

// |fu.toc_perm| is logical section index at each stream slot (decoder-style A).
// WriteDecodedToc expects on-disk Lehmer P: stream slot of each logical section.
static bool WriteDecodedTocFromFuLogicalAtStreamPerm(
    BitWriter& bw, const size_t header_bits_mod8,
    const std::vector<uint32_t>& logical_at_stream,
    const std::vector<uint32_t>& sizes) {
  if (logical_at_stream.empty()) {
    return WriteDecodedToc(bw, header_bits_mod8, logical_at_stream, sizes);
  }
  std::vector<uint32_t> natural_to_stream;
  if (!TocPermLehmerNaturalToStreamFromLogicalAtStream(logical_at_stream,
                                                       sizes.size(),
                                                       &natural_to_stream)) {
    fprintf(stderr,
            "jxltran: internal error: could not invert TOC permutation for "
            "re-encode (logical-at-stream -> natural-to-stream)\n");
    return false;
  }
  return WriteDecodedToc(bw, header_bits_mod8, natural_to_stream, sizes);
}

bool ReadCodestream(const uint8_t* data, size_t size, ParsedCodestream* out) {
  // Reset the full struct so a reused |out| cannot retain stale image metadata
  // (e.g. have_animation) across reads — matches libjxl's decoder behaviour when
  // optional bundles are absent (see ImageMetadata::VisitFields else branch).
  *out = ParsedCodestream{};
  BitReader br(data, size);
  TraceSetReadBase(0);
  if (TraceIsOn()) {
    TraceMarkerAt(0, "read.codestream", "size_bytes=%" PRIuS "", size);
  }
  if (!ReadImageHeader(br, &out->image)) return false;
  if (!br.ok()) return false;
  if ((br.pos() & 7u) != 0) return false;
  size_t pos = br.pos() / 8;
  if (TraceIsOn()) {
    TraceMarkerAt(br.pos(), "read.codestream.after_image_header",
                  "next_frame_byte=%" PRIuS "", pos);
  }
  const uint32_t canvas_w = out->image.size.width;
  const uint32_t canvas_h = out->image.size.height;

  size_t frame_index = 0;
  while (pos < size) {
    FramedUnit fu;
    fu.original_frame_byte_offset = pos;
    const uint8_t* frame_base = data + pos;
    const size_t frame_avail = size - pos;
    TraceSetReadBase(pos * 8);
    if (TraceIsOn()) {
      TraceMarkerAt(pos * 8, "read.codestream.frame",
                    "index=%" PRIuS " byte_off=%" PRIuS "", frame_index, pos);
    }
    BitReader fbr(frame_base, frame_avail);
    if (!ReadFrameHeader(fbr, out->image.metadata, canvas_w, canvas_h,
                         &fu.frame, frame_base)) {
      fprintf(stderr,
              "jxltran: failed to parse frame header at codestream %" PRIuS
              "+%u\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.toc_start_bit = fbr.pos();
    uint64_t frame_data_bytes = 0;
    if (!ReadFrameTOC(fbr, fu.frame, out->image.metadata, canvas_w, canvas_h,
                      frame_base, frame_avail, &frame_data_bytes, &fu)) {
      const size_t abs_bit = pos * 8 + fbr.pos();
      fprintf(stderr,
              "jxltran: failed to parse TOC at codestream %" PRIuS
              "+%u (frame rel bits=%" PRIuS ")\n",
              abs_bit / 8u, static_cast<unsigned>(abs_bit & 7u), fbr.pos());
      return false;
    }
    const size_t body_start_bit = fbr.pos();
    for (uint64_t i = 0; i < frame_data_bytes; ++i) {
      (void)fbr.ReadBits(8);
    }
    if (!fbr.ok()) {
      fprintf(stderr,
              "jxltran: frame data overruns codestream at codestream %" PRIuS
              "+%u\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    if ((fbr.pos() & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u does not end on a "
              "byte boundary\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.body_bit_length = fbr.pos() - body_start_bit;
    if ((body_start_bit & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u: body does not start on "
              "a byte boundary after TOC\n",
              (pos * 8 + body_start_bit) / 8u,
              static_cast<unsigned>((pos * 8 + body_start_bit) & 7u));
      return false;
    }
    if ((fu.body_bit_length & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u: body length is not a "
              "whole number of bytes\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.full_frame_byte_len = fbr.pos() / 8;
    if (fu.full_frame_byte_len == 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u has zero length "
              "(corrupt codestream)\n",
              pos, 0u);
      return false;
    }
    if (pos + fu.full_frame_byte_len > size ||
        pos + fu.full_frame_byte_len < pos) {
      fprintf(stderr,
              "jxltran: frame size overruns codestream at byte %" PRIuS "\n",
              pos);
      return false;
    }

    {
      const uint32_t ftj = fu.frame.frame_type;
      if ((ftj == kFrameTypeRegular || ftj == kFrameTypeSkipProgressive) &&
          fu.body_bit_length != 0) {
        size_t sec0_off = 0;
        uint32_t s0 = 0;
        if (Section0BodySpan(fu, &sec0_off, &s0) && s0 != 0) {
          const size_t body_bytes = fu.body_bit_length / 8;
          if (sec0_off + static_cast<size_t>(s0) <= body_bytes) {
            const size_t body0_byte = body_start_bit / 8;
            if (body0_byte + sec0_off + static_cast<size_t>(s0) <= frame_avail) {
              const uint8_t* lf0 = frame_base + body0_byte + sec0_off;
              BitReader lf_br(lf0, static_cast<size_t>(s0));
              LfGlobalThroughNoise parsed{};
              std::array<std::pair<size_t, size_t>, kNumReferenceSlots> refs{};
              LfGlobalReferenceSlotSizes(*out, out->frames.size(),
                                        out->image.metadata, canvas_w,
                                        canvas_h, &refs);
              if (ReadLfGlobalThroughNoise(lf_br, fu.frame, out->image.metadata,
                                           canvas_w, canvas_h, refs, &parsed)) {
                const size_t lf0_abs =
                    pos * 8 + body_start_bit + sec0_off * 8;
                fu.lf_global_noise_lut_abs_valid = true;
                fu.lf_global_patch_ref_slot_mask = parsed.patch_ref_slot_mask;
                fu.lf_global_noise_lut_rel_bit = parsed.noise_lut_start_bit;
                fu.lf_global_noise_lut_abs_bit =
                    lf0_abs + parsed.noise_lut_start_bit;
                if (parsed.noise_lut_bytes_valid) {
                  fu.lf_global_noise_raw_valid = true;
                  fu.lf_global_noise_raw_bytes = parsed.noise_lut_bytes;
                }
                if (parsed.splines.has_value()) {
                  fu.lf_global_splines = std::move(*parsed.splines);
                }
                if ((fu.frame.flags & kFrameFlagSplines) != 0 &&
                    parsed.splines.has_value()) {
                  fu.lf_global_spline_region_valid = true;
                  fu.lf_global_spline_region_abs_start_bit =
                      lf0_abs + parsed.splines_entropy_start_bit;
                  fu.lf_global_spline_region_abs_end_bit =
                      lf0_abs + parsed.noise_lut_start_bit;
                  fu.lf_global_spline_rel_start_bit =
                      parsed.splines_entropy_start_bit;
                  fu.lf_global_spline_rel_end_bit = parsed.noise_lut_start_bit;
                } else                 if ((fu.frame.flags & kFrameFlagSplines) == 0) {
                  fu.lf_global_spline_region_valid = true;
                  fu.lf_global_spline_region_abs_start_bit =
                      lf0_abs + parsed.splines_entropy_start_bit;
                  fu.lf_global_spline_region_abs_end_bit =
                      lf0_abs + parsed.noise_lut_start_bit;
                  fu.lf_global_spline_rel_start_bit =
                      parsed.splines_entropy_start_bit;
                  fu.lf_global_spline_rel_end_bit = parsed.noise_lut_start_bit;
                }
                fu.lf_global_dc_quant_parsed = true;
                fu.lf_global_dc_quant_rel_region_start = parsed.dc_quant_start_bit;
                fu.lf_global_dc_quant_rel_region_old_len_bits =
                    parsed.dc_quant_end_bit - parsed.dc_quant_start_bit;
                fu.lf_global_dc_quant_all_default = parsed.dc_quant_all_default;
                fu.lf_global_dc_quant_f16_bits = parsed.dc_quant_f16_bits;
                if (parsed.quantizer_parsed) {
                  fu.lf_global_quantizer_parsed = true;
                  fu.lf_global_quantizer_rel_region_start =
                      parsed.quantizer_start_bit;
                  fu.lf_global_quantizer_rel_region_old_len_bits =
                      parsed.quantizer_end_bit - parsed.quantizer_start_bit;
                  fu.lf_global_quantizer_global_scale =
                      parsed.quantizer_global_scale;
                  fu.lf_global_quantizer_quant_dc = parsed.quantizer_quant_dc;
                }
              }
            }
          }
        }
      }
    }

    pos += fu.full_frame_byte_len;
    out->frames.push_back(std::move(fu));
    if (TraceIsOn()) {
      TraceMarkerAt(pos * 8, "read.codestream.frame_done",
                    "index=%" PRIuS " full_frame_bytes=%" PRIuS "", frame_index,
                    out->frames.back().full_frame_byte_len);
    }
    ++frame_index;
    if (out->frames.back().frame.is_last) break;
  }
  return br.ok();
}

bool WriteCodestream(const ParsedCodestream& cs, const uint8_t* original,
                     std::vector<uint8_t>* out) {
  if (original == nullptr) {
    fprintf(stderr,
            "jxltran: WriteCodestream requires original codestream bytes\n");
    return false;
  }
  BitWriter bw;
  const size_t orig_cs_size = CodestreamUsedBytes(cs);
  if (TraceIsOn()) {
    TraceMarkerAt(bw.bit_pos(), "write.codestream", "begin");
  }
  WriteImageHeader(bw, cs.image, original);
  const uint32_t meta_canvas_w = cs.image.size.width;
  const uint32_t meta_canvas_h = cs.image.size.height;
  size_t wframe = 0;
  for (const FramedUnit& fu : cs.frames) {
    const uint32_t ft = fu.frame.frame_type;
    if ((ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) &&
        !fu.frame.rf_reencode) {
      if (TraceIsOn()) {
        TraceMarkerAt(
            bw.bit_pos(), "write.codestream.frame_verbatim",
            "index=%" PRIuS " type=LF_or_RefOnly bytes=%" PRIuS "", wframe,
            fu.full_frame_byte_len);
      }
      bw.ZeroPadToByte();
      bw.AppendRawBytes(original + fu.original_frame_byte_offset,
                        fu.full_frame_byte_len);
      ++wframe;
      continue;
    }
    const uint8_t* frame_base = original + fu.original_frame_byte_offset;
    const size_t frame_bit_base = fu.original_frame_byte_offset * 8;
    const size_t toc_rel = fu.toc_start_bit;
    // Body always follows the TOC in the *source* codestream at this bit offset
    // (independent of whether we re-encoded the TOC to a different bit length).
    const size_t body_src_abs_bit =
        frame_bit_base + toc_rel + fu.toc_bit_length;
    if (TraceIsOn()) {
      TraceMarkerAt(bw.bit_pos(), "write.codestream.frame",
                    "index=%" PRIuS " src_byte=%" PRIuS " toc_bits=%" PRIuS
                    " body_bits=%" PRIuS "",
                    wframe, fu.original_frame_byte_offset, fu.toc_bit_length,
                    fu.body_bit_length);
    }
    // [frame_header] from |fu.frame|, then TOC, then body.
    // When the TOC permutation is unchanged, the size list (U32 section byte
    // lengths) encodes deterministically: matching |fu.toc_decoded_sizes| to the
    // body is sufficient; the only TOC entry we resize for these edits is the
    // physical slot carrying logical LF-global (section 0). Non-empty
    // permutations may still admit multiple valid Lehmer/ANS bitstreams, so we
    // copy the source TOC verbatim when nothing in the frame forces a rewrite.
    // The frame header may change length (e.g. crop). The body is whole bytes;
    // use AppendRawBytes when the writer is aligned.
    if (!WriteFrameHeader(bw, fu.frame, cs.image.metadata, frame_base,
                          meta_canvas_w, meta_canvas_h)) {
      fprintf(stderr, "jxltran: failed to write frame header\n");
      return false;
    }
    const size_t phase_out = bw.bit_pos() % 8u;
    const size_t phase_in = (frame_bit_base + toc_rel) % 8u;
    const bool strip_body = fu.toc_strip_perm_reorder;
    const bool shuffle_body = !fu.toc_body_stream_shuffle.empty();
    const bool spline_toc =
        fu.spline_edit && fu.spline_edit_toc_reencode;
    const bool spline_body_delta_no_perm =
        fu.spline_edit && fu.spline_edit_delta_bits != 0 &&
        fu.toc_perm.empty() && !fu.toc_decoded_sizes.empty();
    const bool photon_toc =
        fu.photon_noise_edit && fu.photon_noise_toc_reencode;
    const bool dc_quant_toc =
        fu.lf_global_dc_quant_edit && fu.lf_global_dc_quant_toc_reencode;
    const bool quantizer_toc =
        fu.lf_global_quantizer_edit && fu.lf_global_quantizer_toc_reencode;
    const bool need_toc_rewrite =
        strip_body || shuffle_body || photon_toc || dc_quant_toc ||
        quantizer_toc || spline_toc || spline_body_delta_no_perm ||
        (phase_out != phase_in);
    if (TraceIsOn()) {
      TraceWriteField(bw.bit_pos(), "write.frame.splice_toc_bits",
                      "src_start_bit=%" PRIuS " count=%" PRIuS " reencode=%d",
                      frame_bit_base + toc_rel, fu.toc_bit_length,
                      need_toc_rewrite ? 1 : 0);
    }
    const bool perm_nonempty = !fu.toc_perm.empty();

    if (!need_toc_rewrite) {
      bw.AppendBitsRange(original, frame_bit_base + toc_rel, fu.toc_bit_length);
    } else {
      const size_t toc_phase = bw.bit_pos() % 8u;
      if (strip_body) {
        static const std::vector<uint32_t> kEmptyTocPerm;
        if (!WriteDecodedToc(bw, toc_phase, kEmptyTocPerm,
                            fu.toc_decoded_sizes)) {
          fprintf(stderr,
                  "jxltran: failed to re-encode TOC (strip permutation)\n");
          return false;
        }
      } else if (shuffle_body) {
        if (!WriteDecodedTocFromFuLogicalAtStreamPerm(
                bw, toc_phase, fu.toc_perm, fu.toc_decoded_sizes)) {
          fprintf(stderr,
                  "jxltran: failed to re-encode TOC (center-first / shuffle)\n");
          return false;
        }
      } else if (!perm_nonempty) {
        static const std::vector<uint32_t> kEmptyTocPerm;
        if (!WriteDecodedToc(bw, toc_phase, kEmptyTocPerm,
                            fu.toc_decoded_sizes)) {
          fprintf(stderr,
                  "jxltran: failed to re-encode TOC (header/TOC bit phase "
                  "changed)\n");
          return false;
        }
      } else {
        if (!WriteDecodedTocFromFuLogicalAtStreamPerm(
                bw, toc_phase, fu.toc_perm, fu.toc_decoded_sizes)) {
          fprintf(stderr,
                  "jxltran: failed to re-encode TOC (with permutation)\n");
          return false;
        }
      }
    }
    if (TraceIsOn()) {
      TraceWriteField(bw.bit_pos(), "write.frame.splice_body",
                      "src_start_bit=%" PRIuS " bits=%" PRIuS " aligned=%d",
                      body_src_abs_bit, fu.body_bit_length,
                      (bw.bit_pos() & 7u) == 0 && (body_src_abs_bit & 7u) == 0
                          ? 1
                          : 0);
    }
    if (strip_body) {
      std::vector<uint8_t> new_body;
      if (!BuildStripPermPhotonBody(fu, original, orig_cs_size, body_src_abs_bit,
                                    &new_body)) {
        return false;
      }
      if ((bw.bit_pos() & 7u) != 0u) {
        fprintf(stderr,
                "jxltran: internal error: body not byte-aligned after TOC\n");
        return false;
      }
      bw.AppendRawBytes(new_body.data(), new_body.size());
    } else if (shuffle_body) {
      std::vector<uint8_t> new_body;
      if (!BuildTocShuffleBody(fu, original, orig_cs_size, body_src_abs_bit,
                               &new_body)) {
        return false;
      }
      if ((bw.bit_pos() & 7u) != 0u) {
        fprintf(stderr,
                "jxltran: internal error: body not byte-aligned after TOC\n");
        return false;
      }
      bw.AppendRawBytes(new_body.data(), new_body.size());
    } else if (fu.spline_edit && !fu.toc_perm.empty()) {
      std::vector<uint8_t> new_body;
      if (!BuildSplineStreamOrderBody(fu, original, orig_cs_size, body_src_abs_bit,
                                      &new_body)) {
        fprintf(stderr,
                "jxltran: failed to rebuild frame body for spline edit "
                "(TOC permutation)\n");
        return false;
      }
      if ((bw.bit_pos() & 7u) != 0u) {
        fprintf(stderr,
                "jxltran: internal error: body not byte-aligned after TOC\n");
        return false;
      }
      bw.AppendRawBytes(new_body.data(), new_body.size());
    } else if (fu.spline_edit) {
      const size_t s0 = fu.lf_global_spline_region_abs_start_bit;
      const size_t s1 = fu.lf_global_spline_region_abs_end_bit;
      const size_t a0 = body_src_abs_bit;
      const int64_t delta_bits = fu.spline_edit_delta_bits;
      const int64_t lf_extra_adj_b = LfChunk0PhotonDcExtraAdjustBytes(fu);
      const int64_t photon_adj_b = PhotonTocChunk0AdjustBytes(fu);
      const int64_t lf_extra_bits_adj = lf_extra_adj_b * 8;
      const size_t old_body_bits = static_cast<size_t>(
          static_cast<int64_t>(fu.body_bit_length) - delta_bits - lf_extra_bits_adj);
      const size_t body_end_orig = a0 + old_body_bits;
      if (s0 < a0 || s1 < s0 || s1 > body_end_orig) {
        fprintf(stderr, "jxltran: spline edit: bad patch span\n");
        return false;
      }
      size_t p0 = 0;
      if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
        fprintf(stderr, "jxltran: spline edit: invalid TOC for logical LF-global\n");
        return false;
      }
      if (p0 >= fu.toc_decoded_sizes.size()) {
        fprintf(stderr, "jxltran: spline edit: TOC index out of range\n");
        return false;
      }
      // |delta_bits| may be negative when the new spline bundle is shorter than
      // the old one; it is always a whole number of bytes (see operations.cc).
      if (delta_bits % 8 != 0) {
        fprintf(stderr,
                "jxltran: spline edit: delta_bits (%lld) is not a multiple of 8\n",
                static_cast<long long>(delta_bits));
        return false;
      }
      const int64_t dbytes = delta_bits / 8;
      const int64_t new_c0_bytes =
          static_cast<int64_t>(fu.toc_decoded_sizes[p0]);
      const int64_t old_c0_bytes_i = new_c0_bytes - dbytes - lf_extra_adj_b;
      if (old_c0_bytes_i < 0 ||
          old_c0_bytes_i > static_cast<int64_t>(UINT32_MAX)) {
        fprintf(stderr,
                "jxltran: spline edit: TOC size mismatch (internal, "
                "delta_bytes=%lld lf_chunk0_extra_adj_bytes=%lld)\n",
                static_cast<long long>(dbytes),
                static_cast<long long>(lf_extra_adj_b));
        return false;
      }
      const uint32_t old_c0_bytes =
          static_cast<uint32_t>(old_c0_bytes_i);
      const size_t chunk0_end_abs = a0 + static_cast<size_t>(old_c0_bytes) * 8u;
      if (s1 > chunk0_end_abs) {
        fprintf(stderr,
                "jxltran: spline edit: spline span extends past LF-global TOC "
                "section (not supported)\n");
        return false;
      }
      const std::vector<uint8_t>& mid = fu.spline_edit_new_mid_bytes;
      const size_t mid_bits = fu.spline_edit_new_mid_bits;
      const int64_t spline_only_c0_bytes =
          new_c0_bytes - lf_extra_adj_b;
      if (spline_only_c0_bytes < 0 ||
          spline_only_c0_bytes > static_cast<int64_t>(UINT32_MAX)) {
        fprintf(stderr,
                "jxltran: spline edit: TOC size mismatch after photon adjustment "
                "(internal)\n");
        return false;
      }
      const size_t spline_only_c0_bits =
          static_cast<size_t>(static_cast<uint32_t>(spline_only_c0_bytes)) * 8u;
      const size_t c0_payload_bits =
          (s0 - a0) + mid_bits + (chunk0_end_abs - s1);
      if (TraceIsOn()) {
        TraceWriteField(
            bw.bit_pos(), "write.frame.splice_spline_bits",
            "a0=%" PRIuS " s0=%" PRIuS " s1=%" PRIuS " chunk0_end=%" PRIuS
            " body_end_orig=%" PRIuS " prefix_bits=%" PRIuS " mid_bits=%" PRIuS
            " c0_tail_bits=%" PRIuS " new_c0_bits=%" PRIuS " c0_payload_bits=%"
            PRIuS "",
            a0, s0, s1, chunk0_end_abs, body_end_orig, s0 - a0, mid_bits,
            chunk0_end_abs - s1, spline_only_c0_bits, c0_payload_bits);
      }
      if (mid_bits > mid.size() * 8u) {
        fprintf(stderr, "jxltran: spline edit: invalid mid bit length\n");
        return false;
      }
      BitWriter c0_bw;
      c0_bw.AppendBitsRange(original, a0, s0 - a0);
      for (size_t bi = 0; bi < mid_bits; ++bi) {
        c0_bw.WriteBits(1, (mid[bi >> 3] >> (bi & 7)) & 1);
      }
      c0_bw.AppendBitsRange(original, s1, chunk0_end_abs - s1);
      if (spline_only_c0_bits < c0_payload_bits) {
        fprintf(stderr,
                "jxltran: spline edit: LF-global TOC section 0 is smaller than the "
                "spliced payload (TOC/body bookkeeping bug; chunk %" PRIuS ")\n",
                p0);
        return false;
      }
      for (size_t z = c0_payload_bits; z < spline_only_c0_bits; ++z) {
        c0_bw.WriteBits(1, 0);
      }
      std::vector<uint8_t> c0_bytes = c0_bw.TakeBytes();
      if (fu.photon_noise_edit) {
        const size_t noise_rel = fu.photon_noise_patch_abs_bit - a0;
        size_t noise_adj = 0;
        if (!NoiseRelAfterSplineSplice(noise_rel, s0 - a0, s1 - a0, mid_bits,
                                       &noise_adj)) {
          fprintf(stderr,
                  "jxltran: spline+photon: noise LUT position overlaps spline "
                  "entropy\n");
          return false;
        }
        std::vector<uint8_t> c0_out;
        if (!SpliceLfPhotonNoise(c0_bytes, noise_adj, photon_adj_b,
                                 fu.photon_noise_new_bytes, &c0_out)) {
          fprintf(stderr,
                  "jxltran: spline+photon: photon noise splice in LF-global chunk "
                  "failed\n");
          return false;
        }
        c0_bytes = std::move(c0_out);
      }
      const size_t new_c0_bits =
          static_cast<size_t>(fu.toc_decoded_sizes[p0]) * 8u;
      if (c0_bytes.size() * 8u != new_c0_bits) {
        fprintf(stderr,
                "jxltran: spline edit: LF-global TOC section 0 byte length mismatch "
                "(%" PRIuS " bytes vs %" PRIuS " bits)\n",
                c0_bytes.size(), new_c0_bits);
        return false;
      }
      bw.AppendRawBytes(c0_bytes.data(), c0_bytes.size());
      bw.AppendBitsRange(original, chunk0_end_abs, body_end_orig - chunk0_end_abs);
      if ((bw.bit_pos() & 7u) != 0u) {
        fprintf(stderr, "jxltran: spline edit: internal body alignment error\n");
        return false;
      }
    } else if (!fu.spline_edit &&
               (fu.photon_noise_edit || fu.lf_global_dc_quant_edit ||
                fu.lf_global_quantizer_edit)) {
      std::vector<uint8_t> new_body;
      if (!BuildLfChunk0PhotonDcBody(fu, original, orig_cs_size, body_src_abs_bit,
                                     &new_body)) {
        return false;
      }
      if ((bw.bit_pos() & 7u) != 0u) {
        fprintf(stderr,
                "jxltran: internal error: body not byte-aligned after TOC\n");
        return false;
      }
      bw.AppendRawBytes(new_body.data(), new_body.size());
    } else if (!fu.photon_noise_edit && !fu.lf_global_dc_quant_edit &&
               !fu.lf_global_quantizer_edit) {
      if ((bw.bit_pos() & 7u) == 0u && (body_src_abs_bit & 7u) == 0u) {
        bw.AppendRawBytes(original + body_src_abs_bit / 8,
                            fu.body_bit_length / 8);
      } else {
        bw.AppendBitsRange(original, body_src_abs_bit, fu.body_bit_length);
      }
    } else {
      const int64_t delta_bits =
          static_cast<int64_t>(fu.photon_noise_delta_bytes) * 8;
      const size_t old_body_bits =
          static_cast<size_t>(static_cast<int64_t>(fu.body_bit_length) -
                              delta_bits);
      const size_t body_end_orig = body_src_abs_bit + old_body_bits;
      const size_t p = fu.photon_noise_patch_abs_bit;
      if (p < body_src_abs_bit || p > body_end_orig) {
        fprintf(stderr, "jxltran: photon noise splice: bad patch offset\n");
        return false;
      }
      if (fu.photon_noise_delta_bytes < 0) {
        if (p + 80 > body_end_orig) {
          fprintf(stderr, "jxltran: photon noise splice: noise past body end\n");
          return false;
        }
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        bw.AppendBitsRange(original, p + 80, body_end_orig - (p + 80));
      } else if (fu.photon_noise_delta_bytes == 0) {
        if (p + 80 > body_end_orig) {
          fprintf(stderr, "jxltran: photon noise splice: noise past body end\n");
          return false;
        }
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        AppendNoise80Bits(&bw, fu.photon_noise_new_bytes);
        bw.AppendBitsRange(original, p + 80, body_end_orig - (p + 80));
      } else {
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        AppendNoise80Bits(&bw, fu.photon_noise_new_bytes);
        bw.AppendBitsRange(original, p, body_end_orig - p);
      }
    }
    ++wframe;
  }
  *out = bw.TakeBytes();
  return true;
}

bool ExtractFramedUnitBodyStreamChunks(const uint8_t* original, size_t orig_size,
                                       const FramedUnit& fu,
                                       std::vector<std::vector<uint8_t>>* chunks_out) {
  if (fu.toc_decoded_sizes.empty()) {
    chunks_out->clear();
    return true;
  }
  const size_t frame_bit_base = fu.original_frame_byte_offset * 8;
  const size_t body_src_abs_bit =
      frame_bit_base + fu.toc_start_bit + fu.toc_bit_length;
  return ExtractStreamOrderBodyChunks(original, orig_size, body_src_abs_bit,
                                      fu.toc_decoded_sizes, chunks_out);
}

}  // namespace jxltran
