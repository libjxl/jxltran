// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verbose human-readable description of TOC → frame section layout (stream
// order vs logical indices, byte offsets in the codestream).

#ifndef TOOLS_JXLTRAN_TOC_LAYOUT_H_
#define TOOLS_JXLTRAN_TOC_LAYOUT_H_

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace jxltran {

struct FrameHeader;
struct FrameTocMetrics;
struct JxlBox;
struct ParsedCodestream;

// Human-readable TOC logical section name (lf_global, lf_group …).
std::string TocLogicalSectionLabel(const FrameTocMetrics& m,
                                   const FrameHeader& fh, size_t logical);

// Prints one line per TOC slot in **stream order** (concatenated frame body):
// codestream byte offset, physical index, size, logical id, and a section
// label (lf_global / lf_group / hf_global / hf_pass…). When a TOC
// permutation is present, |logical| is decoded from the permutation (matches
// libjxl ReadToc + FrameDecoder::InitFrame).
void PrintCodestreamTocLayoutVerbose(FILE* out, const ParsedCodestream& cs);

// Single JSON object (machine-readable) for web UI / tooling: ISOBMFF box
// spans in the file, reassembled codestream layout (image header, frames,
// TOC, body sections with byte offsets in the codestream).
void PrintBitstreamStructureJson(FILE* out, const uint8_t* file_data,
                                 size_t file_size,
                                 const std::vector<JxlBox>& boxes,
                                 bool is_container_file,
                                 const ParsedCodestream& cs,
                                 size_t codestream_byte_length);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_LAYOUT_H_
