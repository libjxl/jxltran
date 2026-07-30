// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// jxltran: lossless JPEG XL bitstream remuxing tool.
// Similar to jpegtran but for JXL: rewrites the container without
// decode+encode, preserving the codestream unchanged.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "box.h"
#include "cmdline.h"
#include "codestream.h"
#include "frame_header.h"
#include "image_header.h"
#include "operations.h"
#include "reversible_undo.h"
#include "opsin_adjust.h"
#include "orientation_compose.h"
#include "printf_macros.h"
#include "restoration_filter.h"
#include "spline_io.h"
#include "toc_group_order.h"
#include "toc_layout.h"
#include "trace.h"

namespace {

// CLI value meaning "leave unchanged". Legacy alias "as-is" is still accepted.
static bool ParseKeepToken(const char* str) {
  return strcmp(str, "keep") == 0 || strcmp(str, "as-is") == 0;
}

// Print `reversible_undo:` once: multi-line (indented) when flags are present so
// bug reports stay readable without duplicate one-line / pretty variants.
static void PrintReversibleUndoLine(const std::string& undo_line) {
  if (undo_line.find(" --") == std::string::npos) {
    printf("reversible_undo: %s\n", undo_line.c_str());
    return;
  }
  std::string pretty = undo_line;
  for (size_t p = 0; (p = pretty.find(" --", p)) != std::string::npos;) {
    pretty.replace(p, 1, "\n  ");
    p += 3;
  }
  printf("reversible_undo:\n  %s\n", pretty.c_str());
}

static bool StoreExtractIccPath(const char* s, const char** out) {
  *out = s;
  return true;
}

// Reject entropy-decoded ICC codec bytes when expansion failed (varint preamble).
static bool IccBytesLookLikeIcc1(const std::vector<uint8_t>& icc) {
  if (icc.size() < 128) return false;
  if ((icc[0] & 0x80) != 0) return false;
  for (size_t i = 0; i + 4 <= icc.size() && i < 128; ++i) {
    if (icc[i] == 'a' && icc[i + 1] == 'c' && icc[i + 2] == 's' &&
        icc[i + 3] == 'p') {
      return true;
    }
  }
  return false;
}

bool ReadFile(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for reading\n", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size < 0) {
    fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  if (fread(out->data(), 1, out->size(), f) != out->size()) {
    fprintf(stderr, "Failed to read '%s'\n", path);
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

bool WriteFile(const char* path, const std::vector<uint8_t>& data) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for writing\n", path);
    return false;
  }
  if (fwrite(data.data(), 1, data.size(), f) != data.size()) {
    fprintf(stderr, "Failed to write '%s'\n", path);
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

enum class ContainerOpt {
  kAsIs,  // keep the same format as the input
  kNo,   // bare codestream: strip container and metadata (see --container help)
  kYes,  // always wrap in a container
  kIfNeeded  // use container only when metadata boxes are present
};

enum class JxlpOpt {
  kAsIs,  // keep jxlp boxes exactly as in the input
  kSort,  // reorder jxlp boxes by counter (in-order delivery, separate boxes)
  kMerge  // sort + merge all jxlp/jxlc into a single jxlc box
};

static bool ParseContainerOpt(const char* str, ContainerOpt* out) {
  if (ParseKeepToken(str)) {
    *out = ContainerOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "no") == 0) {
    *out = ContainerOpt::kNo;
    return true;
  }
  if (strcmp(str, "yes") == 0) {
    *out = ContainerOpt::kYes;
    return true;
  }
  if (strcmp(str, "if-needed") == 0) {
    *out = ContainerOpt::kIfNeeded;
    return true;
  }
  fprintf(
      stderr,
      "Unknown --container value '%s'. Expected: keep, no, yes, if-needed\n",
      str);
  return false;
}

static bool ParseJxlpOpt(const char* str, JxlpOpt* out) {
  if (ParseKeepToken(str)) {
    *out = JxlpOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "sort") == 0) {
    *out = JxlpOpt::kSort;
    return true;
  }
  if (strcmp(str, "merge") == 0) {
    *out = JxlpOpt::kMerge;
    return true;
  }
  fprintf(stderr, "Unknown --jxlp value '%s'. Expected: keep, sort, merge\n",
          str);
  return false;
}

// Parse a comma-separated list of strip types into a bitmask.
// Accepted tokens: exif, xmp, jumbf, jbrd, all.
static bool ParseStripOpt(const char* str, uint32_t* out) {
  *out = 0;
  const char* p = str;
  while (p && *p) {
    const char* end = strchr(p, ',');
    size_t len = end ? static_cast<size_t>(end - p) : strlen(p);
    if (strncmp(p, "exif", len) == 0 && len == 4) {
      *out |= jxltran::kStripExif;
    } else if (strncmp(p, "xmp", len) == 0 && len == 3) {
      *out |= jxltran::kStripXmp;
    } else if (strncmp(p, "jumbf", len) == 0 && len == 5) {
      *out |= jxltran::kStripJumbf;
    } else if (strncmp(p, "jbrd", len) == 0 && len == 4) {
      *out |= jxltran::kStripJbrd;
    } else if (strncmp(p, "all", len) == 0 && len == 3) {
      *out = jxltran::kStripAll;
      return true;
    } else {
      fprintf(stderr,
              "Unknown --strip type '%.*s'. "
              "Expected: exif, xmp, jumbf, jbrd, all\n",
              static_cast<int>(len), p);
      return false;
    }
    p = end ? end + 1 : nullptr;
  }
  return true;
}

// Maps a friendly box-type name (or a raw 4-char type) to a 4-byte box type.
static bool ParseBoxType(const char* p, size_t len, std::array<char, 4>* out) {
  static const struct {
    const char* friendly;
    const char* type4;
  } kMap[] = {
      {"exif", "Exif"}, {"xmp", "xml "},  {"jumbf", "jumb"}, {"jumb", "jumb"},
      {"jbrd", "jbrd"}, {"jxlc", "jxlc"}, {"jxlp", "jxlp"},  {"jxli", "jxli"},
      {"jxll", "jxll"}, {"brob", "brob"}, {"ftyp", "ftyp"},  {nullptr, nullptr},
  };
  for (int i = 0; kMap[i].friendly; ++i) {
    if (strlen(kMap[i].friendly) == len &&
        strncmp(p, kMap[i].friendly, len) == 0) {
      memcpy(out->data(), kMap[i].type4, 4);
      return true;
    }
  }
  if (len == 4) {  // raw 4-char type
    memcpy(out->data(), p, 4);
    return true;
  }
  fprintf(stderr, "Unknown box type '%.*s'\n", static_cast<int>(len), p);
  return false;
}

static bool ParseBoxTypeList(const char* str,
                             std::vector<std::array<char, 4>>* out) {
  const char* p = str;
  while (p && *p) {
    const char* end = strchr(p, ',');
    size_t len = end ? static_cast<size_t>(end - p) : strlen(p);
    std::array<char, 4> t;
    if (!ParseBoxType(p, len, &t)) return false;
    out->push_back(t);
    p = end ? end + 1 : nullptr;
  }
  return true;
}

// Holds a box-order mode plus, for kExplicit, the ordered type list.
struct BoxOrderArg {
  jxltran::BoxOrder order = jxltran::BoxOrder::kAsIs;
  std::vector<std::array<char, 4>> types;
};

static bool ParseBoxOrderOpt(const char* str, BoxOrderArg* out) {
  if (ParseKeepToken(str)) {
    out->order = jxltran::BoxOrder::kAsIs;
    return true;
  }
  if (strcmp(str, "before-codestream") == 0) {
    out->order = jxltran::BoxOrder::kBeforeCodestream;
    return true;
  }
  if (strcmp(str, "after-codestream") == 0) {
    out->order = jxltran::BoxOrder::kAfterCodestream;
    return true;
  }
  // Otherwise treat as a comma-separated explicit type list.
  out->order = jxltran::BoxOrder::kExplicit;
  return ParseBoxTypeList(str, &out->types);
}

// Holds a brob mode plus, for kCompress, the types to compress (empty = all).
struct BrobArg {
  jxltran::BrobOpt mode = jxltran::BrobOpt::kAsIs;
  std::vector<std::array<char, 4>> types;  // empty means all eligible types
};

static bool ParseBrobOpt(const char* str, BrobArg* out) {
  if (ParseKeepToken(str)) {
    out->mode = jxltran::BrobOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "decompress") == 0) {
    out->mode = jxltran::BrobOpt::kDecompress;
    return true;
  }
  if (strcmp(str, "compress") == 0) {
    out->mode = jxltran::BrobOpt::kCompress;
    return true;
  }
  if (strncmp(str, "compress:", 9) == 0) {
    out->mode = jxltran::BrobOpt::kCompress;
    return ParseBoxTypeList(str + 9, &out->types);
  }
  fprintf(stderr,
          "Unknown --brob value '%s'. "
          "Expected: keep, compress, compress:TYPE[,...], decompress\n",
          str);
  return false;
}
static bool ParseOrientationOpt(const char* s, uint32_t* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0') {
    fprintf(stderr,
            "orientation value: expected [1,8] or 90, 180, or 270 (degrees "
            "clockwise), got '%s'\n",
            s);
    return false;
  }
  if (v >= 1 && v <= 8) {
    *out = static_cast<uint32_t>(v);
    return true;
  }
  // Clockwise rotation aliases (EXIF tags 6, 3, 8).
  if (v == 90) {
    *out = 6;
    return true;
  }
  if (v == 180) {
    *out = 3;
    return true;
  }
  if (v == 270) {
    *out = 8;
    return true;
  }
  fprintf(stderr,
          "orientation value: expected [1,8] or 90, 180, or 270 (degrees "
          "clockwise), got '%s'\n",
          s);
  return false;
}

static bool ParseBitsPerSampleOpt(const char* s, uint32_t* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0' || v < 1 || v > 32) {
    fprintf(stderr,
            "--set-bits-per-sample: expected a value in [1,32], got '%s'\n", s);
    return false;
  }
  *out = static_cast<uint32_t>(v);
  return true;
}

struct OpsinFloatOpt {
  bool set = false;
  float value = 0.f;
};

static bool ParseOpsinSlider(const char* s, OpsinFloatOpt* out, const char* flag,
                             float min_v, float max_v) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr, "jxltran: %s: expected a number, got '%s'\n", flag, s);
    return false;
  }
  if (v < min_v || v > max_v) {
    fprintf(stderr, "jxltran: %s: value %g out of range [%g, %g]\n", flag,
            static_cast<double>(v), static_cast<double>(min_v),
            static_cast<double>(max_v));
    return false;
  }
  out->set = true;
  out->value = v;
  return true;
}

static bool ParseOpsinExposureOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-exposure", -10.f, 10.f);
}

static bool ParseOpsinTempOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-temperature", -100.f, 100.f);
}

static bool ParseOpsinTintOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-tint", -100.f, 100.f);
}

static bool ParseOpsinHueOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-hue", -100.f, 100.f);
}

static bool ParseOpsinBiasXOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-bias-x", -0.003f, 0.003f);
}

static bool ParseOpsinBiasYOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-bias-y", -0.003f, 0.003f);
}

static bool ParseOpsinBiasBOpt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-bias-b", -0.03f, 0.03f);
}

static bool ParseOpsinQuantBias0Opt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-quant-bias-0", -20.f, 20.f);
}
static bool ParseOpsinQuantBias1Opt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-quant-bias-1", -20.f, 20.f);
}
static bool ParseOpsinQuantBias2Opt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-quant-bias-2", -20.f, 20.f);
}
static bool ParseOpsinQuantBias3Opt(const char* s, OpsinFloatOpt* out) {
  return ParseOpsinSlider(s, out, "--opsin-quant-bias-3", -20.f, 20.f);
}

struct NumLoopsArg {
  bool set = false;
  uint32_t value = 0;
};

static bool ParseNumLoopsOpt(const char* s, NumLoopsArg* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0' || v < 0) {
    fprintf(stderr,
            "--set-num-loops: expected a non-negative integer, got '%s'\n", s);
    return false;
  }
  out->set = true;
  out->value = static_cast<uint32_t>(v);
  return true;
}

struct MainFloatArg {
  bool set = false;
  uint32_t value = 0;  // 0 or 1
};

static bool ParseMainFloatOpt(const char* s, MainFloatArg* out) {
  char* end;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || (v != 0 && v != 1)) {
    fprintf(stderr,
            "jxltran: --set-main-float: expected 0 or 1, got '%s'\n", s);
    return false;
  }
  out->set = true;
  out->value = static_cast<uint32_t>(v);
  return true;
}

struct MainExpArg {
  bool set = false;
  uint32_t value = 5;
};

static bool ParseMainExpOpt(const char* s, MainExpArg* out) {
  char* end;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || v < 2 || v > 8) {
    fprintf(stderr,
            "jxltran: --set-main-exp: expected integer 2..8, got '%s'\n", s);
    return false;
  }
  out->set = true;
  out->value = static_cast<uint32_t>(v);
  return true;
}

struct TpsArg {
  bool set = false;
  // If true, |percent_value| is resolved against the file's current TPS after
  // ReadCodestream; then cleared and |numerator|/|denominator| are filled.
  bool relative_percent = false;
  double percent_value = 100.0;
  uint32_t numerator = 1;
  uint32_t denominator = 1;
};

static bool ParseTpsOpt(const char* s, TpsArg* out) {
  size_t len = strlen(s);
  while (len > 0 && std::isspace(static_cast<unsigned char>(s[len - 1]))) {
    --len;
  }
  if (len == 0) {
    fprintf(stderr, "--set-tps: empty value\n");
    return false;
  }
  if (s[len - 1] == '%') {
    const char* const pct = s + len - 1;
    char* end = nullptr;
    const double value = strtod(s, &end);
    while (end < pct && std::isspace(static_cast<unsigned char>(*end))) {
      ++end;
    }
    if (end != pct || !(value > 0.0)) {
      fprintf(stderr,
              "--set-tps: expected a positive percentage (e.g. 50%%), got "
              "'%s'\n",
              s);
      return false;
    }
    out->set = true;
    out->relative_percent = true;
    out->percent_value = value;
    out->numerator = 1;
    out->denominator = 1;
    return true;
  }

  char* slash = const_cast<char*>(strchr(s, '/'));
  char* end;
  long num = strtol(s, &end, 10);
  if (num <= 0 || (end != slash && *end != '\0')) {
    fprintf(stderr,
            "--set-tps: expected a positive percentage (e.g. 50%%), a positive "
            "integer N, or N/D, got '%s'\n",
            s);
    return false;
  }
  long den = 1;
  if (slash) {
    den = strtol(slash + 1, &end, 10);
    if (den <= 0 || *end != '\0') {
      fprintf(stderr,
              "--set-tps: expected a positive percentage (e.g. 50%%), a "
              "positive integer N, or N/D, got "
              "'%s'\n",
              s);
      return false;
    }
  }
  out->set = true;
  out->relative_percent = false;
  out->numerator = static_cast<uint32_t>(num);
  out->denominator = static_cast<uint32_t>(den);
  return true;
}

struct IntensityTargetArg {
  bool set = false;
  bool to_default_bundle = false;
  float nits = 0.f;
};

static bool ParseIntensityTargetOpt(const char* s, IntensityTargetArg* out) {
  if (s == nullptr || *s == '\0') {
    fprintf(stderr, "jxltran: --set-intensity-target: empty value\n");
    return false;
  }
  if (strcasecmp(s, "default") == 0) {
    out->set = true;
    out->to_default_bundle = true;
    out->nits = 0.f;
    return true;
  }
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr,
            "jxltran: --set-intensity-target: expected a finite number > 0 "
            "(nits) or 'default', got '%s'\n",
            s);
    return false;
  }
  if (!std::isfinite(v) || !(v > 0.f)) {
    fprintf(stderr,
            "jxltran: --set-intensity-target: value must be finite and > 0; "
            "use 'default' for the implicit default tone-mapping bundle, got "
            "'%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->to_default_bundle = false;
  out->nits = v;
  return true;
}

static bool ResolveTpsPercentFromMetadata(const jxltran::ImageMetadata& meta,
                                          TpsArg* tps) {
  if (!tps->relative_percent) return true;
  if (!meta.have_animation) {
    fprintf(stderr,
            "jxltran: --set-tps: percentage values require an animated image "
            "(have_animation=true)\n");
    return false;
  }
  const uint32_t on = meta.animation.tps_numerator;
  const uint32_t od = meta.animation.tps_denominator;
  if (on == 0 || od == 0) {
    fprintf(stderr,
            "jxltran: --set-tps: ticks per second in file is zero/invalid\n");
    return false;
  }
  const double cur = static_cast<double>(on) / static_cast<double>(od);
  const double next = cur * (tps->percent_value / 100.0);
  if (!(next > 0.0) || !std::isfinite(next)) {
    fprintf(stderr, "jxltran: --set-tps: invalid scaled TPS\n");
    return false;
  }
  constexpr uint64_t kScale = 1000000;
  uint64_t n = static_cast<uint64_t>(std::llround(next * kScale));
  uint64_t d = kScale;
  if (n == 0) n = 1;
  uint64_t g = std::gcd(n, d);
  n /= g;
  d /= g;
  if (n > std::numeric_limits<uint32_t>::max() ||
      d > std::numeric_limits<uint32_t>::max()) {
    fprintf(
        stderr,
        "jxltran: --set-tps: scaled TPS exceeds uint32 range; try a smaller "
        "percentage or use N/D\n");
    return false;
  }
  tps->relative_percent = false;
  tps->numerator = static_cast<uint32_t>(n);
  tps->denominator = static_cast<uint32_t>(d);
  return true;
}

// WxH or WxH followed by two signed offsets (e.g. 3000x2000+500+500,
// 4000x3000-500-500). Output pixel (px,py) shows input (px+X, py+Y).
struct CropArg {
  bool set = false;
  int32_t x = 0, y = 0;
  uint32_t w = 0, h = 0;
};

static bool ParseCropOpt(const char* s, CropArg* out) {
  unsigned wu = 0, hu = 0;
  int consumed = 0;
  if (sscanf(s, "%ux%u%n", &wu, &hu, &consumed) != 2 || wu == 0 || hu == 0) {
    fprintf(stderr,
            "--crop: expected WxH or WxH with two signed offsets "
            "(e.g. 3000x2000+500+500), got '%s'\n",
            s);
    return false;
  }
  const char* tail = s + consumed;
  int32_t x = 0, y = 0;
  if (*tail != '\0') {
    char* endx = nullptr;
    long xl = strtol(tail, &endx, 10);
    if (endx == tail) {
      fprintf(stderr, "--crop: invalid X offset in '%s'\n", s);
      return false;
    }
    char* endy = nullptr;
    long yl = strtol(endx, &endy, 10);
    if (endy == endx) {
      fprintf(stderr,
              "--crop: expected two signed offsets after WxH (e.g. "
              "3000x2000+500+500), got '%s'\n",
              s);
      return false;
    }
    if (*endy != '\0') {
      fprintf(stderr, "--crop: trailing garbage after offsets in '%s'\n", s);
      return false;
    }
    if (xl < INT32_MIN || xl > INT32_MAX || yl < INT32_MIN || yl > INT32_MAX) {
      fprintf(stderr, "--crop: offsets out of range in '%s'\n", s);
      return false;
    }
    x = static_cast<int32_t>(xl);
    y = static_cast<int32_t>(yl);
  }
  out->set = true;
  out->w = wu;
  out->h = hu;
  out->x = x;
  out->y = y;
  return true;
}

static void PatchFtypMinorVersion(std::vector<jxltran::JxlBox>* boxes,
                                  uint32_t version) {
  for (auto& box : *boxes) {
    if (memcmp(box.type, "ftyp", 4) == 0 && box.data.size() >= 8) {
      box.data[4] = (version >> 24) & 0xFF;
      box.data[5] = (version >> 16) & 0xFF;
      box.data[6] = (version >> 8) & 0xFF;
      box.data[7] = version & 0xFF;
      break;
    }
  }
}

static bool ParseGabBlur(const char* s, jxltran::GabArgs* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f) {
    fprintf(stderr,
            "jxltran: --gab-blur expects a non-negative float, got '%s'\n", s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kBlur;
  out->amount = v;
  return true;
}

static bool ParseGabSharpen(const char* s, jxltran::GabArgs* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f) {
    fprintf(stderr,
            "jxltran: --gab-sharpen expects a non-negative float, got '%s'\n",
            s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kSharpen;
  out->amount = v;
  return true;
}

static bool ParseGabWeights(const char* s, jxltran::GabArgs* out) {
  float c[6];
  const int n = std::sscanf(s, "%f,%f,%f,%f,%f,%f", &c[0], &c[1], &c[2], &c[3],
                            &c[4], &c[5]);
  if (n != 6) {
    fprintf(stderr,
            "jxltran: --gab-weights expects six comma-separated floats "
            "(gab_x_weight1,2, gab_y_weight1,2, gab_b_weight1,2), got '%s'\n",
            s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kCustom;
  std::memcpy(out->custom, c, sizeof(c));
  return true;
}

static bool StoreGabStringArg(const char* s, const char** out) {
  *out = s;
  return true;
}

struct PhotonIsoArg {
  bool set = false;
  float iso = 0.f;
};

struct EpfItersArg {
  bool set = false;
  uint32_t iters = 0;
};

static bool ParseEpfItersOpt(const char* s, EpfItersArg* out) {
  char* end = nullptr;
  const unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || v > 3ul) {
    fprintf(stderr,
            "jxltran: --set-epf-iters expects 0, 1, 2, or 3, got '%s'\n", s);
    return false;
  }
  out->set = true;
  out->iters = static_cast<uint32_t>(v);
  return true;
}

struct EpfAmpScaleArg {
  bool set = false;
  float factor = 1.f;
};

static bool ParseEpfAmpScaleOpt(const char* s, EpfAmpScaleArg* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || !(v > 0.f) || !std::isfinite(v)) {
    fprintf(stderr,
            "jxltran: --set-epf-amplitude-scale expects a positive finite float, "
            "got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->factor = v;
  return true;
}

struct EpfUniformArg {
  bool set = false;
  float value = 0.f;
};

static bool ParseEpfUniformOpt(const char* s, EpfUniformArg* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || !std::isfinite(v) || v < 0.f || v > 1.f) {
    fprintf(stderr,
            "jxltran: --set-epf-uniformity expects a float in [0,1], got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->value = v;
  return true;
}

static bool ParsePhotonIsoOpt(const char* s, PhotonIsoArg* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f || !std::isfinite(v)) {
    fprintf(stderr,
            "jxltran: --set-photon-noise-iso expects a non-negative finite "
            "float (0 = remove synthetic noise, >0 = nominal ISO), got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->iso = v;
  return true;
}

struct LfDcQuantMulArg {
  bool set = false;
  float mx = 1.f;
  float my = 1.f;
  float mb = 1.f;
};

static bool ParseLfDcQuantMulOpt(const char* s, LfDcQuantMulArg* out) {
  char* e1 = nullptr;
  const double xd = std::strtod(s, &e1);
  if (e1 == s || *e1 != ',') {
    fprintf(stderr,
            "jxltran: --lf-dc-quant-mul expects three comma-separated floats "
            "(X,Y,B multipliers; 1 = unchanged), got '%s'\n",
            s);
    return false;
  }
  char* e2 = nullptr;
  const double yd = std::strtod(e1 + 1, &e2);
  if (e2 == e1 + 1 || *e2 != ',') {
    fprintf(stderr,
            "jxltran: --lf-dc-quant-mul expects three comma-separated floats "
            "(X,Y,B multipliers; 1 = unchanged), got '%s'\n",
            s);
    return false;
  }
  char* e3 = nullptr;
  const double bd = std::strtod(e2 + 1, &e3);
  if (e3 == e2 + 1 || *e3 != '\0') {
    fprintf(stderr,
            "jxltran: --lf-dc-quant-mul expects three comma-separated floats "
            "(X,Y,B multipliers; 1 = unchanged), got '%s'\n",
            s);
    return false;
  }
  const float x = static_cast<float>(xd);
  const float y = static_cast<float>(yd);
  const float b = static_cast<float>(bd);
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(b)) {
    fprintf(stderr,
            "jxltran: --lf-dc-quant-mul: each value must be finite, got '%s'\n", s);
    return false;
  }
  if (x <= 0.f || y <= 0.f || b <= 0.f) {
    fprintf(stderr,
            "jxltran: --lf-dc-quant-mul: each value must be positive (decoder "
            "rejects non-positive LF dequant), got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->mx = x;
  out->my = y;
  out->mb = b;
  return true;
}

struct LfGlobalScaleArg {
  bool set = false;
  uint32_t value = 0;
};

static bool ParseLfGlobalScaleOpt(const char* s, LfGlobalScaleArg* out) {
  char* end = nullptr;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || v < 1ul || v > 32768ul) {
    fprintf(stderr,
            "jxltran: --lf-global-scale expects an integer 1..32768, got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->value = static_cast<uint32_t>(v);
  return true;
}

struct FramesArg {
  bool set = false;
  /** Sorted unique indices; used with --frames for per-frame filters. */
  std::vector<size_t> indices;
  /** First-mention order without duplicates; legacy use with --keep-listed-frames
   * when --keep-frames is omitted. */
  std::vector<size_t> indices_ordered;
};

static bool ParseFramesArg(const char* s, FramesArg* out) {
  out->set = true;
  out->indices.clear();
  out->indices_ordered.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --frames expects a comma-separated list of non-negative "
            "integers (codestream frame indices, 0-based)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr, "jxltran: --frames: invalid integer near '%s'\n", p);
      return false;
    }
    if (v > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr, "jxltran: --frames: frame index out of range near '%s'\n",
              p);
      return false;
    }
    const size_t idx = static_cast<size_t>(v);
    bool seen = false;
    for (size_t x : out->indices_ordered) {
      if (x == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) out->indices_ordered.push_back(idx);
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == '\0') break;
    fprintf(stderr,
            "jxltran: --frames: expected end of list or ',' before '%s'\n", p);
    return false;
  }
  if (out->indices_ordered.empty()) {
    fprintf(stderr, "jxltran: --frames needs at least one frame index\n");
    return false;
  }
  out->indices = out->indices_ordered;
  std::sort(out->indices.begin(), out->indices.end());
  return true;
}

struct LfChunk0TailTrimArg {
  bool set = false;
  /** (codestream frame index, trim 1..255 whole tail bytes of physical chunk0). */
  std::vector<std::pair<size_t, uint8_t>> pairs;
};

static bool ParseLfChunk0TailTrimArg(const char* s, LfChunk0TailTrimArg* out) {
  out->set = true;
  out->pairs.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes expects "
            "FRAME:BYTES[,FRAME:BYTES] (BYTES is 1..255)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long fi = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: invalid frame index "
              "near '%s'\n",
              p);
      return false;
    }
    if (fi > static_cast<unsigned long>((std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: frame index out of "
              "range near '%s'\n",
              p);
      return false;
    }
    if (*end != ':') {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: expected ':' after "
              "frame index near '%s'\n",
              end);
      return false;
    }
    p = end + 1;
    unsigned long tb = strtoul(p, &end, 10);
    if (end == p || (*end != '\0' && *end != ',' && *end != ' ' && *end != '\t')) {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: invalid byte count "
              "near '%s'\n",
              p);
      return false;
    }
    if (tb < 1 || tb > 255) {
      fprintf(stderr,
              "jxltran: --lf-global-chunk0-tail-trim-bytes: byte count must be 1..255 "
              "(got %lu)\n",
              static_cast<unsigned long>(tb));
      return false;
    }
    const size_t frame_i = static_cast<size_t>(fi);
    for (const auto& pr : out->pairs) {
      if (pr.first == frame_i) {
        fprintf(stderr,
                "jxltran: --lf-global-chunk0-tail-trim-bytes: duplicate frame "
                "%" PRIuS "\n",
                frame_i);
        return false;
      }
    }
    out->pairs.emplace_back(frame_i, static_cast<uint8_t>(tb));
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == '\0') break;
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes: expected ',' or end "
            "near '%s'\n",
            p);
    return false;
  }
  if (out->pairs.empty()) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes needs at least one "
            "FRAME:BYTES entry\n");
    return false;
  }
  return true;
}

struct KeepFramesArg {
  bool set = false;
  std::vector<size_t> indices;
  std::vector<size_t> indices_ordered;
};

static bool ParseKeepFramesArg(const char* s, KeepFramesArg* out) {
  out->set = true;
  out->indices.clear();
  out->indices_ordered.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --keep-frames expects a comma-separated list of non-negative "
            "integers (codestream frame indices, 0-based)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr, "jxltran: --keep-frames: invalid integer near '%s'\n", p);
      return false;
    }
    if (v > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --keep-frames: frame index out of range near '%s'\n", p);
      return false;
    }
    const size_t idx = static_cast<size_t>(v);
    bool seen = false;
    for (size_t x : out->indices_ordered) {
      if (x == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) out->indices_ordered.push_back(idx);
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == '\0') break;
    fprintf(stderr,
            "jxltran: --keep-frames: expected end of list or ',' before '%s'\n", p);
    return false;
  }
  if (out->indices_ordered.empty()) {
    fprintf(stderr, "jxltran: --keep-frames needs at least one frame index\n");
    return false;
  }
  out->indices = out->indices_ordered;
  std::sort(out->indices.begin(), out->indices.end());
  return true;
}

struct SetFrameBlendsArg {
  bool set = false;
  std::vector<jxltran::FrameBlendOverride> overrides;
};

struct SetFrameDurationsArg {
  bool set = false;
  std::vector<std::pair<size_t, uint32_t>> pairs;
};

struct SetFrameNamesArg {
  bool set = false;
  std::vector<std::pair<size_t, std::vector<uint8_t>>> pairs;
};

struct SetFrameRegionsArg {
  bool set = false;
  std::vector<std::pair<size_t, CropArg>> pairs;
};

static int HexNibbleForFrameName(const char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

static bool ParseSetFrameNamesArg(const char* s, SetFrameNamesArg* out) {
  out->set = true;
  out->pairs.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --set-frame-names expects INDEX:VALUE pairs "
            "(comma-separated); VALUE is an even-length hex string (UTF-8 bytes, "
            "same as --info name_hex=), a literal UTF-8 run when it is not "
            "even-length all-hex, or a double-quoted literal (spaces, commas, "
            "hex-like ASCII, …; use \\\" and \\\\ inside quotes); use INDEX: "
            "alone to clear the name\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long fi = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --set-frame-names: expected frame index near '%s'\n",
              p);
      return false;
    }
    if (fi > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --set-frame-names: frame index out of range near '%s'\n",
              p);
      return false;
    }
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != ':') {
      fprintf(stderr,
              "jxltran: --set-frame-names: expected ':' after frame index near "
              "'%s'\n",
              p);
      return false;
    }
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    std::vector<uint8_t> name;
    if (*p == '"') {
      ++p;
      for (;;) {
        if (*p == '\0') {
          fprintf(stderr,
                  "jxltran: --set-frame-names: unterminated quoted value in "
                  "'%s'\n",
                  s);
          return false;
        }
        if (*p == '"') {
          ++p;
          break;
        }
        if (*p == '\\' && p[1] != '\0') {
          ++p;
          name.push_back(static_cast<uint8_t>(*p++));
          continue;
        }
        name.push_back(static_cast<uint8_t>(*p++));
      }
    } else {
      const char* tok_start = p;
      while (*p != '\0' && *p != ',') ++p;
      const char* tok_end = p;
      while (tok_end > tok_start &&
             (tok_end[-1] == ' ' || tok_end[-1] == '\t')) {
        --tok_end;
      }
      const size_t len = static_cast<size_t>(tok_end - tok_start);
      if (len == 0) {
        // clear
      } else {
        bool all_hex = true;
        for (size_t i = 0; i < len; ++i) {
          if (!std::isxdigit(static_cast<unsigned char>(tok_start[i]))) {
            all_hex = false;
            break;
          }
        }
        if (all_hex && (len % 2) == 0) {
          name.reserve(len / 2);
          for (size_t i = 0; i < len; i += 2) {
            const int hi = HexNibbleForFrameName(tok_start[i]);
            const int lo = HexNibbleForFrameName(tok_start[i + 1]);
            if (hi < 0 || lo < 0) {
              fprintf(stderr,
                      "jxltran: --set-frame-names: invalid hex near '%.*s'\n",
                      static_cast<int>(len), tok_start);
              return false;
            }
            name.push_back(static_cast<uint8_t>((hi << 4) | lo));
          }
        } else {
          name.assign(reinterpret_cast<const uint8_t*>(tok_start),
                      reinterpret_cast<const uint8_t*>(tok_start) + len);
        }
      }
    }
    out->pairs.emplace_back(static_cast<size_t>(fi), std::move(name));
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == '\0') break;
    fprintf(stderr,
            "jxltran: --set-frame-names: expected ',' or end of string near "
            "'%s'\n",
            p);
    return false;
  }
  if (out->pairs.empty()) {
    fprintf(stderr, "jxltran: --set-frame-names needs at least one pair\n");
    return false;
  }
  return true;
}

static bool ParseBlendModeToken(const char* token, uint32_t* mode_out) {
  char* end = nullptr;
  unsigned long v = strtoul(token, &end, 10);
  if (end != token && *end == '\0') {
    if (v <= 4u) {
      *mode_out = static_cast<uint32_t>(v);
      return true;
    }
    fprintf(stderr,
            "jxltran: blend mode must be 0–4 or a slug (replace, add, …), got "
            "'%s'\n",
            token);
    return false;
  }
  if (strcmp(token, "replace") == 0) {
    *mode_out = 0;
    return true;
  }
  if (strcmp(token, "add") == 0) {
    *mode_out = 1;
    return true;
  }
  if (strcmp(token, "blend") == 0) {
    *mode_out = 2;
    return true;
  }
  if (strcmp(token, "alpha_weighted_add") == 0) {
    *mode_out = 3;
    return true;
  }
  if (strcmp(token, "mul") == 0) {
    *mode_out = 4;
    return true;
  }
  fprintf(stderr,
          "jxltran: unknown blend mode '%s' (use 0–4 or replace|add|blend|"
          "alpha_weighted_add|mul)\n",
          token);
  return false;
}

static bool LooksLikeFrameIndexColon(const char* p) {
  while (*p == ' ' || *p == '\t') ++p;
  if (*p == '\0') return false;
  if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
  char* end = nullptr;
  (void)strtoul(p, &end, 10);
  return end != nullptr && end > p && *end == ':';
}

static bool ParseBlendBoolVal(const std::string& val, bool* out) {
  if (val == "0" || val == "false" || val == "no") {
    *out = false;
    return true;
  }
  if (val == "1" || val == "true" || val == "yes") {
    *out = true;
    return true;
  }
  return false;
}

static bool ApplyBlendOverrideKeyVal(jxltran::FrameBlendOverride* o,
                                     const std::string& key,
                                     const std::string& val) {
  if (key == "alpha") {
    char* e = nullptr;
    unsigned long v = strtoul(val.c_str(), &e, 10);
    if (e != val.c_str() + val.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: alpha= expects a non-negative "
              "integer, got '%s'\n",
              val.c_str());
      return false;
    }
    if (v > static_cast<unsigned long>(UINT32_MAX)) return false;
    o->set_alpha_channel = true;
    o->alpha_channel = static_cast<uint32_t>(v);
    return true;
  }
  if (key == "clamp") {
    bool b = false;
    if (!ParseBlendBoolVal(val, &b)) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: clamp= expects 0|1|true|false|"
              "yes|no, got '%s'\n",
              val.c_str());
      return false;
    }
    o->set_clamp = true;
    o->clamp = b;
    return true;
  }
  if (key == "source") {
    char* e = nullptr;
    unsigned long v = strtoul(val.c_str(), &e, 10);
    if (e != val.c_str() + val.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: source= expects an integer, got '%s'\n",
              val.c_str());
      return false;
    }
    if (v > static_cast<unsigned long>(UINT32_MAX)) return false;
    o->set_source = true;
    o->source = static_cast<uint32_t>(v);
    return true;
  }
  if (key == "target" || key == "save_as_reference") {
    char* e = nullptr;
    unsigned long v = strtoul(val.c_str(), &e, 10);
    if (e != val.c_str() + val.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: target= expects an integer 0–3, "
              "got '%s'\n",
              val.c_str());
      return false;
    }
    if (v > 3u) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: target= must be in range 0–3\n");
      return false;
    }
    o->set_save_as_reference = true;
    o->save_as_reference = static_cast<uint32_t>(v);
    return true;
  }
  fprintf(stderr,
          "jxltran: --set-frame-blends: unknown field '%s' (use alpha=, "
          "clamp=, source=, target=)\n",
          key.c_str());
  return false;
}

static bool ParseSetFrameBlendsArg(const char* s, SetFrameBlendsArg* out) {
  out->set = true;
  out->overrides.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --set-frame-blends expects INDEX:MODE[,FIELD=VAL...] "
            "entries\n"
            "    (comma-separated frames; optional alpha=, clamp=, source=, "
            "target= per frame)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;

    jxltran::FrameBlendOverride o{};
    char* end = nullptr;
    unsigned long fi = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: expected frame index near '%s'\n", p);
      return false;
    }
    if (fi > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame index out of range near '%s'\n",
              p);
      return false;
    }
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != ':') {
      fprintf(stderr,
              "jxltran: --set-frame-blends: expected ':' after frame index near "
              "'%s'\n",
              p);
      return false;
    }
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    const char* mode_start = p;
    while (*p != '\0') {
      if (*p == ',') {
        if (LooksLikeFrameIndexColon(p + 1)) break;
        break;
      }
      if (*p == ' ' || *p == '\t') break;
      ++p;
    }
    if (p == mode_start) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: missing blend mode after ':'\n");
      return false;
    }
    std::string mode_token(mode_start, p);
    if (!ParseBlendModeToken(mode_token.c_str(), &o.mode)) return false;
    o.frame_index = static_cast<size_t>(fi);

    bool done_frame = false;
    while (!done_frame) {
      while (*p == ' ' || *p == '\t') ++p;
      if (*p == '\0') {
        out->overrides.push_back(o);
        done_frame = true;
        break;
      }
      if (*p == ',') {
        if (LooksLikeFrameIndexColon(p + 1)) {
          ++p;
          out->overrides.push_back(o);
          done_frame = true;
          break;
        }
        ++p;
        continue;
      }
      if (LooksLikeFrameIndexColon(p)) {
        out->overrides.push_back(o);
        done_frame = true;
        break;
      }

      const char* key_start = p;
      while (*p != '\0' && *p != '=' && *p != ',' && *p != ' ' && *p != '\t') {
        ++p;
      }
      if (*p != '=') {
        fprintf(stderr,
                "jxltran: --set-frame-blends: expected key=value near '%s'\n",
                key_start);
        return false;
      }
      std::string key(key_start, p);
      ++p;
      const char* val_start = p;
      while (*p != '\0' && *p != ',' && *p != ' ' && *p != '\t') ++p;
      if (p == val_start) {
        fprintf(stderr,
                "jxltran: --set-frame-blends: empty value for key '%s'\n",
                key.c_str());
        return false;
      }
      std::string val(val_start, p);
      if (!ApplyBlendOverrideKeyVal(&o, key, val)) return false;
    }
  }
  if (out->overrides.empty()) {
    fprintf(stderr, "jxltran: --set-frame-blends needs at least one entry\n");
    return false;
  }
  return true;
}

static bool ParseSetFrameDurationsArg(const char* s, SetFrameDurationsArg* out) {
  out->set = true;
  out->pairs.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --set-frame-durations expects INDEX:TICKS pairs "
            "(comma-separated)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long fi = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: expected frame index near '%s'\n",
              p);
      return false;
    }
    if (fi > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: frame index out of range near "
              "'%s'\n",
              p);
      return false;
    }
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != ':') {
      fprintf(stderr,
              "jxltran: --set-frame-durations: expected ':' after frame index "
              "near '%s'\n",
              p);
      return false;
    }
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    unsigned long ticks = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: missing tick count after ':'\n");
      return false;
    }
    if (ticks > static_cast<unsigned long>(UINT32_MAX)) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: duration out of uint32 range\n");
      return false;
    }
    if (*end != '\0' && *end != ',' && *end != ' ' && *end != '\t') {
      fprintf(stderr,
              "jxltran: --set-frame-durations: invalid duration near '%s'\n",
              end);
      return false;
    }
    out->pairs.emplace_back(static_cast<size_t>(fi), static_cast<uint32_t>(ticks));
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == '\0') break;
    fprintf(stderr,
            "jxltran: --set-frame-durations: expected end or ',' before '%s'\n",
            p);
    return false;
  }
  if (out->pairs.empty()) {
    fprintf(stderr, "jxltran: --set-frame-durations needs at least one pair\n");
    return false;
  }
  return true;
}

static bool ParseSetFrameRegionsArg(const char* s, SetFrameRegionsArg* out) {
  out->set = true;
  out->pairs.clear();
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --set-frame-region expects INDEX:WxH+X+Y (comma-separated "
            "entries allowed)\n");
    return false;
  }
  const char* p = s;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    char* end = nullptr;
    unsigned long fi = strtoul(p, &end, 10);
    if (end == p) {
      fprintf(stderr,
              "jxltran: --set-frame-region: expected frame index near '%s'\n",
              p);
      return false;
    }
    if (fi > static_cast<unsigned long>(
            (std::numeric_limits<size_t>::max)())) {
      fprintf(stderr,
              "jxltran: --set-frame-region: frame index out of range near '%s'\n",
              p);
      return false;
    }
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != ':') {
      fprintf(stderr,
              "jxltran: --set-frame-region: expected ':' after frame index near "
              "'%s'\n",
              p);
      return false;
    }
    ++p;
    const char* crop_start = p;
    while (*p != '\0') {
      if (*p == ',' && LooksLikeFrameIndexColon(p + 1)) {
        break;
      }
      ++p;
    }
    while (p > crop_start && (p[-1] == ' ' || p[-1] == '\t')) {
      --p;
    }
    std::string crop_str(crop_start, p);
    if (crop_str.empty()) {
      fprintf(stderr,
              "jxltran: --set-frame-region: missing WxH+X+Y after frame index\n");
      return false;
    }
    unsigned wu = 0, hu = 0;
    int consumed = 0;
    if (std::sscanf(crop_str.c_str(), "%ux%u%n", &wu, &hu, &consumed) != 2 ||
        wu == 0 || hu == 0) {
      fprintf(stderr,
              "jxltran: --set-frame-region: expected WxH or WxH with two signed "
              "offsets (e.g. 3000x2000+5-3), got '%s'\n",
              crop_str.c_str());
      return false;
    }
    const char* tail = crop_str.c_str() + consumed;
    int32_t x = 0;
    int32_t y = 0;
    if (*tail != '\0') {
      char* endx = nullptr;
      long xl = std::strtol(tail, &endx, 10);
      if (endx == tail) {
        fprintf(stderr,
                "jxltran: --set-frame-region: invalid X offset in '%s'\n",
                crop_str.c_str());
        return false;
      }
      char* endy = nullptr;
      long yl = std::strtol(endx, &endy, 10);
      if (endy == endx) {
        fprintf(stderr,
                "jxltran: --set-frame-region: expected two signed offsets after "
                "WxH in '%s'\n",
                crop_str.c_str());
        return false;
      }
      if (*endy != '\0') {
        fprintf(stderr,
                "jxltran: --set-frame-region: trailing garbage after offsets in "
                "'%s'\n",
                crop_str.c_str());
        return false;
      }
      if (xl < INT32_MIN || xl > INT32_MAX || yl < INT32_MIN ||
          yl > INT32_MAX) {
        fprintf(stderr,
                "jxltran: --set-frame-region: offsets out of range in '%s'\n",
                crop_str.c_str());
        return false;
      }
      x = static_cast<int32_t>(xl);
      y = static_cast<int32_t>(yl);
    }
    CropArg cr;
    cr.set = true;
    cr.w = wu;
    cr.h = hu;
    cr.x = x;
    cr.y = y;
    out->pairs.emplace_back(static_cast<size_t>(fi), cr);
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') break;
    if (*p == ',') {
      ++p;
      continue;
    }
    fprintf(stderr,
            "jxltran: --set-frame-region: expected end or ',' before '%s'\n",
            p);
    return false;
  }
  if (out->pairs.empty()) {
    fprintf(stderr, "jxltran: --set-frame-region needs at least one entry\n");
    return false;
  }
  return true;
}

static bool ParseGroupOrder(const char* s, jxltran::TocGroupOrderCli* out) {
  if (strcmp(s, "keep") == 0) {
    *out = jxltran::TocGroupOrderCli::kKeep;
    return true;
  }
  if (strcmp(s, "0") == 0) {
    *out = jxltran::TocGroupOrderCli::kCanonical;
    return true;
  }
  if (strcmp(s, "1") == 0) {
    *out = jxltran::TocGroupOrderCli::kCenterFirst;
    return true;
  }
  if (strcmp(s, "progressive") == 0) {
    *out = jxltran::TocGroupOrderCli::kProgressive;
    return true;
  }
  fprintf(stderr,
          "jxltran: --group_order expects keep|0|1|progressive, got '%s'\n", s);
  return false;
}

static bool ParseCenterI64(const char* s, int64_t* out) {
  char* end = nullptr;
  const long long v = strtoll(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "jxltran: expected integer, got '%s'\n", s);
    return false;
  }
  *out = static_cast<int64_t>(v);
  return true;
}

static bool AppendSetExtraChannelSpec(const char* s,
                                      std::vector<std::string>* out) {
  if (s == nullptr || s[0] == '\0') {
    fprintf(stderr,
            "jxltran: --set-extra-channel expects INDEX:key=value,... (see "
            "--help)\n");
    return false;
  }
  out->push_back(s);
  return true;
}

static bool ExtraChannelTypeFromSlug(const char* slug,
                                     jxltran::ExtraChannelType* out) {
  if (strcasecmp(slug, "alpha") == 0) {
    *out = jxltran::ExtraChannelType::kAlpha;
    return true;
  }
  if (strcasecmp(slug, "depth") == 0) {
    *out = jxltran::ExtraChannelType::kDepth;
    return true;
  }
  if (strcasecmp(slug, "spot") == 0) {
    *out = jxltran::ExtraChannelType::kSpotColour;
    return true;
  }
  if (strcasecmp(slug, "selection") == 0) {
    *out = jxltran::ExtraChannelType::kSelectionMask;
    return true;
  }
  if (strcasecmp(slug, "black") == 0) {
    *out = jxltran::ExtraChannelType::kBlack;
    return true;
  }
  if (strcasecmp(slug, "cfa") == 0) {
    *out = jxltran::ExtraChannelType::kCFA;
    return true;
  }
  if (strcasecmp(slug, "thermal") == 0) {
    *out = jxltran::ExtraChannelType::kThermal;
    return true;
  }
  if (strcasecmp(slug, "optional") == 0) {
    *out = jxltran::ExtraChannelType::kOptional;
    return true;
  }
  if (strcasecmp(slug, "nonoptional") == 0) {
    *out = jxltran::ExtraChannelType::kNonOptional;
    return true;
  }
  fprintf(stderr,
          "jxltran: --set-extra-channel: unknown type '%s' (use alpha, depth, "
          "spot, selection, black, cfa, thermal, optional, nonoptional)\n",
          slug);
  return false;
}

static bool ParseOneExtraChannelPatch(const char* spec,
                                      jxltran::ExtraChannelHeaderPatch* p) {
  *p = jxltran::ExtraChannelHeaderPatch{};
  const char* colon = std::strchr(spec, ':');
  if (colon == nullptr) {
    fprintf(stderr,
            "jxltran: --set-extra-channel: expected INDEX:key=value,... got "
            "'%s'\n",
            spec);
    return false;
  }
  char* end = nullptr;
  unsigned long idx = std::strtoul(spec, &end, 10);
  if (end != colon) {
    fprintf(stderr,
            "jxltran: --set-extra-channel: bad channel index before ':' in "
            "'%s'\n",
            spec);
    return false;
  }
  if (idx > static_cast<unsigned long>((std::numeric_limits<size_t>::max)())) {
    fprintf(stderr, "jxltran: --set-extra-channel: index out of range\n");
    return false;
  }
  p->index = static_cast<size_t>(idx);
  const char* q = colon + 1;
  while (*q != '\0') {
    while (*q == ' ' || *q == '\t' || *q == ',') ++q;
    if (*q == '\0') break;
    const char* eq = std::strchr(q, '=');
    if (eq == nullptr) {
      fprintf(stderr,
              "jxltran: --set-extra-channel: expected key=value in '%s'\n", spec);
      return false;
    }
    std::string key(q, eq);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
      key.pop_back();
    }
    const char* val = eq + 1;
    const char* val_end = val;
    while (*val_end != '\0' && *val_end != ',') ++val_end;
    std::string v(val, val_end);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) {
      v.pop_back();
    }
    while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) {
      v.erase(v.begin());
    }
    q = val_end;
    if (strcasecmp(key.c_str(), "dim_shift") == 0) {
      fprintf(stderr,
              "jxltran: --set-extra-channel: dim_shift cannot be changed "
              "(per-channel subsampling is fixed in the codestream)\n");
      return false;
    }
    if (strcasecmp(key.c_str(), "d_alpha") == 0) {
      if (v != "0" && v != "1") {
        fprintf(stderr,
                "jxltran: --set-extra-channel: d_alpha must be 0 or 1\n");
        return false;
      }
      p->set_d_alpha = true;
      p->d_alpha = (v == "1");
      continue;
    }
    if (strcasecmp(key.c_str(), "type") == 0) {
      jxltran::ExtraChannelType t;
      if (!ExtraChannelTypeFromSlug(v.c_str(), &t)) return false;
      p->set_type = true;
      p->type = t;
      continue;
    }
    if (strcasecmp(key.c_str(), "bits") == 0) {
      char* ve = nullptr;
      unsigned long b = std::strtoul(v.c_str(), &ve, 10);
      if (ve == v.c_str() || *ve != '\0' || b > 32u) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: bits= expects integer 1..32\n");
        return false;
      }
      p->set_ec_bits_per_sample = true;
      p->bit_depth.bits_per_sample = static_cast<uint32_t>(b);
      continue;
    }
    if (strcasecmp(key.c_str(), "float") == 0) {
      if (v != "0" && v != "1") {
        fprintf(stderr,
                "jxltran: --set-extra-channel: float= must be 0 or 1\n");
        return false;
      }
      p->set_ec_float_sample = true;
      p->bit_depth.float_sample = (v == "1");
      continue;
    }
    if (strcasecmp(key.c_str(), "exp") == 0) {
      char* ve = nullptr;
      unsigned long e = std::strtoul(v.c_str(), &ve, 10);
      if (ve == v.c_str() || *ve != '\0' || e > 15u) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: exp= expects small integer "
                "(float exponent bits)\n");
        return false;
      }
      p->set_ec_exp_bits = true;
      p->bit_depth.exp_bits = static_cast<uint32_t>(e);
      continue;
    }
    if (strcasecmp(key.c_str(), "assoc") == 0 ||
        strcasecmp(key.c_str(), "alpha_associated") == 0) {
      if (v != "0" && v != "1") {
        fprintf(stderr,
                "jxltran: --set-extra-channel: assoc= must be 0 or 1\n");
        return false;
      }
      p->set_alpha_associated = true;
      p->alpha_associated = (v == "1");
      continue;
    }
    if (strcasecmp(key.c_str(), "name_hex") == 0) {
      if ((v.size() % 2) != 0) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: name_hex must have even length\n");
        return false;
      }
      p->set_name = true;
      p->name.clear();
      for (size_t i = 0; i < v.size(); i += 2) {
        const int hi = HexNibbleForFrameName(v[i]);
        const int lo = HexNibbleForFrameName(v[i + 1]);
        if (hi < 0 || lo < 0) {
          fprintf(stderr,
                  "jxltran: --set-extra-channel: invalid hex in name_hex\n");
          return false;
        }
        p->name.push_back(static_cast<uint8_t>((hi << 4) | lo));
      }
      continue;
    }
    if (strcasecmp(key.c_str(), "cfa") == 0) {
      char* ve = nullptr;
      unsigned long c = std::strtoul(v.c_str(), &ve, 10);
      if (ve == v.c_str() || *ve != '\0' || c == 0ul) {
        fprintf(stderr, "jxltran: --set-extra-channel: cfa= expects uint >= 1\n");
        return false;
      }
      p->set_cfa_channel = true;
      p->cfa_channel = static_cast<uint32_t>(c);
      continue;
    }
    if (strcasecmp(key.c_str(), "spot_r") == 0) {
      char* ve = nullptr;
      const double d = std::strtod(v.c_str(), &ve);
      if (ve == v.c_str() || *ve != '\0' || !std::isfinite(d)) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: spot_r= expects a finite float\n");
        return false;
      }
      p->set_spot_r = true;
      p->spot_r = static_cast<float>(d);
      continue;
    }
    if (strcasecmp(key.c_str(), "spot_g") == 0) {
      char* ve = nullptr;
      const double d = std::strtod(v.c_str(), &ve);
      if (ve == v.c_str() || *ve != '\0' || !std::isfinite(d)) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: spot_g= expects a finite float\n");
        return false;
      }
      p->set_spot_g = true;
      p->spot_g = static_cast<float>(d);
      continue;
    }
    if (strcasecmp(key.c_str(), "spot_b") == 0) {
      char* ve = nullptr;
      const double d = std::strtod(v.c_str(), &ve);
      if (ve == v.c_str() || *ve != '\0' || !std::isfinite(d)) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: spot_b= expects a finite float\n");
        return false;
      }
      p->set_spot_b = true;
      p->spot_b = static_cast<float>(d);
      continue;
    }
    if (strcasecmp(key.c_str(), "spot_solidity") == 0 ||
        strcasecmp(key.c_str(), "spot_a") == 0) {
      char* ve = nullptr;
      const double d = std::strtod(v.c_str(), &ve);
      if (ve == v.c_str() || *ve != '\0' || !std::isfinite(d)) {
        fprintf(stderr,
                "jxltran: --set-extra-channel: %s= expects a finite float\n",
                key.c_str());
        return false;
      }
      p->set_spot_solidity = true;
      p->spot_solidity = static_cast<float>(d);
      continue;
    }
    fprintf(stderr,
            "jxltran: --set-extra-channel: unknown key '%s' (keys: d_alpha, "
            "type, bits, float, exp, assoc, name_hex, cfa, spot_r, spot_g, "
            "spot_b, spot_solidity)\n",
            key.c_str());
    return false;
  }
  return true;
}

struct SetTransferArg {
  bool set = false;
  uint32_t transfer_function = 13;
};

bool ParseSetTransferOpt(const char* arg, SetTransferArg* out) {
  SetTransferArg* p = out;
  std::string s = arg;
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "709") p->transfer_function = 1;
  else if (s == "unknown") p->transfer_function = 2;
  else if (s == "linear") p->transfer_function = 8;
  else if (s == "srgb") p->transfer_function = 13;
  else if (s == "pq") p->transfer_function = 16;
  else if (s == "dci") p->transfer_function = 17;
  else if (s == "hlg") p->transfer_function = 18;
  else {
    fprintf(stderr,
            "jxltran: --set-transfer: invalid transfer function '%s'. Valid "
            "values: 709, unknown, linear, srgb, pq, dci, hlg.\n",
            arg);
    return false;
  }
  p->set = true;
  return true;
}


struct SetWhitePointArg {
  bool set = false;
  uint32_t white_point = 1;
};
bool ParseSetWhitePointOpt(const char* arg, SetWhitePointArg* out) {
  std::string s = arg;
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "d65") out->white_point = 1;
  else if (s == "e") out->white_point = 10;
  else if (s == "dci") out->white_point = 11;
  else {
    fprintf(stderr, "jxltran: --set-white-point: invalid value '%s'. Valid: d65, e, dci.\n", arg);
    return false;
  }
  out->set = true;
  return true;
}

struct SetPrimariesArg {
  bool set = false;
  uint32_t primaries = 1;
};
bool ParseSetPrimariesOpt(const char* arg, SetPrimariesArg* out) {
  std::string s = arg;
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "srgb") out->primaries = 1;
  else if (s == "2100") out->primaries = 9;
  else if (s == "p3") out->primaries = 11;
  else {
    fprintf(stderr, "jxltran: --set-primaries: invalid value '%s'. Valid: srgb, 2100, p3.\n", arg);
    return false;
  }
  out->set = true;
  return true;
}

struct SetRenderingIntentArg {
  bool set = false;
  uint32_t rendering_intent = 1;
};
bool ParseSetRenderingIntentOpt(const char* arg, SetRenderingIntentArg* out) {
  std::string s = arg;
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "perceptual") out->rendering_intent = 0;
  else if (s == "relative") out->rendering_intent = 1;
  else if (s == "saturation") out->rendering_intent = 2;
  else if (s == "absolute") out->rendering_intent = 3;
  else {
    fprintf(stderr, "jxltran: --set-rendering-intent: invalid value '%s'. Valid: perceptual, relative, saturation, absolute.\n", arg);
    return false;
  }
  out->set = true;
  return true;
}

struct Args {
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool info = false;
  bool info_structure = false;
  ContainerOpt container = ContainerOpt::kAsIs;
  JxlpOpt jxlp = JxlpOpt::kAsIs;
  uint32_t strip = 0;
  BoxOrderArg box_order;
  BrobArg brob;
  uint32_t set_orientation = 0;      // 0 = no change; 1-8 = new orientation
  uint32_t rel_orientation = 0;    // 0 = no change; 1-8 = compose after current
  uint32_t set_bits_per_sample = 0;  // 0 = no change
  MainFloatArg main_float;
  MainExpArg main_exp;
  NumLoopsArg num_loops;
  TpsArg tps;
  IntensityTargetArg intensity_target;
  std::vector<std::string> set_extra_channel_specs;
  OpsinFloatOpt opsin_exposure;
  OpsinFloatOpt opsin_temperature;
  OpsinFloatOpt opsin_tint;
  OpsinFloatOpt opsin_hue;
  OpsinFloatOpt opsin_bias_x;
  OpsinFloatOpt opsin_bias_y;
  OpsinFloatOpt opsin_bias_b;
  OpsinFloatOpt opsin_quant_bias_0;
  OpsinFloatOpt opsin_quant_bias_1;
  OpsinFloatOpt opsin_quant_bias_2;
  OpsinFloatOpt opsin_quant_bias_3;
  bool opsin_inverse = false;
  CropArg crop;
  FramesArg frames;
  KeepFramesArg keep_frames;
  bool keep_listed_frames = false;
  SetFrameBlendsArg set_frame_blends;
  SetFrameDurationsArg set_frame_durations;
  SetFrameNamesArg set_frame_names;
  SetFrameRegionsArg set_frame_regions;
  jxltran::TocGroupOrderCli group_order = jxltran::TocGroupOrderCli::kKeep;
  int64_t center_x = -1;
  int64_t center_y = -1;
  PhotonIsoArg photon_iso;
  const char* photon_weights = nullptr;
  LfDcQuantMulArg lf_dc_quant_mul;
  LfGlobalScaleArg lf_global_scale;
  EpfItersArg epf_iters;
  EpfAmpScaleArg epf_amp_scale;
  EpfUniformArg epf_uniform;
  const char* gab_blur = nullptr;
  const char* gab_sharpen = nullptr;
  const char* gab_weights = nullptr;
  const char* extract_icc = nullptr;
  const char* extract_exif = nullptr;
  const char* extract_xmp = nullptr;
  const char* extract_jumbf = nullptr;
  const char* extract_splines = nullptr;
  const char* set_splines_from = nullptr;
  FramesArg clear_splines_frames;
  LfChunk0TailTrimArg lf_chunk0_tail_trim;
  const char* set_exif = nullptr;
  const char* set_xmp = nullptr;
  const char* set_jumbf = nullptr;
  bool verbose = false;
  bool check_reversible = false;
  const char* append_jxl = nullptr;
  bool append_jxl_skip_compat_check = false;
  bool append_dummy_tail = false;
  bool strip_icc = false;
  SetTransferArg set_transfer;
  SetWhitePointArg set_white_point;
  SetPrimariesArg set_primaries;
  SetRenderingIntentArg set_rendering_intent;

  int GabOptionCount() const {
    return (gab_blur != nullptr) + (gab_sharpen != nullptr) +
           (gab_weights != nullptr);
  }

  bool NeedsHeaderMod() const {
    return set_orientation != 0 || rel_orientation != 0 ||
           set_bits_per_sample != 0 || main_float.set || main_exp.set ||
           num_loops.set || tps.set ||
           intensity_target.set || !set_extra_channel_specs.empty() ||
           set_transfer.set || strip_icc ||
           set_white_point.set || set_primaries.set || set_rendering_intent.set;
  }

  bool NeedsOpsinSliderArgs() const {
    return opsin_exposure.set || opsin_temperature.set || opsin_tint.set ||
           opsin_hue.set;
  }

  bool NeedsOpsinAdjust() const {
    return NeedsOpsinSliderArgs() || opsin_bias_x.set || opsin_bias_y.set ||
           opsin_bias_b.set || opsin_quant_bias_0.set || opsin_quant_bias_1.set ||
           opsin_quant_bias_2.set || opsin_quant_bias_3.set;
  }

  bool NeedsPhotonNoiseIso() const { return photon_iso.set; }

  bool NeedsPhotonNoiseWeights() const { return photon_weights != nullptr; }

  bool NeedsPhotonNoiseAny() const {
    return NeedsPhotonNoiseIso() || NeedsPhotonNoiseWeights();
  }

  bool NeedsLfDcQuantMul() const { return lf_dc_quant_mul.set; }

  bool NeedsLfGlobalScale() const { return lf_global_scale.set; }

  bool NeedsEpfIters() const { return epf_iters.set; }

  bool NeedsEpfAmpScale() const { return epf_amp_scale.set; }

  bool NeedsEpfUniformity() const { return epf_uniform.set; }

  bool NeedsSplinesSet() const { return set_splines_from != nullptr; }

  bool NeedsClearSplinesFrames() const { return clear_splines_frames.set; }

  bool NeedsLfGlobalChunk0TailTrim() const { return lf_chunk0_tail_trim.set; }

  bool NeedsKeepListedFrames() const { return keep_listed_frames; }

  bool NeedsSetFrameBlends() const { return set_frame_blends.set; }

  bool NeedsSetFrameDurations() const { return set_frame_durations.set; }

  bool NeedsSetFrameNames() const { return set_frame_names.set; }

  bool NeedsSetFrameRegions() const { return set_frame_regions.set; }

  void AddCommandLineOptions(jpegxl::tools::CommandLineParser* cmdline) {
    cmdline->AddPositionalOption("INPUT", /*required=*/true,
                                 "The JPEG XL input file.", &file_in);
    cmdline->AddPositionalOption(
        "OUTPUT", /*required=*/false,
        "The JPEG XL output file (omit with --info, --info-structure, "
        "--extract-*, and other read-only modes).",
        &file_out);
    cmdline->AddHelpText("\nInspect and extract (no OUTPUT file):", 0);
    cmdline->AddOptionFlag(
        '\0', "info",
        "Print oriented display dimensions, animation flag, and one "
        "frame_region line per frame (WxH+X+Y in display space, same "
        "orientation as dimensions), then exit. No OUTPUT file; cannot be "
        "combined with transform or remux options.",
        &info, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionFlag(
        '\0', "info-structure",
        "Print one JSON object with BMFF box byte spans, codestream fragment "
        "mapping, and per-frame header / TOC / body / TOC-section layout "
        "(codestream byte offsets), then exit. No OUTPUT file; same "
        "combination rules as --info (cannot be combined with --info).",
        &info_structure, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionValue(
        '\0', "extract-icc", "FILE",
        "Write the codestream-embedded ICC profile to FILE (when present) and "
        "exit. No OUTPUT file; same combination rules as --info (only "
        "optional -v/--verbose).",
        &extract_icc, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "extract-exif", "FILE",
        "Write the first Exif box payload to FILE (raw TIFF Exif block) and "
        "exit.\n"
        "brob-wrapped Exif is decompressed when jxltran is built with brotli. "
        "No\n"
        "OUTPUT file; same combination rules as --extract-icc.",
        &extract_exif, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "extract-xmp", "FILE",
        "Write the first XMP metadata box (type xml ) payload to FILE and "
        "exit.\n"
        "brob-wrapped XMP is decompressed when built with brotli. No OUTPUT; "
        "same\n"
        "rules as --extract-icc.",
        &extract_xmp, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "extract-jumbf", "FILE",
        "Write the first JUMBF box (type jumb) payload to FILE and exit.\n"
        "brob-wrapped JUMBF is decompressed when built with brotli. No OUTPUT; "
        "same\n"
        "rules as --extract-icc.",
        &extract_jumbf, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "extract-splines", "FILE",
        "Write LF-global spline geometry and quantized coefficients as text "
        "and exit.\n"
        "Format matches --set-splines-from. Optional --frames selects codestream "
        "frames.\n"
        "No OUTPUT; same combination rules as --extract-icc.",
        &extract_splines, StoreExtractIccPath);
    cmdline->AddHelpText("\nContainer and metadata boxes:", 0);
    cmdline->AddOptionValue(
        '\0', "container", "keep|no|yes|if-needed",
        "Container output mode (default: keep).\n"
        "    keep      : preserve input format (container or bare "
        "codestream).\n"
        "    no        : output only the bare codestream. Drops the ISOBMFF\n"
        "                wrapper and all metadata boxes (Exif, XMP, JUMBF, "
        "…).\n"
        "                Multiple jxlp payloads are concatenated in counter\n"
        "                order, producing the same codestream bytes as after\n"
        "                --jxlp merge (merge is implied; --jxlp is ignored "
        "for\n"
        "                bare output).\n"
        "    yes       : always wrap in a container.\n"
        "    if-needed : use container only when metadata boxes are present.",
        &container, ParseContainerOpt);
    cmdline->AddOptionValue(
        '\0', "jxlp", "keep|sort|merge",
        "jxlp box handling when the output is a container (default: keep).\n"
        "    keep  : keep jxlp boxes unchanged in the written container.\n"
        "    sort  : reorder jxlp boxes by counter (normalize OOO delivery).\n"
        "    merge : sort and merge all jxlp into a single jxlc box.",
        &jxlp, ParseJxlpOpt);
    cmdline->AddOptionValue(
        '\0', "strip", "exif|xmp|jumbf|jbrd|all[,...]",
        "Strip metadata boxes from the container (comma-separated list).\n"
        "    exif  : strip Exif boxes (and brob wrappers around them).\n"
        "    xmp   : strip XMP boxes (xml  type).\n"
        "    jumbf : strip JUMBF boxes.\n"
        "    jbrd  : strip JPEG bitstream reconstruction data.\n"
        "    all   : strip all non-structural metadata.",
        &strip, ParseStripOpt);
    cmdline->AddOptionValue(
        '\0', "box-order",
        "keep|before-codestream|after-codestream|TYPE[,TYPE...]",
        "Position of boxes in the output container (default: keep).\n"
        "    keep               : preserve input box order.\n"
        "    before-codestream  : all metadata precedes the codestream.\n"
        "    after-codestream   : all metadata follows the codestream.\n"
        "    TYPE[,TYPE...]     : explicit order (e.g. exif,jbrd,jxlc,xmp);\n"
        "                         unspecified boxes are appended at the end.\n"
        "    Known aliases: exif, xmp, jumbf, jbrd, jxlc, jxlp, jxli, jxll,\n"
        "                   brob, ftyp; or any raw 4-char box type.",
        &box_order, ParseBoxOrderOpt);
    cmdline->AddOptionValue(
        '\0', "brob", "keep|compress[:TYPE[,...]]|decompress",
        "brob (brotli-compressed box) handling (default: keep)."
#ifdef JXLTRAN_HAVE_BROTLI
        "\n"
        "    keep               : preserve brob boxes unchanged.\n"
        "    compress           : compress all eligible metadata\n"
        "                         (Exif/XMP/JUMBF) into brob format.\n"
        "    compress:TYPE[,...]: compress only the listed types,\n"
        "                         e.g. compress:xmp or compress:xmp,jumbf.\n"
        "    decompress         : decompress brob boxes to their original "
        "form.",
#else
        "\n    (brotli support not compiled in; only keep is effective)",
#endif
        &brob, ParseBrobOpt);
    cmdline->AddHelpText("\nReplace metadata from files:", 0);
    cmdline->AddOptionValue(
        '\0', "set-exif", "FILE",
        "Replace (or add) the Exif metadata box with the contents of FILE. "
        "Removes\n"
        "any existing Exif and brob-wrapped Exif boxes. Bare codestream input is "
        "wrapped\n"
        "in a minimal container.",
        &set_exif, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "set-xmp", "FILE",
        "Replace (or add) the XMP box (xml ) with the contents of FILE; same "
        "semantics\n"
        "as --set-exif.",
        &set_xmp, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "set-jumbf", "FILE",
        "Replace (or add) the JUMBF box (jumb) with the contents of FILE; same "
        "semantics\n"
        "as --set-exif.",
        &set_jumbf, StoreExtractIccPath);
    cmdline->AddHelpText("\nImage header:", 0);
    cmdline->AddOptionValue('\0', "set-orientation", "1..8|90|180|270",
                            "Set the orientation field in the image header.\n"
                            "Orientation values (Exif semantics):\n"
                            "  1 = normal (no transform)\n"
                            "  2 = flip horizontally\n"
                            "  3 = rotate 180 (alias: 180)\n"
                            "  4 = flip vertically\n"
                            "  5 = transpose (rotate 90 CW + flip H)\n"
                            "  6 = rotate 90 clockwise (alias: 90)\n"
                            "  7 = flip H + rotate 90 CW\n"
                            "  8 = rotate 90 counter-clockwise (alias: 270)\n"
                            "Aliases 90, 180, 270 are degrees of rotation clockwise.",
                            &set_orientation, ParseOrientationOpt);
    cmdline->AddOptionValue(
        '\0', "set-orientation-relative", "1..8|90|180|270",
        "Like --set-orientation, but |delta| is composed *after* the file's\n"
        "current orientation tag (Exif semantics). Example: if the file is\n"
        "90° CCW (8) and you pass --set-orientation-relative=6 (90° CW), the\n"
        "result is identity (1). Value 1 leaves the orientation tag unchanged.\n"
        "Rotation aliases 90, 180, 270 are clockwise degrees (same as\n"
        "--set-orientation). Mutually exclusive with --set-orientation.",
        &rel_orientation, ParseOrientationOpt);
    cmdline->AddOptionValue(
        '\0', "set-bits-per-sample", "1..32",
        "Set the bits_per_sample field in the image header (1..32).\n"
        "For XYB-encoded images this is primarily a decoder output-precision\n"
        "hint and does not change compressed XYB coefficients. For non-XYB\n"
        "images it is part of the coded representation and may change decoded\n"
        "samples when altered. Combine with --set-main-float and --set-main-exp\n"
        "for float main samples.",
        &set_bits_per_sample, ParseBitsPerSampleOpt);
    cmdline->AddOptionValue(
        '\0', "set-main-float", "0|1",
        "Set the main image header float_sample flag (0 = integer, 1 = float).\n"
        "When set to 1, use --set-main-exp for exponent bits (2..8). Validates\n"
        "the resulting BitDepth bundle like extra channels.",
        &main_float, ParseMainFloatOpt);
    cmdline->AddOptionValue(
        '\0', "set-main-exp", "2..8",
        "Set exponent bits for float main image samples (only meaningful when\n"
        "float_sample is true).",
        &main_exp, ParseMainExpOpt);
    cmdline->AddOptionValue(
        '\0', "set-num-loops", "N",
        "Set the animation loop count (0 = loop forever).\n"
        "Only valid for animated images (have_animation=true).",
        &num_loops, ParseNumLoopsOpt);
    cmdline->AddOptionValue(
        '\0', "set-tps", "N[/D]|P%",
        "Set the animation ticks-per-second.\n"
        "  Absolute: N or N/D (positive integers), e.g. 24, 25, 30, 60, or\n"
        "            30000/1001 (NTSC).\n"
        "  Relative: P% scales the file's current TPS (from the image "
        "header),\n"
        "            e.g. 50% for half speed, 200% for double.\n"
        "Only valid for animated images (have_animation=true).",
        &tps, ParseTpsOpt);
    cmdline->AddOptionValue(
        '\0', "set-intensity-target", "NITS|default",
        "Set ToneMapping intensity_target (upper bound on scene intensity, in "
        "nits).\n"
        "    Pass 'default' to store the implicit default bundle (255 nits with "
        "libjxl's other tone-mapping defaults), which omits an explicit "
        "ToneMapping\n"
        "    block when the rest of the header matches. If the image header is "
        "packed\n"
        "    (all_default), jxltran expands it in memory to decoder defaults so "
        "this flag\n"
        "    can still be applied.",
        &intensity_target, ParseIntensityTargetOpt);
    cmdline->AddOptionValue(
        '\0', "set-extra-channel", "INDEX:key=val,...",
        "Edit one ImageMetadata ExtraChannelInfo bundle (JPEG XL part1 image "
        "header).\n"
        "    Format: 0-based index, then comma-separated key=value pairs. Keys: "
        "d_alpha (0|1 compact default-alpha),\n"
        "    type (alpha|depth|spot|selection|black|cfa|thermal|optional|"
        "nonoptional), bits, float (0|1), exp (float exponent bits),\n"
        "    assoc / alpha_associated (0|1), name_hex (even hex UTF-8), cfa "
        "(uint), spot_r, spot_g, spot_b, spot_solidity / spot_a.\n"
        "    num_extra and dim_shift cannot be changed. Repeat the flag for "
        "multiple channels.",
        &set_extra_channel_specs, AppendSetExtraChannelSpec);
    cmdline->AddOptionValue(
        '\0', "opsin-exposure", "EV",
        "XYB only: exposure in stops (linear RGB after XYB→RGB). Reversible via\n"
        "    a non-default OpsinInverseMatrix (CustomTransformData). Range about "
        "±10.\n"
        "    Combine with --opsin-temperature / --opsin-tint / --opsin-hue. "
        "reversible_undo\n"
        "    uses --opsin-inverse with the same magnitudes (not negated).",
        &opsin_exposure, ParseOpsinExposureOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-temperature", "T",
        "XYB only: white balance warmth in arbitrary units −100..+100 (warm = "
        "positive).\n"
        "    Applied on the decoder inverse opsin matrix (like a Lightroom "
        "Temperature slider).",
        &opsin_temperature, ParseOpsinTempOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-tint", "T",
        "XYB only: green vs magenta tint, −100..+100 (green = positive).",
        &opsin_tint, ParseOpsinTintOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-hue", "T",
        "XYB only: HSL-like hue shift: rotate decoder linear RGB about Rec. 709 "
        "luma,\n"
        "    −100..+100 maps to −90°..+90° (avoids ±180° ambiguity).",
        &opsin_hue, ParseOpsinHueOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-bias-x", "B",
        "XYB only: add a delta to the X opsin bias field (CustomTransformData; "
        "about −0.003..+0.003; small values have a large decoder effect).\n"
        "    reversible_undo emits the additive inverse (implicit-default minus "
        "stored F16) with --check_reversible.",
        &opsin_bias_x, ParseOpsinBiasXOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-bias-y", "D",
        "XYB only: add a delta to the Y opsin bias field (about −0.003..+0.003).",
        &opsin_bias_y, ParseOpsinBiasYOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-bias-b", "D",
        "XYB only: add a delta to the B opsin bias field (about −0.03..+0.03).",
        &opsin_bias_b, ParseOpsinBiasBOpt);
    cmdline->AddOptionValue(
        '\0', "opsin-quant-bias-0", "D",
        "XYB only: add a delta to quant_biases[0] (about −2..+2).",
        &opsin_quant_bias_0, ParseOpsinQuantBias0Opt);
    cmdline->AddOptionValue(
        '\0', "opsin-quant-bias-1", "D",
        "XYB only: add a delta to quant_biases[1] (about −2..+2).",
        &opsin_quant_bias_1, ParseOpsinQuantBias1Opt);
    cmdline->AddOptionValue(
        '\0', "opsin-quant-bias-2", "D",
        "XYB only: add a delta to quant_biases[2] (about −2..+2).",
        &opsin_quant_bias_2, ParseOpsinQuantBias2Opt);
    cmdline->AddOptionValue(
        '\0', "opsin-quant-bias-3", "D",
        "XYB only: add a delta to quant_biases[3] (about −2..+2).",
        &opsin_quant_bias_3, ParseOpsinQuantBias3Opt);
    cmdline->AddOptionValue(
        '\0', "set-transfer", "TF",
        "Force a specific transfer curve in ColourEncoding (discards custom gamma).\n"
        "    Requires --strip-icc if the image has an embedded ICC profile.\n"
        "    Valid values: 709, unknown, linear, srgb, pq, dci, hlg.",
        &set_transfer, ParseSetTransferOpt);
    cmdline->AddOptionFlag(
        '\0', "strip-icc",
        "Remove embedded ICC profile if present (forces use of color encoding enums).",
        &strip_icc, &jpegxl::tools::SetBooleanTrue);

    cmdline->AddOptionValue(
        '\0', "set-white-point", "WP",
        "Force a specific white point in ColourEncoding.\n"
        "    Requires --strip-icc if the image has an embedded ICC profile.\n"
        "    Valid values: d65, e, dci.",
        &set_white_point, ParseSetWhitePointOpt);
    cmdline->AddOptionValue(
        '\0', "set-primaries", "PR",
        "Force specific primaries in ColourEncoding.\n"
        "    Requires --strip-icc if the image has an embedded ICC profile.\n"
        "    Valid values: srgb, 2100, p3.",
        &set_primaries, ParseSetPrimariesOpt);
    cmdline->AddOptionValue(
        '\0', "set-rendering-intent", "RI",
        "Force a specific rendering intent in ColourEncoding.\n"
        "    Requires --strip-icc if the image has an embedded ICC profile.\n"
        "    Valid values: perceptual, relative, saturation, absolute.",
        &set_rendering_intent, ParseSetRenderingIntentOpt);
    cmdline->AddOptionFlag(
        '\0', "opsin-inverse",
        "XYB only: apply the exact inverse of a prior opsin slider pass. Use the "
        "same\n"
        "    --opsin-exposure / --opsin-temperature / --opsin-tint / --opsin-hue "
        "*values*\n"
        "    as the forward run (do not negate). At least one of those sliders "
        "must be set.",
        &opsin_inverse, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddHelpText("\nCrop:", 0);
    cmdline->AddOptionValue(
        '\0', "crop", "WxH or WxHXY",
        "Reversible metadata-only canvas resize / crop. WxH is the new canvas\n"
        "size. X and Y are optional signed offsets: output pixel (px,py) "
        "shows\n"
        "input pixel (px+X, py+Y). Coordinates are in display space (as "
        "decoders\n"
        "such as djxl show the image after applying the orientation tag); "
        "when\n"
        "the tag is identity, this matches codestream pixel coordinates. "
        "Any\n"
        "signed combination is allowed (e.g. 500x500+200-300 on a 600x600 "
        "image\n"
        "can add padding on the right/bottom). Encoded frame dimensions stay "
        "as in\n"
        "the bitstream; when a frame has no crop, it is treated as "
        "full-canvas for\n"
        "positioning. LF and ReferenceOnly frames are not modified. No "
        "compressed\n"
        "sample data is re-encoded.",
        &crop, ParseCropOpt);
    cmdline->AddHelpText("\nFrames (selection, order, append):", 0);
    cmdline->AddOptionValue(
        '\0', "frames", "N[,N...]",
        "Apply the frame-level operations below only to the listed codestream "
        "frame\n"
        "indices (0-based, in codestream order; match the first number on each\n"
        "`frame_region:` line from --info). Omit --frames to affect every "
        "regular or\n"
        "skip-progressive frame (LF and reference-only frames are always left "
        "unchanged).\n"
        "With --keep-listed-frames and no --keep-frames, the same list selects "
        "which\n"
        "frames to retain (legacy); prefer --keep-frames for that case so "
        "--frames can\n"
        "still mean “filter only” independently.",
        &frames, ParseFramesArg);
    cmdline->AddOptionValue(
        '\0', "keep-frames", "N[,N...]",
        "With --keep-listed-frames only: codestream frame indices to keep, in "
        "output\n"
        "order (reorder and/or subset; first occurrence wins; duplicate indices "
        "are\n"
        "deduplicated). Other frame-level operations still use --frames when "
        "given.\n"
        "When --keep-listed-frames is set and --keep-frames is omitted, "
        "--frames supplies\n"
        "the keep list (backward compatible).",
        &keep_frames, ParseKeepFramesArg);
    cmdline->AddOptionFlag(
        '\0', "keep-listed-frames",
        "Rewrite the codestream to contain only the frames listed in "
        "--keep-frames\n"
        "(or in --frames when --keep-frames is omitted), in the order given "
        "(reorder\n"
        "and/or subset; first occurrence of each index wins; duplicates are "
        "deduplicated).\n"
        "Verbatim compressed frame data is preserved; the last kept frame must "
        "be regular or\n"
        "skip-progressive.\n"
        "Decoding the result can fail if a kept frame still depends on a "
        "dropped reference\n"
        "or blend source. Requires OUTPUT and (--keep-frames or --frames).",
        &keep_listed_frames, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionValue(
        '\0', "append-jxl", "FILE",
        "After reading INPUT, concatenate the codestream frames of FILE after "
        "those of INPUT.\n"
        "The output canvas is max(INPUT, FILE) in width and height; frames that "
        "used implicit full-canvas\n"
        "geometry on the smaller side get an explicit crop to that side's canvas "
        "size at origin (0,0).\n"
        "Compatibility is only: same number of extra channels; both XYB or both "
        "non-XYB;\n"
        "when non-XYB, the main bit depth must match. The last frame of INPUT\n"
        "is rewritten with is_last=false; FILE's image header bytes are "
        "skipped. Decoder output may still\n"
        "fail if appended frames depend on reference state that does not match "
        "the end of INPUT. Requires OUTPUT.",
        &append_jxl, StoreExtractIccPath);
    cmdline->AddOptionFlag(
        '\0', "append-jxl-skip-compat-check",
        "With --append-jxl only: skip the XYB / bit-depth / extra-channel "
        "header compatibility check (may produce invalid output).",
        &append_jxl_skip_compat_check, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionFlag(
        '\0', "append-dummy-tail",
        "Append a fixed minimal 42×3 modular all-zero terminal frame after "
        "INPUT's frames (skips header compatibility vs INPUT). After merge the "
        "previous frame saves to reference slot 1 and the tail uses kAdd with "
        "blending source 1 (zeros added to that slot). "
        "Requires OUTPUT; do not use with --append-jxl.",
        &append_dummy_tail, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddHelpText("\nFrame blend, timing, names, and layout:", 0);
    cmdline->AddOptionValue(
        '\0', "set-frame-blends", "SPEC[,SPEC...]",
        "Per-frame blend overrides. Each SPEC is INDEX:MODE with optional "
        ",FIELD=VAL\n"
        "    tails: alpha=N, clamp=0|1|true|false, source=N, target=N "
        "(comma-separated within one\n"
        "    frame). target= sets save_as_reference (reference slot 0–3) and "
        "is only allowed when\n"
        "    the frame is not last in the codestream. Values are written as "
        "given (decoder behavior may vary if "
        "they do not\n"
        "    match the bitstream rules for the chosen mode). MODE is 0–4 or "
        "replace|add|blend|alpha_weighted_add|mul. Separate frames\n"
        "    with a comma only when the next token looks like a new INDEX: "
        "e.g.\n"
        "    0:blend,alpha=0,clamp=1,1:add\n"
        "    Optional fields apply only when serialized for that mode (see "
        "ReadBlendingInfo).",
        &set_frame_blends, ParseSetFrameBlendsArg);
    cmdline->AddOptionValue(
        '\0', "set-frame-durations", "INDEX:TICKS[,INDEX:TICKS...]",
        "Set frame duration (animation ticks) on regular / skip-progressive "
        "frames.\n"
        "Requires have_animation in the codestream.",
        &set_frame_durations, ParseSetFrameDurationsArg);
    cmdline->AddOptionValue(
        '\0', "set-frame-names", "INDEX:VALUE[,INDEX:VALUE...]",
        "Set per-frame name bytes (UTF-8) on regular / skip-progressive "
        "frames.\n"
        "    VALUE : even-length hex (same encoding as `name_hex=` from "
        "--info), an unquoted\n"
        "            literal when it is not even-length all-hex (no spaces), "
        "or a double-quoted\n"
        "            literal for spaces, commas, hex-like ASCII, etc. "
        "(use \\\" and \\\\ inside quotes).\n"
        "            Use INDEX: with nothing after ':' to clear. Indices "
        "refer to codestream order\n"
        "            after other edits in this run (e.g. after "
        "--keep-listed-frames).",
        &set_frame_names, ParseSetFrameNamesArg);
    cmdline->AddOptionValue(
        '\0', "set-frame-region", "INDEX:WxH+X+Y[,INDEX:WxH+X+Y...]",
        "Reposition a regular or skip-progressive frame's display rectangle "
        "(WxH+X+Y in\n"
        "    the same oriented display space as `jxltran --info` and --crop). "
        "Width and height\n"
        "    must match the frame's current display size (move only). Applied "
        "before --crop in\n"
        "    this run. LF and reference-only frames are rejected; a full-canvas "
        "frame cannot\n"
        "    be moved.",
        &set_frame_regions, ParseSetFrameRegionsArg);
    cmdline->AddHelpText(
        "\nCodestream signal (TOC, noise, filtering, splines):", 0);
    cmdline->AddOptionValue(
        '\0', "group_order", "keep|0|1|progressive",
        "TOC section order in regular / skip-progressive frames.\n"
        "    keep         : leave permutation and sizes unchanged.\n"
        "    0            : strip TOC permutation (logical order in the "
        "bitstream).\n"
        "    1            : center-first AC group order (matches cjxl "
        "--group_order=1; rewrites\n"
        "                   stream-order section bytes and the TOC permutation).\n"
        "    progressive  : if stream order is non-progressive (HF spatial "
        "groups appear\n"
        "                   before lf_global and every lf_group have been "
        "signaled),\n"
        "                   strip permutation like 0; otherwise keep (e.g. "
        "center-first\n"
        "                   order that still signals all LF data before HF).",
        &group_order, ParseGroupOrder);
    cmdline->AddOptionValue(
        '\0', "center_x", "X",
        "With --group_order=1, image X coordinate (pixels) for the center used "
        "to pick the\n"
        "starting side of the center-first spiral (-1 = image center, matching "
        "cjxl). Ignored for other modes.",
        &center_x, ParseCenterI64);
    cmdline->AddOptionValue(
        '\0', "center_y", "Y",
        "With --group_order=1, image Y coordinate for the spiral center (-1 = "
        "image center).\n"
        "Ignored for other modes.",
        &center_y, ParseCenterI64);
    cmdline->AddOptionValue(
        '\0', "set-photon-noise-iso", "ISO",
        "Rewrite the synthetic noise LUT in regular / skip-progressive frames "
        "(modular\n"
        "or VarDCT), using the same ISO-based model as libjxl "
        "(cjxl --photon_noise_iso).\n"
        "0 clears the kNoise flag and removes the 80-bit LUT from DC global. "
        "Each\n"
        "edited frame must have no patches or splines. When DC-global size "
        "changes,\n"
        "the TOC permutation entropy is copied verbatim from the input and only "
        "the\n"
        "U32 size codewords are re-encoded. Compressed sample values are "
        "unchanged.",
        &photon_iso, ParsePhotonIsoOpt);
    cmdline->AddOptionValue(
        '\0', "set-photon-noise-weights", "WEIGHTS",
        "Like --set-photon-noise-iso but set the 8×10-bit LUT directly: either "
        "eight\n"
        "comma-separated floats (same scale as the ISO model LUT) or 20 hex "
        "digits for\n"
        "the verbatim 10-byte bitstream block. All-zero / empty LUT removes "
        "kNoise.\n"
        "Mutually exclusive with --set-photon-noise-iso on the same run.",
        &photon_weights, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "lf-dc-quant-mul", "X,Y,B",
        "Multiply the three LF channel dequantization factors in DC global "
        "(after photon noise, same bundle as libjxl DequantMatrices::DecodeDC). "
        "Each factor must be a finite positive float; 1 leaves that channel "
        "unchanged. "
        "Default implicit wire values are 1/32, 1/4, 1/2 per channel. Requires a "
        "successful LF-global prefix parse, no spline edit on the frame, and no "
        "TOC strip / center-first shuffle (TOC permutation may be non-empty).",
        &lf_dc_quant_mul, ParseLfDcQuantMulOpt);
    cmdline->AddOptionValue(
        '\0', "lf-global-scale", "N",
        "VarDCT only: set the quantizer global_scale U32 in LF-global (same "
        "QuantizerParams bundle as libjxl immediately after LF dequant). "
        "Integer 1..32768; leaves quant_dc unchanged from the input. Same spline / "
        "TOC-strip constraints as --lf-dc-quant-mul.",
        &lf_global_scale, ParseLfGlobalScaleOpt);
    cmdline->AddOptionValue(
        '\0', "set-epf-iters", "N",
        "Set edge-preserving filter (EPF) iteration count (0–3) in the frame "
        "restoration\n"
        "bundle (same 2-bit field as libjxl). VarDCT fully-specified defaults "
        "use 2;\n"
        "modular defaults use 0. Enabling EPF on modular sets "
        "epf_sigma_for_modular to 1.0\n"
        "when it was unset/zero. LF and reference-only frames are skipped.",
        &epf_iters, ParseEpfItersOpt);
    cmdline->AddOptionValue(
        '\0', "set-epf-amplitude-scale", "F",
        "Scale overall EPF strength by factor F (>1 stronger, <1 weaker).\n"
        "VarDCT: multiplies effective epf_quant_mul (implicit 0.46 when\n"
        "epf_sigma_custom is false). Modular: multiplies epf_sigma_for_modular\n"
        "(implicit 1.0). Values within tolerance of decoder defaults snap to\n"
        "implicit encoding (VarDCT: epf_sigma_custom off; modular: sigma 1.0).\n"
        "Skipped on frames with epf_iters==0 and on LF / reference-only frames.",
        &epf_amp_scale, ParseEpfAmpScaleOpt);
    cmdline->AddOptionValue(
        '\0', "set-epf-uniformity", "U",
        "VarDCT only (ignored on modular). When EPF is on, sets epf_sharp_lut "
        "by mixing\n"
        "the decoder default ramp {0,1/7,…,1} with all ones: U=0 keeps the "
        "implicit\n"
        "ramp (epf_sharp_custom off), U=1 is all ones, U=0.5 is in between. "
        "LF and\n"
        "reference-only frames are skipped.",
        &epf_uniform, ParseEpfUniformOpt);
    cmdline->AddOptionValue(
        '\0', "gab-blur", "A",
        "Reversible Gaborish blur (A>=0). Per XYB channel, s=w1+w2 is moved in\n"
        "logit space (s clamped to [-0.45, 2]); blur and sharpen use opposite\n"
        "steps so the same A cancels. From the default kernel sum, sharpen A=1\n"
        "moves s halfway toward the minimum (max sharp). Weights matching the\n"
        "implicit encoder defaults snap to gab_custom off. Effective baseline\n"
        "matches --info gab_effective: gab off → 0; packed restoration\n"
        "all_default → implicit default gab (libjxl defaults); gab on implicit →\n"
        "same; custom → stored weights.\n"
        "LF and reference-only frames are skipped. Mutually exclusive with\n"
        "--gab-sharpen and --gab-weights.",
        &gab_blur, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "gab-sharpen", "A",
        "Reversible Gaborish sharpen (A>=0): inverse of --gab-blur in logit(s)\n"
        "with the same A. libjxl rejects only near-singular kernels where\n"
        "|1+4*(w1+w2)| < 1e-8 per channel. LF and reference-only frames are\n"
        "skipped. Mutually exclusive with --gab-blur and --gab-weights.",
        &gab_sharpen, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "gab-weights", "x1,x2,y1,y2,b1,b2",
        "Set all six Gaborish weights (comma-separated, unnormalized).\n"
        "Order: gab_x_weight1, gab_x_weight2, gab_y_weight1, gab_y_weight2,\n"
        "gab_b_weight1, gab_b_weight2. Values may be negative if each channel\n"
        "still satisfies |1+4*(w1+w2)| >= 1e-8 after normalization. LF and\n"
        "reference-only frames are skipped. Mutually exclusive with --gab-blur "
        "and\n"
        "--gab-sharpen.",
        &gab_weights, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "set-splines-from", "FILE",
        "Replace LF-global spline bundles from FILE (from --extract-splines), or "
        "insert\n"
        "new splines when the frame has no kSplines flag (the flag is then set). "
        "Requires\n"
        "a successful LF-global prefix parse. Re-encoded spline entropy may use "
        "any\n"
        "bit length; the frame body stays byte-aligned with trailing zero bits "
        "when\n"
        "needed (downstream bit phase may shift). Incompatible with stripped TOC\n"
        "permutation on the same pass.",
        &set_splines_from, StoreExtractIccPath);
    cmdline->AddOptionValue(
        '\0', "clear-splines-frames", "LIST",
        "Remove LF-global spline bundles on the listed codestream frame indices "
        "(comma-\n"
        "separated, 0-based). Inverse of --set-splines-from when those frames had "
        "no splines\n"
        "before insertion. Mutually exclusive with --set-splines-from on the "
        "same pass.\n"
        "Standalone use is not in the reversible-undo scope.",
        &clear_splines_frames, ParseFramesArg);
    cmdline->AddOptionValue(
        '\0', "lf-global-chunk0-tail-trim-bytes", "LIST",
        "After --clear-splines-frames in the same pass, shrink LF-global physical "
        "TOC\n"
        "section 0 by LIST trailing whole zero bytes (syntax: FRAME:BYTES with "
        "BYTES\n"
        "1 or 2, comma-separated). Required for exact byte-for-byte undo when "
        "spline\n"
        "insert/remove left extra zero padding vs the pre-edit codestream. Each "
        "FRAME\n"
        "must be listed in --clear-splines-frames.",
        &lf_chunk0_tail_trim, ParseLfChunk0TailTrimArg);
    cmdline->AddHelpText("\nDebugging:", 0);
    cmdline->AddOptionFlag(
        'v', "verbose",
        "Log codestream header parse and rewrite to stderr.\n"
        "Each line: BYTE+BIT | field | value. Positions are absolute from the "
        "start of the codestream (reads) or from the start of the written "
        "codestream (writes), split as byte offset + bit index within that "
        "byte.",
        &verbose, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionFlag(
        '\0', "check_reversible",
        "After writing OUTPUT, verify lossless codestream round-trip: identity "
        "copy\n"
        "(no codestream-changing flags), or the emitted reversible_undo steps "
        "vs the\n"
        "codestream snapshot before transforms. --container remux (bare or "
        "ISOBMFF)\n"
        "and --group_order TOC rewrites use the same check. Prints "
        "reversible_check:\n"
        "not_reversible | ok | mismatch.",
        &check_reversible, &jpegxl::tools::SetBooleanTrue);
  }
};

}  // namespace

static uint32_t EpfItersEffectiveForInfo(const jxltran::FrameHeader& fh) {
  if (fh.restoration.all_default) {
    return (fh.encoding == jxltran::kFrameEncModular) ? 0u : 2u;
  }
  return std::min(3u, fh.restoration.epf_iters);
}

static bool InfoModeUsesNonDefaultOptions(const Args& a) {
  return a.container != ContainerOpt::kAsIs || a.jxlp != JxlpOpt::kAsIs ||
         a.strip != 0 ||
         a.box_order.order != jxltran::BoxOrder::kAsIs ||
         a.brob.mode != jxltran::BrobOpt::kAsIs || a.set_orientation != 0 ||
         a.rel_orientation != 0 || a.set_bits_per_sample != 0 ||
         a.main_float.set || a.main_exp.set ||
         a.num_loops.set || a.tps.set || a.intensity_target.set ||
         a.crop.set || a.frames.set ||
         a.keep_frames.set || a.keep_listed_frames ||
         a.group_order != jxltran::TocGroupOrderCli::kKeep || a.photon_iso.set ||
         a.photon_weights != nullptr || a.lf_dc_quant_mul.set ||
         a.lf_global_scale.set || a.set_splines_from != nullptr ||
         a.clear_splines_frames.set ||
         a.lf_chunk0_tail_trim.set ||
         a.gab_blur != nullptr ||
         a.gab_sharpen != nullptr || a.gab_weights != nullptr ||
         a.NeedsEpfIters() || a.NeedsEpfAmpScale() || a.NeedsEpfUniformity() ||
         a.NeedsOpsinAdjust() || a.opsin_inverse || a.NeedsSetFrameBlends() ||
         a.NeedsSetFrameDurations() || a.NeedsSetFrameNames() ||
         a.NeedsSetFrameRegions() || a.append_jxl != nullptr ||
         a.append_jxl_skip_compat_check || a.append_dummy_tail ||
         a.check_reversible;
}

static const char* FrameTypeInfoSlug(uint32_t ft) {
  switch (ft) {
    case jxltran::kFrameTypeRegular:
      return "regular";
    case jxltran::kFrameTypeLF:
      return "lf";
    case jxltran::kFrameTypeReferenceOnly:
      return "ref_only";
    case jxltran::kFrameTypeSkipProgressive:
      return "skip_prog";
    default:
      return "unknown";
  }
}

static const char* BlendModeInfoSlug(uint32_t m) {
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
      return "unknown";
  }
}

static bool IccDataColorSpaceIsCmyk(const std::vector<uint8_t>& icc) {
  if (icc.size() < 20) return false;
  auto lower = [](uint8_t b) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(b)));
  };
  return lower(icc[16]) == 'c' && lower(icc[17]) == 'm' &&
         lower(icc[18]) == 'y' && lower(icc[19]) == 'k';
}

static bool MetadataHasExplicitBlackExtra(const jxltran::ImageMetadata& im) {
  const size_t n = static_cast<size_t>(im.num_extra);
  for (size_t i = 0; i < n && i < im.ec_info.size(); ++i) {
    const jxltran::ExtraChannelInfo& ec = im.ec_info[i];
    if (!ec.d_alpha && ec.type == jxltran::ExtraChannelType::kBlack) {
      return true;
    }
  }
  return false;
}

static void PrintJxlInfoStdout(const jxltran::ParsedCodestream& parsed) {
  uint32_t orient = parsed.image.metadata.orientation;
  if (orient < 1 || orient > 8) orient = 1;
  uint32_t dw = 0;
  uint32_t dh = 0;
  const uint32_t canvas_w = parsed.image.size.width;
  const uint32_t canvas_h = parsed.image.size.height;
  jxltran::OrientedCanvasDimensions(orient, canvas_w, canvas_h, &dw, &dh);
  printf("dimensions: %u x %u\n", dw, dh);
  printf("exif_orientation: %u\n", orient);
  printf("animation: %s\n",
         parsed.image.metadata.have_animation ? "yes" : "no");
  printf("xyb_encoded: %s\n",
         parsed.image.metadata.xyb_encoded ? "yes" : "no");
  const jxltran::ImageMetadata& im = parsed.image.metadata;
  const bool icc_cmyk =
      im.colour_encoding.want_icc &&
      IccDataColorSpaceIsCmyk(parsed.image.icc_bytes);
  const char* mclabel = "RGB";
  if (im.xyb_encoded) {
    mclabel = "XYB";
  } else {
    bool any_ycbcr = false;
    bool regular_without_ycbcr = false;
    for (const jxltran::FramedUnit& fu : parsed.frames) {
      const jxltran::FrameHeader& fh = fu.frame;
      if (fh.frame_type != jxltran::kFrameTypeRegular &&
          fh.frame_type != jxltran::kFrameTypeSkipProgressive) {
        continue;
      }
      if (fh.do_YCbCr) {
        any_ycbcr = true;
      } else {
        regular_without_ycbcr = true;
      }
    }
    if (any_ycbcr && regular_without_ycbcr) {
      mclabel = "mixed";
    } else if (any_ycbcr) {
      mclabel = "YCbCr";
    } else if (im.colour_encoding.colour_space == jxltran::ColourSpace::kGrey) {
      mclabel = "Grey";
    } else {
      mclabel = "RGB";
    }
  }
  const bool cmyk_rgb_main =
      icc_cmyk && !im.xyb_encoded && std::strcmp(mclabel, "RGB") == 0;
  const char* mclabel_out = cmyk_rgb_main ? "CMY" : mclabel;
  printf("main_colour: label=%s bits=%" PRIu32 " float_sample=%s",
         mclabel_out, im.bit_depth.bits_per_sample,
         im.bit_depth.float_sample ? "yes" : "no");
  if (im.bit_depth.float_sample) {
    printf(" exp_bits=%" PRIu32, im.bit_depth.exp_bits);
  }
  printf(" want_icc=%s\n", im.colour_encoding.want_icc ? "yes" : "no");
  if (icc_cmyk) {
    fputs("icc_data_space: CMYK\n", stdout);
  }
  printf("num_extra: %" PRIu32 "\n", parsed.image.metadata.num_extra);
  {
    const jxltran::ToneMapping& tm = im.tone_mapping;
    if (im.all_default) {
      printf("tone_mapping: default (image metadata all_default)\n");
    } else if (tm.all_default) {
      printf("tone_mapping: default\n");
    } else {
      printf(
          "tone_mapping: intensity_target=%.9g min_nits=%.9g "
          "relative_to_max_display=%s linear_below=%.9g\n",
          static_cast<double>(jxltran::F16BitsToFloat(tm.intensity_target)),
          static_cast<double>(jxltran::F16BitsToFloat(tm.min_nits)),
          tm.relative_to_max_display ? "true" : "false",
          static_cast<double>(jxltran::F16BitsToFloat(tm.linear_below)));
    }
  }
  bool has_alpha_channel = false;
  for (const auto& ec : parsed.image.metadata.ec_info) {
    if (ec.type == jxltran::ExtraChannelType::kAlpha) {
      has_alpha_channel = true;
      break;
    }
  }
  printf("has_alpha_channel: %s\n", has_alpha_channel ? "yes" : "no");

  for (uint32_t i = 0; i < parsed.image.metadata.num_extra; ++i) {
    const jxltran::ExtraChannelInfo& ec = parsed.image.metadata.ec_info[i];
    printf("extra_channel: %" PRIu32 " d_alpha=%s", i,
           ec.d_alpha ? "yes" : "no");
    if (!ec.d_alpha) {
      const char* tslug = "unknown";
      switch (ec.type) {
        case jxltran::ExtraChannelType::kAlpha:
          tslug = "alpha";
          break;
        case jxltran::ExtraChannelType::kDepth:
          tslug = "depth";
          break;
        case jxltran::ExtraChannelType::kSpotColour:
          tslug = "spot";
          break;
        case jxltran::ExtraChannelType::kSelectionMask:
          tslug = "selection";
          break;
        case jxltran::ExtraChannelType::kBlack:
          tslug = "black";
          break;
        case jxltran::ExtraChannelType::kCFA:
          tslug = "cfa";
          break;
        case jxltran::ExtraChannelType::kThermal:
          tslug = "thermal";
          break;
        case jxltran::ExtraChannelType::kNonOptional:
          tslug = "nonoptional";
          break;
        case jxltran::ExtraChannelType::kOptional:
          tslug = "optional";
          break;
      }
      printf(" type=%s dim_shift=%" PRIu32 " bits=%" PRIu32 " float_sample=%s",
             tslug, ec.dim_shift, ec.bit_depth.bits_per_sample,
             ec.bit_depth.float_sample ? "yes" : "no");
      if (ec.bit_depth.float_sample) {
        printf(" exp_bits=%" PRIu32, ec.bit_depth.exp_bits);
      }
      fputs(" name_hex=", stdout);
      for (uint8_t c : ec.name) printf("%02x", c);
      if (ec.type == jxltran::ExtraChannelType::kAlpha) {
        printf(" alpha_associated=%s", ec.alpha_associated ? "yes" : "no");
      }
      if (ec.type == jxltran::ExtraChannelType::kSpotColour) {
        printf(" spot_rgb_solidity=%.9g,%.9g,%.9g,%.9g",
               static_cast<double>(jxltran::F16BitsToFloat(ec.spot_red)),
               static_cast<double>(jxltran::F16BitsToFloat(ec.spot_green)),
               static_cast<double>(jxltran::F16BitsToFloat(ec.spot_blue)),
               static_cast<double>(jxltran::F16BitsToFloat(ec.spot_solidity)));
      }
      if (ec.type == jxltran::ExtraChannelType::kCFA) {
        printf(" cfa_channel=%" PRIu32,
               static_cast<unsigned>(ec.cfa_channel));
      }
    } else {
      printf(" type=alpha(implicit) dim_shift=%" PRIu32, ec.dim_shift);
    }
    fputc('\n', stdout);
  }
  if (cmyk_rgb_main && !MetadataHasExplicitBlackExtra(im)) {
    fputs("implicit_cmyk_k: yes\n", stdout);
  }

  for (size_t i = 0; i < parsed.frames.size(); ++i) {
    const jxltran::FrameHeader& fh = parsed.frames[i].frame;
    int64_t sx0 = 0;
    int64_t sy0 = 0;
    uint64_t rw = canvas_w;
    uint64_t rh = canvas_h;
    if (fh.have_crop) {
      const int32_t x0 = (fh.frame_type != jxltran::kFrameTypeReferenceOnly)
                             ? jxltran::UnpackSigned(fh.ux0)
                             : 0;
      const int32_t y0 = (fh.frame_type != jxltran::kFrameTypeReferenceOnly)
                             ? jxltran::UnpackSigned(fh.uy0)
                             : 0;
      sx0 = x0;
      sy0 = y0;
      rw = fh.crop_width;
      rh = fh.crop_height;
    }
    int64_t dx = 0;
    int64_t dy = 0;
    uint64_t dtw = 0;
    uint64_t dth = 0;
    jxltran::StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx0, sy0, rw,
                                        rh, &dx, &dy, &dtw, &dth);
    printf("frame_region: %" PRIuS " %" PRIu64 "x%" PRIu64, i, dtw, dth);
    if (dx >= 0) {
      printf("+%" PRId64, dx);
    } else {
      printf("%" PRId64, dx);
    }
    if (dy >= 0) {
      printf("+%" PRId64 "\n", dy);
    } else {
      printf("%" PRId64 "\n", dy);
    }
    printf("frame_upsampling: %" PRIuS " %u\n", i,
           std::max<uint32_t>(1u, fh.upsampling));
    const uint32_t ft = fh.frame_type;
    const char* encslug = "-";
    if (ft == jxltran::kFrameTypeRegular ||
        ft == jxltran::kFrameTypeSkipProgressive) {
      encslug = (fh.encoding == jxltran::kFrameEncModular) ? "modular" : "vardct";
    }
    printf("frame_layer: %" PRIuS " type=%s encoding=%s", i, FrameTypeInfoSlug(ft),
           encslug);
    if (ft == jxltran::kFrameTypeRegular ||
        ft == jxltran::kFrameTypeSkipProgressive) {
      printf(" blend=%s duration=%" PRIu32 " alpha_ch=%" PRIu32
             " clamp=%u source=%" PRIu32 " target=%" PRIu32 " is_last=%u",
             BlendModeInfoSlug(fh.blending_info.mode),
             parsed.image.metadata.have_animation ? fh.duration : 0u,
             fh.blending_info.alpha_channel,
             fh.blending_info.clamp ? 1u : 0u, fh.blending_info.source,
             (!fh.is_last) ? fh.save_as_reference : 0u,
             fh.is_last ? 1u : 0u);
    } else {
      fputs(" blend=skip duration=0", stdout);
    }
    if (!fh.name.empty()) {
      fputs(" name_hex=", stdout);
      for (uint8_t c : fh.name) printf("%02x", c);
    }
    fputc('\n', stdout);
    if (ft != jxltran::kFrameTypeLF &&
        ft != jxltran::kFrameTypeReferenceOnly) {
      float ew[6];
      jxltran::GabEffectiveWeights6(
          fh.encoding == jxltran::kFrameEncModular, fh.restoration.all_default,
          fh.restoration, ew);
      printf("gab_effective: %" PRIuS
             " %.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
             i, ew[0], ew[1], ew[2], ew[3], ew[4], ew[5]);
      const uint32_t epf_it = EpfItersEffectiveForInfo(fh);
      printf("epf_iters: %" PRIuS " %u\n", i, epf_it);
      float epf_amp = 0.f;
      if (jxltran::EpfEffectiveAmplitudeForInfo(
              fh.encoding == jxltran::kFrameEncModular,
              fh.restoration.all_default, fh.restoration, &epf_amp)) {
        printf("epf_amp_effective: %" PRIuS " %.9g\n", i, epf_amp);
      }
      const bool vardct = (fh.encoding != jxltran::kFrameEncModular);
      printf("epf_uniform_applies: %" PRIuS " %u\n", i,
             (vardct && epf_it > 0) ? 1u : 0u);
      if (vardct && epf_it > 0) {
        printf("epf_uniform_effective: %" PRIuS " %.9g\n", i,
               jxltran::EpfSharpUniformityForInfo(fh.restoration.all_default,
                                                  fh.restoration));
      }
      const jxltran::FramedUnit& fu = parsed.frames[i];
      if (vardct && fu.lf_global_quantizer_parsed) {
        printf("lf_global_scale: %" PRIuS " %u\n", i,
               static_cast<unsigned>(fu.lf_global_quantizer_global_scale));
      }
    }
  }
}

// Replaces all jxlc/jxlp boxes with a single jxlc carrying |codestream|.
static void ReplaceCodestreamInBoxes(std::vector<jxltran::JxlBox>* boxes,
                                      std::vector<uint8_t> codestream) {
  boxes->erase(
      std::remove_if(
          boxes->begin(), boxes->end(),
          [](const jxltran::JxlBox& b) {
            return memcmp(b.type, "jxlc", 4) == 0 ||
                   memcmp(b.type, "jxlp", 4) == 0;
          }),
      boxes->end());
  size_t insert_at = 0;
  for (size_t i = 0; i < boxes->size(); ++i) {
    if (memcmp((*boxes)[i].type, "JXL ", 4) == 0 ||
        memcmp((*boxes)[i].type, "ftyp", 4) == 0 ||
        memcmp((*boxes)[i].type, "jxll", 4) == 0) {
      insert_at = i + 1;
    }
  }
  jxltran::JxlBox jxlc;
  memcpy(jxlc.type, "jxlc", 4);
  jxlc.data = std::move(codestream);
  boxes->insert(boxes->begin() + insert_at, std::move(jxlc));
  PatchFtypMinorVersion(boxes, 0);
}

int main(int argc, const char* argv[]) {
  Args args;
  jpegxl::tools::CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return 1;
  }
  if (cmdline.HelpFlagPassed() || !args.file_in) {
    cmdline.PrintHelp();
    return 0;
  }
  const bool meta_extract = args.extract_exif != nullptr ||
                            args.extract_xmp != nullptr ||
                            args.extract_jumbf != nullptr;
  const bool meta_set = args.set_exif != nullptr || args.set_xmp != nullptr ||
                         args.set_jumbf != nullptr;
  if (args.GabOptionCount() > 1) {
    fprintf(stderr,
            "jxltran: use at most one of --gab-blur, --gab-sharpen, "
            "--gab-weights\n");
    return 1;
  }
  if (args.frames.set && !args.keep_listed_frames &&
      args.GabOptionCount() == 0 &&
      !args.NeedsPhotonNoiseAny() && !args.NeedsLfDcQuantMul() &&
      !args.NeedsLfGlobalScale() &&
      !args.NeedsSplinesSet() &&
      !args.NeedsClearSplinesFrames() && !args.NeedsLfGlobalChunk0TailTrim() &&
      args.group_order == jxltran::TocGroupOrderCli::kKeep &&
      !args.NeedsEpfIters() && !args.NeedsEpfAmpScale() &&
      !args.NeedsEpfUniformity() && !args.NeedsSetFrameBlends() &&
      !args.NeedsSetFrameDurations() && !args.NeedsSetFrameNames() &&
      !args.NeedsSetFrameRegions()) {
    fprintf(stderr,
            "jxltran: --frames requires at least one of --gab-blur, "
            "--gab-sharpen, --gab-weights, --set-epf-iters, "
            "--set-epf-amplitude-scale, --set-epf-uniformity, "
            "--set-photon-noise-iso, --set-photon-noise-weights, "
            "--lf-dc-quant-mul, --lf-global-scale, "
            "--set-splines-from, --clear-splines-frames, "
            "--lf-global-chunk0-tail-trim-bytes, --set-frame-names, "
            "--set-frame-region, or a\n"
            "non-default --group_order\n");
    return 1;
  }
  if (args.keep_frames.set && !args.keep_listed_frames) {
    fprintf(stderr,
            "jxltran: --keep-frames is only valid together with "
            "--keep-listed-frames\n");
    return 1;
  }
  if (args.keep_listed_frames) {
    if (!args.keep_frames.set && !args.frames.set) {
      fprintf(stderr,
              "jxltran: --keep-listed-frames requires --keep-frames or --frames\n");
      return 1;
    }
    if (!args.file_out) {
      fprintf(stderr,
              "jxltran: --keep-listed-frames requires an OUTPUT file\n");
      return 1;
    }
  }
  if (args.append_jxl && !args.file_out) {
    fprintf(stderr, "jxltran: --append-jxl requires OUTPUT.\n");
    return 1;
  }
  if (args.append_dummy_tail && !args.file_out) {
    fprintf(stderr, "jxltran: --append-dummy-tail requires OUTPUT.\n");
    return 1;
  }
  if (args.append_jxl_skip_compat_check && !args.append_jxl) {
    fprintf(stderr,
            "jxltran: --append-jxl-skip-compat-check requires --append-jxl\n");
    return 1;
  }
  if (args.append_dummy_tail && args.append_jxl) {
    fprintf(stderr,
            "jxltran: do not combine --append-dummy-tail with --append-jxl\n");
    return 1;
  }
  if (args.NeedsSplinesSet() && args.NeedsClearSplinesFrames()) {
    fprintf(stderr,
            "jxltran: do not combine --set-splines-from with "
            "--clear-splines-frames\n");
    return 1;
  }
  if (args.NeedsLfGlobalChunk0TailTrim() && !args.NeedsClearSplinesFrames() &&
      !args.NeedsLfGlobalScale()) {
    fprintf(stderr,
            "jxltran: --lf-global-chunk0-tail-trim-bytes requires "
            "--clear-splines-frames or --lf-global-scale in the same invocation\n");
    return 1;
  }
  if (args.NeedsLfGlobalChunk0TailTrim() && args.NeedsClearSplinesFrames()) {
    for (const auto& pr : args.lf_chunk0_tail_trim.pairs) {
      if (!std::binary_search(args.clear_splines_frames.indices.begin(),
                              args.clear_splines_frames.indices.end(),
                              pr.first)) {
        fprintf(stderr,
                "jxltran: --lf-global-chunk0-tail-trim-bytes: frame %" PRIuS
                " must appear in --clear-splines-frames\n",
                pr.first);
        return 1;
      }
    }
  }
  if ((args.extract_exif && args.set_exif) ||
      (args.extract_xmp && args.set_xmp) ||
      (args.extract_jumbf && args.set_jumbf)) {
    fprintf(stderr,
            "jxltran: cannot combine --extract-* and --set-* for the same "
            "metadata type.\n");
    return 1;
  }
  if (args.set_orientation != 0 && args.rel_orientation != 0) {
    fprintf(stderr,
            "jxltran: use only one of --set-orientation and "
            "--set-orientation-relative\n");
    return 1;
  }
  if (args.info && args.info_structure) {
    fprintf(stderr,
            "jxltran: use only one of --info and --info-structure.\n");
    return 1;
  }
  if ((args.info || args.info_structure || args.extract_icc ||
        args.extract_splines || meta_extract) &&
      args.file_out) {
    fprintf(stderr,
            "jxltran: --info, --info-structure, and --extract-* options do not "
            "take an output file; omit OUTPUT.\n");
    return 1;
  }
  if ((args.info || args.info_structure || args.extract_icc ||
        args.extract_splines || meta_extract) &&
      InfoModeUsesNonDefaultOptions(args)) {
    fprintf(stderr,
            "jxltran: --info, --info-structure, and --extract-* cannot be "
            "combined with other options (only optional -v/--verbose).\n");
    return 1;
  }
  if (meta_set &&
      (args.info || args.info_structure || args.extract_icc || meta_extract)) {
    fprintf(stderr,
            "jxltran: --set-exif/--set-xmp/--set-jumbf cannot be combined with "
            "--info, --info-structure, --extract-icc, or --extract-exif/xmp/"
            "jumbf.\n");
    return 1;
  }
  if (meta_set && !args.file_out) {
    fprintf(stderr,
            "jxltran: --set-exif, --set-xmp, and --set-jumbf require OUTPUT.\n");
    return 1;
  }
  if (args.check_reversible && !args.file_out) {
    fprintf(stderr,
            "jxltran: --check_reversible requires an OUTPUT file argument\n");
    return 1;
  }
  if (args.opsin_inverse && !args.NeedsOpsinSliderArgs()) {
    fprintf(stderr,
            "jxltran: --opsin-inverse requires at least one of --opsin-exposure, "
            "--opsin-temperature, --opsin-tint, --opsin-hue\n");
    return 1;
  }

  if (args.photon_iso.set && args.photon_weights != nullptr) {
    fprintf(stderr,
            "jxltran: use only one of --set-photon-noise-iso and "
            "--set-photon-noise-weights\n");
    return 1;
  }

  if (!args.info && !args.info_structure && !args.file_out &&
      !args.extract_icc && !args.extract_splines && !meta_extract) {
    fprintf(stderr, "No output file specified.\n");
    return 1;
  }
  jxltran::GabArgs gab_args{};
  if (args.gab_blur) {
    if (!ParseGabBlur(args.gab_blur, &gab_args)) return 1;
  } else if (args.gab_sharpen) {
    if (!ParseGabSharpen(args.gab_sharpen, &gab_args)) return 1;
  } else if (args.gab_weights) {
    if (!ParseGabWeights(args.gab_weights, &gab_args)) return 1;
  }

  jxltran::TraceInit(args.verbose ? stderr : nullptr);

  std::vector<uint8_t> input;
  if (!ReadFile(args.file_in, &input)) return 1;

  bool is_container = jxltran::IsJxlContainer(input.data(), input.size());
  const bool input_was_container = is_container;
  bool is_codestream = jxltran::IsJxlCodestream(input.data(), input.size());

  if (!is_container && !is_codestream) {
    fprintf(stderr, "Input file is not a JPEG XL file.\n");
    return 1;
  }

  // Parse boxes if input is a container; otherwise |codestream| holds the raw
  // bytes and |boxes| stays empty.
  std::vector<jxltran::JxlBox> boxes;
  std::vector<uint8_t> codestream;
  if (is_container) {
    if (!jxltran::ParseBoxes(input.data(), input.size(), &boxes)) return 1;
  } else {
    codestream = input;
  }

  if (meta_set && !input_was_container) {
    jxltran::JxlBox jxl_magic;
    memcpy(jxl_magic.type, "JXL ", 4);
    jxl_magic.data.assign({0x0d, 0x0a, 0x87, 0x0a});
    jxltran::JxlBox ftyp;
    memcpy(ftyp.type, "ftyp", 4);
    ftyp.data.assign({'j', 'x', 'l', ' ', 0, 0, 0, 0, 'j', 'x', 'l', ' '});
    jxltran::JxlBox jxlc;
    memcpy(jxlc.type, "jxlc", 4);
    jxlc.data = std::move(codestream);
    boxes.push_back(std::move(jxl_magic));
    boxes.push_back(std::move(ftyp));
    boxes.push_back(std::move(jxlc));
    is_container = true;
  }

  if (args.info || args.info_structure || args.extract_icc ||
      args.extract_splines || meta_extract) {
    if (meta_extract && !input_was_container) {
      fprintf(stderr,
              "jxltran: --extract-exif, --extract-xmp, and --extract-jumbf "
              "require a JXL container input.\n");
      return 1;
    }
    if (args.info || args.info_structure || args.extract_icc ||
        args.extract_splines) {
      std::vector<uint8_t> cs;
      if (input_was_container) {
        if (!jxltran::ReassembleCodestream(boxes, &cs)) return 1;
      } else {
        cs = std::move(codestream);
      }
      jxltran::ParsedCodestream parsed;
      if (!jxltran::ReadCodestream(cs.data(), cs.size(), &parsed)) return 1;
      if (args.extract_icc) {
        if (!parsed.image.metadata.colour_encoding.want_icc) {
          fprintf(stderr,
                  "jxltran: codestream has no embedded ICC profile "
                  "(colour_encoding.want_icc is false).\n");
          return 1;
        }
        if (!IccBytesLookLikeIcc1(parsed.image.icc_bytes)) {
          fprintf(stderr,
                  "jxltran: embedded ICC could not be expanded to a valid ICC.1 "
                  "profile (internal ICC codec mismatch).\n");
          return 1;
        }
        if (!WriteFile(args.extract_icc, parsed.image.icc_bytes)) return 1;
      }
      if (args.extract_splines) {
        std::string text;
        const std::vector<size_t>* fsel =
            args.frames.set ? &args.frames.indices : nullptr;
        if (!jxltran::SplinesTextFromCodestream(parsed, fsel, &text)) {
          fprintf(stderr,
                  "jxltran: no spline text to write (no matching frames with "
                  "splines, or parse error).\n");
          return 1;
        }
        std::vector<uint8_t> bytes(text.begin(), text.end());
        if (!WriteFile(args.extract_splines, bytes)) return 1;
      }
      if (args.info_structure) {
        jxltran::PrintBitstreamStructureJson(
            stdout, input.data(), input.size(), boxes, input_was_container,
            parsed, cs.size());
      }
      if (args.info) {
        PrintJxlInfoStdout(parsed);
        if (args.verbose) {
          jxltran::PrintCodestreamTocLayoutVerbose(stdout, parsed);
        }
      }
    }
    if (meta_extract) {
      std::vector<uint8_t> payload;
      if (args.extract_exif) {
        if (!jxltran::ExtractMetadataPayloadToBuffer(boxes, "Exif", &payload)) {
          return 1;
        }
        if (!WriteFile(args.extract_exif, payload)) return 1;
      }
      if (args.extract_xmp) {
        if (!jxltran::ExtractMetadataPayloadToBuffer(boxes, "xml ", &payload)) {
          return 1;
        }
        if (!WriteFile(args.extract_xmp, payload)) return 1;
      }
      if (args.extract_jumbf) {
        if (!jxltran::ExtractMetadataPayloadToBuffer(boxes, "jumb", &payload)) {
          return 1;
        }
        if (!WriteFile(args.extract_jumbf, payload)) return 1;
      }
    }
    return 0;
  }

  // ── Apply strip ───────────────────────────────────────────────────────────
  if (is_container && args.strip != 0) {
    jxltran::StripBoxesByType(&boxes, args.strip);
  }

  // ── Apply brob transform ──────────────────────────────────────────────────
  if (is_container && args.brob.mode != jxltran::BrobOpt::kAsIs) {
#ifdef JXLTRAN_HAVE_BROTLI
    if (args.brob.mode == jxltran::BrobOpt::kDecompress) {
      if (!jxltran::DecompressBrobBoxes(&boxes)) return 1;
    } else {
      if (!jxltran::CompressMetadataBoxes(&boxes, args.brob.types)) return 1;
    }
#else
    fprintf(stderr,
            "jxltran was compiled without brotli support; "
            "--brob=%s is not available.\n",
            args.brob.mode == jxltran::BrobOpt::kCompress ? "compress"
                                                          : "decompress");
    return 1;
#endif
  }

  // ── Apply jxlp transform ─────────────────────────────────────────────────
  if (is_container && args.jxlp != JxlpOpt::kAsIs) {
    if (args.jxlp == JxlpOpt::kSort) {
      if (!jxltran::SortJxlpBoxes(&boxes)) return 1;
      PatchFtypMinorVersion(&boxes, 0);
    } else {  // kMerge
      if (!jxltran::MergeJxlpBoxes(&boxes)) return 1;
      PatchFtypMinorVersion(&boxes, 0);
    }
  }

  // ── Apply box ordering ────────────────────────────────────────────────────
  if (is_container && args.box_order.order != jxltran::BoxOrder::kAsIs) {
    jxltran::ReorderBoxes(&boxes, args.box_order.order, args.box_order.types);
  }

  if (is_container) {
    if (args.set_exif) {
      std::vector<uint8_t> payload;
      if (!ReadFile(args.set_exif, &payload)) return 1;
      jxltran::ReplaceMetadataBox(&boxes, "Exif", std::move(payload));
    }
    if (args.set_xmp) {
      std::vector<uint8_t> payload;
      if (!ReadFile(args.set_xmp, &payload)) return 1;
      jxltran::ReplaceMetadataBox(&boxes, "xml ", std::move(payload));
    }
    if (args.set_jumbf) {
      std::vector<uint8_t> payload;
      if (!ReadFile(args.set_jumbf, &payload)) return 1;
      jxltran::ReplaceMetadataBox(&boxes, "jumb", std::move(payload));
    }
  }

  if (args.append_jxl || args.append_dummy_tail) {
    std::vector<uint8_t> append_cs;
    if (args.append_dummy_tail) {
      append_cs = jxltran::BuiltinAppendDummyTailCodestream();
    } else {
      std::vector<uint8_t> append_file;
      if (!ReadFile(args.append_jxl, &append_file)) return 1;
      const bool append_is_cont =
          jxltran::IsJxlContainer(append_file.data(), append_file.size());
      const bool append_is_raw =
          jxltran::IsJxlCodestream(append_file.data(), append_file.size());
      if (!append_is_cont && !append_is_raw) {
        fprintf(stderr, "jxltran: --append-jxl: not a JPEG XL file\n");
        return 1;
      }
      std::vector<jxltran::JxlBox> append_boxes;
      if (append_is_cont) {
        if (!jxltran::ParseBoxes(append_file.data(), append_file.size(),
                                 &append_boxes)) {
          return 1;
        }
        if (!jxltran::ReassembleCodestream(append_boxes, &append_cs)) return 1;
      } else {
        append_cs = std::move(append_file);
      }
    }

    std::vector<uint8_t> primary_cs;
    if (is_container) {
      if (!jxltran::ReassembleCodestream(boxes, &primary_cs)) return 1;
    } else {
      primary_cs = codestream;
    }

    const bool skip_header_compat =
        args.append_dummy_tail || args.append_jxl_skip_compat_check;
    std::string append_err;
    std::vector<uint8_t> merged_cs;
    if (!jxltran::AppendCodestreamMerge(primary_cs, append_cs, &merged_cs,
                                        &append_err, skip_header_compat)) {
      const char* append_label =
          args.append_dummy_tail ? "--append-dummy-tail" : "--append-jxl";
      fprintf(stderr, "jxltran: %s: %s\n", append_label, append_err.c_str());
      return 1;
    }
    if (is_container) {
      ReplaceCodestreamInBoxes(&boxes, std::move(merged_cs));
    } else {
      codestream = std::move(merged_cs);
    }
  }

  // ── Codestream transforms (read → mutate ParsedCodestream → write)
  // ───────────
  jxltran::UndoRecorder undo_rec;
  std::vector<uint8_t> codestream_before_transform;
  if (args.file_out) {
    const bool meta_set_cur = args.set_exif != nullptr ||
                               args.set_xmp != nullptr ||
                               args.set_jumbf != nullptr;
    if (!jxltran::ArgsBoxPipelineReversibleForUndo(
            args.strip != 0, args.jxlp != JxlpOpt::kAsIs,
            args.box_order.order != jxltran::BoxOrder::kAsIs,
            args.brob.mode != jxltran::BrobOpt::kAsIs,
            args.append_jxl != nullptr, meta_set_cur)) {
      undo_rec.SetBoxPipelineNotReversible(
          "strip/jxlp/box-order/brob/append-jxl/metadata");
    }
    if (args.NeedsClearSplinesFrames() && !args.NeedsLfGlobalChunk0TailTrim()) {
      undo_rec.SetCodestreamNotReversible(
          "standalone --clear-splines-frames (undo not emitted)");
    }
    if (args.gab_weights != nullptr || args.NeedsSetFrameDurations()) {
      undo_rec.SetCodestreamNotReversible(
          "operation not in undo scope (see WIP.md)");
    }
  }

  const bool needs_codestream_transform =
      args.NeedsHeaderMod() || args.crop.set || args.GabOptionCount() > 0 ||
      args.NeedsPhotonNoiseAny() || args.NeedsLfDcQuantMul() ||
      args.NeedsLfGlobalScale() || args.NeedsSplinesSet() ||
      args.NeedsClearSplinesFrames() || args.NeedsLfGlobalChunk0TailTrim() ||
      args.NeedsEpfIters() || args.NeedsEpfAmpScale() ||
      args.NeedsEpfUniformity() ||
      args.group_order != jxltran::TocGroupOrderCli::kKeep ||
      args.NeedsOpsinAdjust() || args.NeedsKeepListedFrames() ||
      args.NeedsSetFrameBlends() || args.NeedsSetFrameDurations() ||
      args.NeedsSetFrameNames() || args.NeedsSetFrameRegions();
  if (args.file_out && args.check_reversible && !needs_codestream_transform) {
    if (is_container) {
      if (!jxltran::ReassembleCodestream(boxes, &codestream_before_transform)) {
        return 1;
      }
    } else {
      codestream_before_transform = codestream;
    }
  }
  if (needs_codestream_transform) {
    if (args.file_out) {
      if (is_container) {
        if (!jxltran::ReassembleCodestream(boxes, &codestream_before_transform)) {
          return 1;
        }
      } else {
        codestream_before_transform = codestream;
      }
    }
    auto transform_codestream = [&](std::vector<uint8_t>* cs) -> bool {
      const uint8_t* orig_bytes = cs->data();
      const size_t orig_size = cs->size();
      jxltran::ParsedCodestream parsed;
      if (!jxltran::ReadCodestream(orig_bytes, orig_size, &parsed)) {
        return false;
      }
      if (args.file_out) {
        undo_rec.CapturePreHeader(parsed);
        if (args.frames.set) {
          undo_rec.SetSelectiveFramesCopy(args.frames.indices);
        }
      }
      if (args.tps.set &&
          !ResolveTpsPercentFromMetadata(parsed.image.metadata, &args.tps)) {
        return false;
      }
      bool wrote_anything = args.NeedsHeaderMod();
      uint32_t input_orient_for_crop = parsed.image.metadata.orientation;
      if (input_orient_for_crop < 1 || input_orient_for_crop > 8) {
        input_orient_for_crop = 1;
      }
      const uint32_t input_sw_for_crop = parsed.image.size.width;
      const uint32_t input_sh_for_crop = parsed.image.size.height;
      if (args.NeedsHeaderMod()) {
        jxltran::HeaderMod mod;
        if (args.set_orientation != 0) {
          mod.set_orientation = args.set_orientation;
        } else if (args.rel_orientation != 0) {
          uint32_t cur = parsed.image.metadata.orientation;
          if (cur < 1 || cur > 8) cur = 1;
          mod.set_orientation = jxltran::ComposeExifOrientationAfter(
              cur, args.rel_orientation);
        } else {
          mod.set_orientation = 0;
        }
        mod.set_bits_per_sample = args.set_bits_per_sample;
        if (args.main_float.set) {
          mod.set_main_float_sample = true;
          mod.main_float_sample = (args.main_float.value != 0);
        }
        if (args.main_exp.set) {
          mod.set_main_exp_bits = true;
          mod.main_exp_bits = args.main_exp.value;
        }
        mod.have_set_num_loops = args.num_loops.set;
        mod.set_num_loops = args.num_loops.value;
        mod.have_set_tps = args.tps.set;
        mod.set_tps_numerator = args.tps.numerator;
        mod.set_tps_denominator = args.tps.denominator;
        mod.have_set_intensity_target = args.intensity_target.set;
        mod.intensity_target_to_default_bundle =
            args.intensity_target.to_default_bundle;
        mod.intensity_target_nits = args.intensity_target.nits;
        for (const std::string& spec : args.set_extra_channel_specs) {
          jxltran::ExtraChannelHeaderPatch ep;
          if (!ParseOneExtraChannelPatch(spec.c_str(), &ep)) return false;
          mod.extra_channel_patches.push_back(std::move(ep));
        }
        if (args.file_out && !mod.extra_channel_patches.empty()) {
          undo_rec.NoteExtraChannelHeaderBeforeApply(parsed,
                                                     mod.extra_channel_patches);
        }

        bool changing_color_enums = args.set_transfer.set ||
                                    args.set_white_point.set || args.set_primaries.set ||
                                    args.set_rendering_intent.set;

        if (changing_color_enums && parsed.image.metadata.colour_encoding.want_icc && !args.strip_icc) {
          fprintf(stderr, "jxltran: Cannot set color encoding enums because the image uses an ICC profile. Use --strip-icc if you want to replace the ICC profile with standard color encoding.\n");
          return false;
        }

        mod.strip_icc = args.strip_icc;
        mod.set_transfer = args.set_transfer.set;
        mod.transfer_function = args.set_transfer.transfer_function;
        mod.set_white_point = args.set_white_point.set;
        mod.white_point = args.set_white_point.white_point;
        mod.set_primaries = args.set_primaries.set;
        mod.primaries = args.set_primaries.primaries;
        mod.set_rendering_intent = args.set_rendering_intent.set;
        mod.rendering_intent = args.set_rendering_intent.rendering_intent;
        if (!jxltran::ApplyHeaderMod(&parsed, mod)) return false;
      }
      if (args.file_out) {
        const bool hdr_undo_fields =
            args.set_orientation != 0 || args.rel_orientation != 0 ||
            args.set_bits_per_sample != 0 || args.main_float.set ||
            args.main_exp.set || args.num_loops.set ||
            args.tps.set || args.intensity_target.set ||
            args.set_transfer.set || args.strip_icc ||
            args.set_white_point.set || args.set_primaries.set || args.set_rendering_intent.set;
        if (hdr_undo_fields) {
          bool color_enums_changed = args.set_transfer.set ||
                                     args.set_white_point.set || args.set_primaries.set ||
                                     args.set_rendering_intent.set;
          if (color_enums_changed || args.strip_icc) {
            undo_rec.SetCodestreamNotReversible("Color encoding modification (undo not emitted)");
          }
          uint32_t orient_after = parsed.image.metadata.orientation;
          if (orient_after < 1 || orient_after > 8) orient_after = 1;
          undo_rec.NoteHeaderChanges(
              args.set_orientation != 0, args.rel_orientation != 0,
              orient_after,
              args.set_bits_per_sample != 0 || args.main_float.set ||
                  args.main_exp.set,
              args.num_loops.set,
              args.tps.set, args.intensity_target.set);
        }
      }
      if (args.NeedsOpsinAdjust()) {
        jxltran::OpsinAdjustParams oa;
        oa.undo_inverse = args.opsin_inverse;
        if (args.opsin_exposure.set) oa.exposure_ev = args.opsin_exposure.value;
        if (args.opsin_temperature.set) {
          oa.temperature = args.opsin_temperature.value;
        }
        if (args.opsin_tint.set) oa.tint = args.opsin_tint.value;
        if (args.opsin_hue.set) oa.hue = args.opsin_hue.value;
        if (args.opsin_bias_x.set) {
          oa.opsin_bias_xyb[0].set = true;
          oa.opsin_bias_xyb[0].value = args.opsin_bias_x.value;
        }
        if (args.opsin_bias_y.set) {
          oa.opsin_bias_xyb[1].set = true;
          oa.opsin_bias_xyb[1].value = args.opsin_bias_y.value;
        }
        if (args.opsin_bias_b.set) {
          oa.opsin_bias_xyb[2].set = true;
          oa.opsin_bias_xyb[2].value = args.opsin_bias_b.value;
        }
        if (args.opsin_quant_bias_0.set) {
          oa.quant_bias_delta[0].set = true;
          oa.quant_bias_delta[0].value = args.opsin_quant_bias_0.value;
        }
        if (args.opsin_quant_bias_1.set) {
          oa.quant_bias_delta[1].set = true;
          oa.quant_bias_delta[1].value = args.opsin_quant_bias_1.value;
        }
        if (args.opsin_quant_bias_2.set) {
          oa.quant_bias_delta[2].set = true;
          oa.quant_bias_delta[2].value = args.opsin_quant_bias_2.value;
        }
        if (args.opsin_quant_bias_3.set) {
          oa.quant_bias_delta[3].set = true;
          oa.quant_bias_delta[3].value = args.opsin_quant_bias_3.value;
        }
        bool opsin_changed = false;
        if (!jxltran::ApplyOpsinAdjust(&parsed, oa, &opsin_changed)) {
          return false;
        }
        if (opsin_changed) wrote_anything = true;
      }
      if (args.file_out && args.NeedsOpsinSliderArgs() && !args.opsin_inverse) {
        undo_rec.NoteOpsinAdjust(
            args.opsin_exposure.set, args.opsin_temperature.set,
            args.opsin_tint.set, args.opsin_hue.set, args.opsin_exposure.value,
            args.opsin_temperature.value, args.opsin_tint.value,
            args.opsin_hue.value);
      }
      if (args.file_out && !args.opsin_inverse &&
          (args.opsin_bias_x.set || args.opsin_bias_y.set ||
           args.opsin_bias_b.set)) {
        const bool set_bias[3] = {args.opsin_bias_x.set, args.opsin_bias_y.set,
                                  args.opsin_bias_b.set};
        float undo_bias[3];
        jxltran::ComputeOpsinBiasUndoToImplicitDefault(parsed.image.metadata,
                                                       set_bias, undo_bias);
        undo_rec.NoteOpsinBiasAdjust(
            args.opsin_bias_x.set, args.opsin_bias_y.set, args.opsin_bias_b.set,
            undo_bias[0], undo_bias[1], undo_bias[2]);
      }
      if (args.file_out && !args.opsin_inverse &&
          (args.opsin_quant_bias_0.set || args.opsin_quant_bias_1.set ||
           args.opsin_quant_bias_2.set || args.opsin_quant_bias_3.set)) {
        const bool set_q[4] = {
            args.opsin_quant_bias_0.set, args.opsin_quant_bias_1.set,
            args.opsin_quant_bias_2.set, args.opsin_quant_bias_3.set};
        float undo_q[4];
        jxltran::ComputeQuantBiasUndoToImplicitDefault(parsed.image.metadata, set_q,
                                                     undo_q);
        undo_rec.NoteOpsinQuantBiasAdjust(
            args.opsin_quant_bias_0.set, args.opsin_quant_bias_1.set,
            args.opsin_quant_bias_2.set, args.opsin_quant_bias_3.set, undo_q[0],
            undo_q[1], undo_q[2], undo_q[3]);
      }
      if (args.set_frame_regions.set) {
        for (const auto& pr : args.set_frame_regions.pairs) {
          if (args.file_out &&
              !undo_rec.CaptureFrameRegionBeforeMove(parsed, pr.first,
                                                     input_orient_for_crop)) {
            return false;
          }
          const CropArg& cr = pr.second;
          bool fr_changed = false;
          if (!jxltran::ApplySetFrameRegionFromDisplay(
                  &parsed, pr.first, input_orient_for_crop, cr.x, cr.y, cr.w,
                  cr.h, &fr_changed)) {
            return false;
          }
          if (fr_changed) wrote_anything = true;
        }
      }
      if (args.crop.set) {
        int32_t crop_x = args.crop.x;
        int32_t crop_y = args.crop.y;
        uint32_t crop_w = args.crop.w;
        uint32_t crop_h = args.crop.h;
        if (input_orient_for_crop != 1) {
          if (!jxltran::ConvertDisplayCropToStorageCanvas(
                  input_orient_for_crop, input_sw_for_crop, input_sh_for_crop,
                  args.crop.x, args.crop.y, args.crop.w, args.crop.h, &crop_x,
                  &crop_y, &crop_w, &crop_h)) {
            return false;
          }
        }
        const bool crop_noop = crop_w == input_sw_for_crop &&
                               crop_h == input_sh_for_crop && crop_x == 0 &&
                               crop_y == 0;
        if (!crop_noop) {
          if (args.file_out) {
            undo_rec.NoteCropApplied(input_sw_for_crop, input_sh_for_crop, crop_x,
                                     crop_y, crop_w, crop_h);
          }
          if (!jxltran::ApplyCrop(&parsed, crop_x, crop_y, crop_w, crop_h)) {
            return false;
          }
          wrote_anything = true;
        }
      }
      const std::vector<size_t>* frame_sel =
          args.frames.set ? &args.frames.indices : nullptr;
      if (args.GabOptionCount() > 0) {
        if (args.file_out) {
          if (args.gab_blur) {
            undo_rec.NoteGabBlur(gab_args.amount);
          } else if (args.gab_sharpen) {
            undo_rec.NoteGabSharpen(gab_args.amount);
          }
        }
        if (!jxltran::ApplyGabArgs(&parsed, gab_args, frame_sel)) return false;
        wrote_anything = true;
      }
      const bool any_epf = args.NeedsEpfIters() || args.NeedsEpfAmpScale() ||
                           args.NeedsEpfUniformity();
      uint32_t epf_mask = 0;
      if (args.file_out && any_epf) {
        undo_rec.CaptureRestorationBeforeEpfOps(parsed, frame_sel);
      }
      if (args.NeedsEpfIters()) {
        if (!jxltran::ApplyEpfIters(&parsed, args.epf_iters.iters, frame_sel)) {
          return false;
        }
        wrote_anything = true;
        if (args.file_out) {
          epf_mask |= 1u;
        }
      }
      if (args.NeedsEpfAmpScale()) {
        if (!jxltran::ApplyEpfAmplitudeScale(&parsed, args.epf_amp_scale.factor,
                                            frame_sel)) {
          return false;
        }
        if (args.epf_amp_scale.factor != 1.f) {
          wrote_anything = true;
          if (args.file_out) {
            epf_mask |= 2u;
            undo_rec.NoteEpfAmplitudeScale(args.epf_amp_scale.factor);
          }
        }
      }
      if (args.NeedsEpfUniformity()) {
        bool epf_u_changed = false;
        if (!jxltran::ApplyEpfSharpUniformity(&parsed, args.epf_uniform.value,
                                              frame_sel, &epf_u_changed)) {
          return false;
        }
        if (epf_u_changed) {
          wrote_anything = true;
          if (args.file_out) {
            epf_mask |= 4u;
          }
        }
      }
      if (args.file_out && any_epf) {
        undo_rec.FinalizeEpfRestorationUndo(epf_mask);
      }
      if (args.NeedsSplinesSet()) {
        jxltran::SplinesApplySummary spline_summary;
        if (!jxltran::ApplySplinesFromFile(&parsed, args.set_splines_from,
                                           frame_sel, &spline_summary)) {
          return false;
        }
        if (args.file_out) {
          if (spline_summary.replaced_existing_splines) {
            undo_rec.SetCodestreamNotReversible(
                "--set-splines-from replaced existing splines (undo not emitted)");
          } else if (!spline_summary.added_on_spline_free.empty()) {
            undo_rec.NoteSplinesAddedOnFrames(spline_summary.added_on_spline_free);
          }
        }
        wrote_anything = true;
      }
      if (args.NeedsPhotonNoiseAny()) {
        if (args.file_out) {
          undo_rec.CapturePhotonNoiseBeforeApply(parsed, frame_sel);
        }
        bool ok = true;
        if (args.NeedsPhotonNoiseIso()) {
          ok = jxltran::ApplyPhotonNoiseIso(&parsed, true, args.photon_iso.iso,
                                            frame_sel);
        } else {
          ok = jxltran::ApplyPhotonNoiseWeights(&parsed, true, args.photon_weights,
                                                frame_sel);
        }
        if (!ok) return false;
        if (args.file_out) {
          undo_rec.FinalizePhotonNoiseAfterApply(parsed);
        }
        wrote_anything = true;
      }
      if (args.NeedsLfDcQuantMul()) {
        float m[3] = {args.lf_dc_quant_mul.mx, args.lf_dc_quant_mul.my,
                      args.lf_dc_quant_mul.mb};
        std::vector<size_t> lf_dc_splice;
        if (args.file_out) {
          jxltran::ListLfDcQuantMulTargetFrames(parsed, m, frame_sel,
                                                &lf_dc_splice);
        }
        if (!jxltran::ApplyLfDcQuantMul(&parsed, true, m, frame_sel)) {
          return false;
        }
        if (args.file_out) {
          undo_rec.NoteLfDcQuantMulForward(lf_dc_splice, m[0], m[1], m[2]);
        }
        wrote_anything = true;
      }
      if (args.NeedsLfGlobalScale()) {
        if (args.file_out) {
          undo_rec.CaptureLfGlobalScaleBeforeApply(parsed, frame_sel,
                                                   args.lf_global_scale.value);
        }
        if (!jxltran::ApplyLfGlobalScale(&parsed, true, args.lf_global_scale.value,
                                         frame_sel)) {
          return false;
        }
        wrote_anything = true;
      }
      if (args.group_order != jxltran::TocGroupOrderCli::kKeep) {
        if (args.file_out) {
          undo_rec.CaptureTocBeforeGroupOrder(parsed, frame_sel);
        }
        bool toc_changed = false;
        if (!jxltran::ApplyTocGroupOrder(&parsed, args.group_order, args.center_x,
                                        args.center_y, frame_sel,
                                        &toc_changed)) {
          return false;
        }
        if (args.file_out) {
          undo_rec.FinalizeTocGroupOrderAfterApply(toc_changed);
        }
        if (toc_changed) wrote_anything = true;
      }
      const std::vector<size_t>& keep_order_sel =
          args.keep_frames.set ? args.keep_frames.indices_ordered
                               : args.frames.indices_ordered;
      const bool keep_and_blends =
          args.NeedsKeepListedFrames() && args.NeedsSetFrameBlends();
      bool did_early_keep_for_blends = false;
      if (keep_and_blends) {
        const size_t n_before_keep = parsed.frames.size();
        bool keep_changed = false;
        jxltran::KeepReorderUndoSpec keep_undo;
        if (!jxltran::ApplyKeepListedFrames(
                &parsed, keep_order_sel, &keep_changed,
                args.file_out ? &keep_undo : nullptr)) {
          return false;
        }
        if (args.file_out && !keep_undo.forward_order.empty()) {
          undo_rec.NoteKeepReorder(std::move(keep_undo));
        } else if (args.file_out && args.keep_listed_frames && keep_changed &&
                   keep_undo.forward_order.empty()) {
          undo_rec.SetCodestreamNotReversible(
              "--keep-listed-frames: --check_reversible only supports a full "
              "reorder (each codestream frame index kept exactly once)");
        }
        if (keep_changed) wrote_anything = true;
        std::vector<jxltran::FrameBlendOverride> blend_ov =
            args.set_frame_blends.overrides;
        if (!jxltran::RemapFrameBlendOverridesForKeepOrder(
                keep_order_sel, n_before_keep, &blend_ov)) {
          return false;
        }
        if (args.file_out) {
          undo_rec.NoteFrameBlendBeforeOverrides(parsed, blend_ov);
        }
        if (!jxltran::ApplyFrameBlendOverrides(&parsed, blend_ov)) {
          return false;
        }
        wrote_anything = true;
        did_early_keep_for_blends = true;
      }
      if (args.NeedsSetFrameBlends() && !did_early_keep_for_blends) {
        if (args.file_out) {
          undo_rec.NoteFrameBlendBeforeOverrides(
              parsed, args.set_frame_blends.overrides);
        }
        if (!jxltran::ApplyFrameBlendOverrides(&parsed,
                                              args.set_frame_blends.overrides)) {
          return false;
        }
        wrote_anything = true;
      }
      if (args.NeedsClearSplinesFrames()) {
        if (!jxltran::ApplyClearSplinesOnFrames(
                &parsed, args.clear_splines_frames.indices)) {
          return false;
        }
        wrote_anything = true;
      }
      if (args.NeedsLfGlobalChunk0TailTrim()) {
        for (const auto& pr : args.lf_chunk0_tail_trim.pairs) {
          if (!jxltran::ApplyLfGlobalPhysicalChunk0TailTrim(&parsed, pr.first,
                                                            pr.second)) {
            return false;
          }
        }
        wrote_anything = true;
      }
      if (args.NeedsSetFrameDurations()) {
        if (!jxltran::ApplyFrameDurationOverrides(
                &parsed, args.set_frame_durations.pairs)) {
          return false;
        }
        wrote_anything = true;
      }
      if (args.NeedsKeepListedFrames() && !did_early_keep_for_blends) {
        bool keep_changed = false;
        jxltran::KeepReorderUndoSpec keep_undo;
        if (!jxltran::ApplyKeepListedFrames(&parsed, keep_order_sel, &keep_changed,
                                            args.file_out ? &keep_undo : nullptr)) {
          return false;
        }
        if (args.file_out && !keep_undo.forward_order.empty()) {
          undo_rec.NoteKeepReorder(std::move(keep_undo));
        } else if (args.file_out && args.keep_listed_frames && keep_changed &&
                   keep_undo.forward_order.empty()) {
          undo_rec.SetCodestreamNotReversible(
              "--keep-listed-frames: --check_reversible only supports a full "
              "reorder (each codestream frame index kept exactly once)");
        }
        if (keep_changed) wrote_anything = true;
      }
      if (args.NeedsSetFrameNames()) {
        if (args.file_out) {
          for (const auto& pr : args.set_frame_names.pairs) {
            undo_rec.NoteFrameNameBeforeChange(parsed, pr.first);
          }
        }
        if (!jxltran::ApplySetFrameNames(&parsed, args.set_frame_names.pairs)) {
          return false;
        }
        wrote_anything = true;
      }
      if (!wrote_anything) {
        return true;
      }
      std::vector<uint8_t> out;
      if (!jxltran::WriteCodestream(parsed, orig_bytes, &out)) return false;
      if (args.file_out) {
        undo_rec.FinalizeSplineLfGlobalChunk0TailTrim(orig_bytes, orig_size,
                                                      out.data(), out.size());
      }
      *cs = std::move(out);
      return true;
    };

    auto replace_codestream_boxes = [&](std::vector<uint8_t> cs) {
      ReplaceCodestreamInBoxes(&boxes, std::move(cs));
    };

    if (is_container) {
      std::vector<uint8_t> cs;
      if (!jxltran::ReassembleCodestream(boxes, &cs)) return 1;
      if (!transform_codestream(&cs)) return 1;
      replace_codestream_boxes(std::move(cs));
    } else {
      if (!transform_codestream(&codestream)) return 1;
    }
    if (args.file_out) {
      undo_rec.FinalizePipelineCompatibility();
    }
  }

  // ── Determine whether output uses a container ────────────────────────────
  bool output_container;
  switch (args.container) {
    case ContainerOpt::kAsIs:
      output_container = is_container;
      break;
    case ContainerOpt::kNo:
      output_container = false;
      break;
    case ContainerOpt::kYes:
      output_container = true;
      break;
    case ContainerOpt::kIfNeeded:
      output_container = is_container && jxltran::HasMetadataBoxes(boxes);
      break;
  }

  if (meta_set && !output_container) {
    fprintf(stderr,
            "jxltran: --set-exif/--set-xmp/--set-jumbf require container output "
            "(--container cannot be no).\n");
    return 1;
  }

  // ── Produce output ────────────────────────────────────────────────────────
  std::vector<uint8_t> output;
  if (!output_container) {
    // Output a bare codestream.
    if (is_container) {
      if (!jxltran::ReassembleCodestream(boxes, &output)) return 1;
    } else {
      output = std::move(codestream);
    }
  } else {
    // Output a container.
    if (is_container) {
      // Serialize the (possibly transformed) box list.
      for (const auto& box : boxes) {
        jxltran::SerializeBox(box, &output);
      }
    } else {
      // Wrap bare codestream in a minimal container.
      jxltran::WriteJxlSignatureBox(&output);
      jxltran::WriteJxlFtypBox(&output);
      jxltran::JxlBox jxlc;
      memcpy(jxlc.type, "jxlc", 4);
      jxlc.data = std::move(codestream);
      jxltran::SerializeBox(jxlc, &output);
    }
  }

  if (!WriteFile(args.file_out, output)) return 1;

  if (args.file_out) {
    std::string undo_line;
    if (undo_rec.BuildUndoCommandLine(argv[0], args.file_out,
                                      "INPUT_RESTORED.jxl", &undo_line)) {
      PrintReversibleUndoLine(undo_line);
    }
    if (args.check_reversible) {
      if (!undo_rec.box_pipeline_reversible() ||
          !undo_rec.codestream_undo_supported()) {
        printf("reversible_check: not_reversible\n");
      } else {
        std::vector<uint8_t> out_cs;
        const bool out_cont =
            jxltran::IsJxlContainer(output.data(), output.size());
        if (out_cont) {
          std::vector<jxltran::JxlBox> out_boxes;
          if (!jxltran::ParseBoxes(output.data(), output.size(), &out_boxes)) {
            return 1;
          }
          if (!jxltran::ReassembleCodestream(out_boxes, &out_cs)) return 1;
        } else {
          out_cs = output;
        }
        const jxltran::ReversibleCheckStatus st =
            undo_rec.VerifyRoundtripCodestream(codestream_before_transform,
                                               out_cs);
        if (st == jxltran::ReversibleCheckStatus::kOk) {
          printf("reversible_check: ok\n");
        } else if (st ==
                   jxltran::ReversibleCheckStatus::kNotReversibleSkipped) {
          printf("reversible_check: not_reversible\n");
        } else {
          printf("reversible_check: mismatch\n");
        }
      }
    }
  }
  return 0;
}
