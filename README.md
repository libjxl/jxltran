# jxltran

**jxltran** is a small command-line tool in the spirit of **jpegtran**, but for **JPEG XL** (ISO/IEC 18181). It rewrites JXL files **without** a full decode-and-encode round trip: it operates on the container and codestream layout so pixel data and compression remain bit-identical where the tool does not intentionally change them.

Typical uses:

- Remux the **ISOBMFF container** (strip or add a container, adjust `jxlp` box layout).
- **Reorder or merge** `jxlp` partial-codestream boxes (including out-of-order delivery cases).
- Edit **orientation** and **crop** metadata in the image header when that can be done as a lossless metadata change.
- Adjust **restoration filter (Gaborish)** parameters where the bitstream path supports it.

It is **not** a general-purpose encoder or decoder; for creating or decoding pixels to/from JXL, use tools such as **cjxl** and **djxl** from the JPEG XL reference implementation.

## Requirements

- **C++17** compiler  
- **CMake** 3.16 or newer  
- Optional: **Brotli** (encode + decode libraries and headers) for `brob`-wrapped boxes. Without Brotli, the tool still builds; brob handling is disabled at compile time.

## Build (standalone)

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The `jxltran` binary is written to `build/` (Release or your chosen configuration).

To disable the optional Brotli probe:

```bash
cmake -S . -B build -DJXLTRAN_ENABLE_BROTLI=OFF
```

## Usage

Synopsis:

```text
jxltran INPUT [OUTPUT] [OPTION]...
jxltran INPUT --info
```

With **`--info`**, omit **`OUTPUT`**: the tool prints to stdout: `dimensions: W x H` (orientation-correct, like `djxl`), `animation: yes|no`, and one **`frame_region: N WxH+X+Y`** line per frame (same display space and `WxH+X+Y` convention as **`--crop`**), then exits.

Use **`jxltran -h`** or **`jxltran --help`** for the full generated help (including verbose per-option text). The sections below group options by theme.

### Positional arguments

| Argument | Role |
|----------|------|
| `INPUT` | Source JPEG XL file (container or bare codestream). |
| `OUTPUT` | Destination file (omit with **`--info`**, **`--extract-*`**, and other read-only modes). |

### Inspection (no `OUTPUT`)

| Option | What it does |
|--------|----------------|
| `--info` | Print dimensions, animation flag, and one `frame_region` line per frame (oriented display space); exit. No other transform/remux flags except optional **`-v`**. |

### Container and metadata (ISOBMFF)

These flags concern the **file wrapper**: whether output is bare codestream or a BMFF container, how `jxlc` / `jxlp` and other boxes are ordered or merged, stripping metadata, Brotli `brob` handling, and replacing or extracting sidecar payloads (ICC lives in the codestream; Exif/XMP/JUMBF are boxes).

| Option | Argument | What it does |
|--------|----------|----------------|
| `--container` | `keep` \| `no` \| `yes` \| `if-needed` | Container vs bare codestream (default `keep`). `no` drops the wrapper and **all** metadata boxes; multiple `jxlp` parts are concatenated in counter order (same payload as after **`--jxlp merge`**; **`--jxlp`** is ignored for bare output). |
| `--jxlp` | `keep` \| `sort` \| `merge` | When output is a container: leave `jxlp` layout as-is, sort by counter, or merge into one `jxlc` (default `keep`). |
| `--strip` | `exif` \| `xmp` \| `jumbf` \| `jbrd` \| `all` \| comma list | Remove listed metadata boxes from the container. |
| `--box-order` | `keep` \| `before-codestream` \| `after-codestream` \| `TYPE[,…]` | Control ordering of boxes in the output container. |
| `--brob` | `keep` \| `compress[:TYPE[,…]]` \| `decompress` | `brob`-wrapped metadata: leave, compress eligible types, or decompress. Requires a Brotli-enabled build; otherwise only `keep` is effective. |
| `--set-exif` | `FILE` | Replace or add the Exif box from `FILE`; removes existing Exif / brob-wrapped Exif. Bare input is wrapped in a minimal container. |
| `--set-xmp` | `FILE` | Same idea for the XMP (`xml `) box. |
| `--set-jumbf` | `FILE` | Same idea for the JUMBF (`jumb`) box. |
| `--extract-icc` | `FILE` | Write embedded ICC from the codestream to `FILE` and exit. |
| `--extract-exif` | `FILE` | Write first Exif box payload (raw TIFF Exif block) to `FILE` and exit; brob-wrapped Exif is expanded when Brotli is available. |
| `--extract-xmp` | `FILE` | Write first XMP box payload to `FILE` and exit. |
| `--extract-jumbf` | `FILE` | Write first JUMBF payload to `FILE` and exit. |
| `--extract-splines` | `FILE` | Write LF-global spline data as text to `FILE` and exit (format matches **`--set-splines-from`**; optional **`--frames`**). |

### Image header, animation timing, and canvas

Fields in the **image header** (and related metadata-only canvas changes): orientation, animation loop / ticks-per-second, XYB opsin adjustments, bits-per-sample hint, and **`--crop`** (display-space canvas size and origin).

| Option | Argument | What it does |
|--------|----------|----------------|
| `--set-orientation` | `1`–`8`, `90`, `180`, `270` | Set Exif-style orientation (absolute). `90` / `180` / `270` are clockwise degrees (same as tags `6` / `3` / `8`). |
| `--set-orientation-relative` | `1`–`8`, `90`, `180`, `270` | Compose this transform *after* the file’s current orientation. Mutually exclusive with **`--set-orientation`**. |
| `--set-bits-per-sample` | `1`–`32` | Set image-header `bits_per_sample`. For XYB-encoded images this is mainly a decoder output-precision hint; for non-XYB images it is part of the coded representation. |
| `--set-main-float` | `0` or `1` | Set image-header `float_sample` for the main channel (`0` = integer, `1` = float). |
| `--set-main-exp` | `2`–`8` | Exponent bits for float main samples (meaningful with `--set-main-float=1` or when the file is already float). |
| `--set-extra-channel` | `INDEX:key=val,...` | Edit **ExtraChannelInfo** for extra channel `INDEX` (`0`…`num_extra−1`). **`dim_shift`** and the number of extra channels are fixed. Keys: `d_alpha` (`0`/`1`), `type` (slug: `alpha`, `spot`, `depth`, `black`, `cfa`, `thermal`, `optional`, `unknown`), `bits`, `float`, `exp`, `assoc` / `alpha_associated`, `name_hex`, `cfa`, `spot_r` / `spot_g` / `spot_b` / `spot_solidity` / `spot_a`. Repeat the option for multiple channels. **`--info`** prints one `extra_channel:` line per channel. |
| `--set-white-point` | `WP` | Force a specific white point in ColourEncoding. Valid values: `d65`, `e`, `dci`. Requires `--strip-icc` if the image has an embedded ICC profile. |
| `--set-primaries` | `PR` | Force specific primaries in ColourEncoding. Valid values: `srgb`, `2100`, `p3`. Requires `--strip-icc` if the image has an embedded ICC profile. |
| `--set-transfer` | `TF` | Force a specific transfer curve in ColourEncoding (discards custom gamma). Valid values: `709`, `unknown`, `linear`, `srgb`, `pq`, `dci`, `hlg`. Requires `--strip-icc` if the image has an embedded ICC profile. |
| `--set-rendering-intent` | `RI` | Force a specific rendering intent in ColourEncoding. Valid values: `perceptual`, `relative`, `saturation`, `absolute`. Requires `--strip-icc` if the image has an embedded ICC profile. |
| `--strip-icc` | | Remove embedded ICC profile if present, forcing the decoder to use the explicit color encoding enums. |
| `--set-num-loops` | `N` | Animation loop count (`0` = infinite); animated images only. |
| `--set-tps` | `N`, `N/D`, or `P%` | Ticks per second (`P%` scales the file’s current TPS); animated images only. |
| `--opsin-exposure` | `EV` | XYB only: exposure in stops (linear RGB after XYB→RGB). |
| `--opsin-temperature` | `T` | XYB only: warmth (−100..+100). |
| `--opsin-tint` | `T` | XYB only: green vs magenta (−100..+100). |
| `--opsin-hue` | `T` | XYB only: Rec. 709 luma–axis hue (HSL-like); −100..100 → −90°..+90°. |
| `--crop` | `WxH` or `WxH+X+Y` | Metadata-only canvas resize/crop; **coordinates are display space** (after orientation, like `djxl`). Extending past the current canvas adds implicit padding. |

### Codestream layout: frames and TOC order

Choose **which codestream frames** are kept and in **which order**, optionally **append** another JXL’s frames, and adjust **TOC section order** (permutation of entropy-coded groups in regular / skip-progressive frames). **`--frames`** scopes many later edits to a subset of frames.

| Option | Argument | What it does |
|--------|----------|----------------|
| `--frames` | `N[,N…]` | Limit subsequent frame-level edits to listed codestream indices (`0`-based; see **`--info`**). With **`--keep-listed-frames`** and no **`--keep-frames`**, legacy: same list selects frames to keep. |
| `--keep-frames` | `N[,N…]` | With **`--keep-listed-frames`**: explicit keep list and output order (deduplicated). |
| `--keep-listed-frames` | — | Rewrite codestream to only the frames in **`--keep-frames`** (or **`--frames`** if **`--keep-frames`** omitted); verbatim compressed data; needs `OUTPUT`. |
| `--append-jxl` | `FILE` | After `INPUT`, append frames from another codestream; needs `OUTPUT` and header compatibility. |
| `--append-jxl-skip-compat-check` | — | With **`--append-jxl`**: skip compatibility check (unsafe). |
| `--append-dummy-tail` | — | Append a fixed minimal terminal frame after `INPUT` (alternative to **`--append-jxl`**); needs `OUTPUT`. |
| `--group_order` | `keep` \| `0` \| `1` \| `progressive` | TOC: leave as-is; strip permutation (`0`); **center-first AC order (`1`, matches cjxl)**; or strip only when stream order is non-progressive. |
| `--center_x`, `--center_y` | `X`, `Y` | With `--group_order=1`: spiral center in pixels (`-1` = image center). Ignored otherwise. |

### Per-frame signal: blending, timing, layout, restoration

Edits to **per-frame** codestream fields: blend parameters, animation tick duration, frame name bytes, display **`frame_region`**, **Gaborish** restoration weights, **EPF** loop-filter strength, **photon noise** LUT, and **LF-global splines** (replace from a text file produced by **`--extract-splines`**).

| Option | Argument | What it does |
|--------|----------|----------------|
| `--set-frame-blends` | `SPEC[,SPEC…]` | Per-frame blend overrides (`INDEX:MODE` and optional fields). |
| `--set-frame-durations` | `INDEX:TICKS[,…]` | Set animation tick duration per frame. |
| `--set-frame-names` | `INDEX:VALUE[,…]` | Set or clear per-frame name bytes: `VALUE` is even-length hex (as in **`--info`** `name_hex=`), an unquoted literal when it is not even-length all-hex (no spaces), or a double-quoted literal for spaces and similar (`\\`, `\"` inside quotes); `INDEX:` alone clears. |
| `--set-frame-region` | `INDEX:WxH+X+Y[,…]` | Move a frame’s display rectangle (same oriented space as **`--info`** / **`--crop`**); size unchanged. |
| `--gab-blur` | `A` | Reversible Gaborish blur (`A` ≥ 0); mutually exclusive with other **`--gab-*`**. |
| `--gab-sharpen` | `A` | Reversible Gaborish sharpen; mutually exclusive with other **`--gab-*`**. |
| `--gab-weights` | `x1,x2,y1,y2,b1,b2` | Six explicit Gaborish weights; mutually exclusive with other **`--gab-*`**. |
| `--set-epf-iters` | `N` | Edge-preserving filter iteration count (`0`–`3`). |
| `--set-epf-amplitude-scale` | `F` | Scale overall EPF strength. |
| `--set-epf-uniformity` | `U` | VarDCT: mix default EPF sharpness ramp toward uniform (`0`..`1`). |
| `--set-photon-noise-iso` | `ISO` | Rewrite synthetic photon-noise LUT from an ISO-like model (`0` clears). |
| `--set-splines-from` | `FILE` | Replace or insert LF-global splines from text (`FILE`); see restrictions in **`--help`**. |

### Verbosity

| Option | What it does |
|--------|----------------|
| `-v`, `--verbose` | Trace codestream header parse/rewrite to stderr (byte + bit positions). More `-v` flags mainly affect **`--help`** verbosity. |


## License

Same as the JPEG XL reference implementation (BSD-style; see the repository `LICENSE` file).
