# Implementation Plan - HASSL Interactive UI Editing & CI Test Suite

Implement slice-level interactive mask editing in the Web UI (Option 2) and an automated synthetic CI test suite with a pre-commit hook (Option 1).

## User Review Required

> [!IMPORTANT]
> - **Option 2 (Interactive Mask Editing)**: Adds an HTML5 canvas overlay in the Web UI allowing users to draw (Brush) and erase (Eraser) segmentation masks directly on 2D slices. Edits are sent to `POST /api/volume/{vol_id}/slice_edit` and updated in the 3D mask file.
> - **Option 1 (CI & Pre-commit)**: Creates `tests/test_pipeline_ci.py` to run `--phase all` on synthetic data and verify pipeline artifact assertions (splits, spatial metadata, provenance, AL diversity), plus a pre-commit check for `test_imports.py`.

## Proposed Changes

### Phase 1: Interactive Slice Mask Editing (Option 2)

#### [MODIFY] [server.py](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/server.py)
- Add `GET /api/volume/{vol_id}/mask_slice`: Returns 2D mask slice array as PNG/raw binary for the requested axis and index.
- Add `POST /api/volume/{vol_id}/slice_edit`: Receives updated 2D slice binary image/mask data, updates the cached 3D mask array at slice index, and saves updated `.seg.nrrd` file.

#### [MODIFY] [index.html](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.html)
- Add an HTML5 `<canvas id="mask-canvas">` overlaid on top of `#slice-image`.
- Add drawing controls toolbar:
  - Tool toggle: **Brush** (Draw green foreground) vs **Eraser** (Clear foreground).
  - Brush radius control slider (1px to 30px).
  - **Save Edits** button.

#### [MODIFY] [app.js](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/app.js)
- Implement interactive mouse drawing (down, move, up) on `<canvas id="mask-canvas">`.
- Render mask slice on canvas with current tool color & opacity.
- Send edited canvas slice buffer to `POST /api/volume/{vol_id}/slice_edit`.

#### [MODIFY] [index.css](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.css)
- Style the canvas overlay and toolbar for smooth interactive drawing.

---

### Phase 2: Automated Synthetic CI Test Suite & Pre-Commit Hook (Option 1)

#### [NEW] [test_pipeline_ci.py](file:///f:/Projects/Canvas/AcftiveLearningV1/tests/test_pipeline_ci.py)
- Create pytest suite running end-to-end synthetic pipeline (`--phase all`).
- Add artifact assertions:
  1. `splits.json` contains $>1$ patient group and non-empty val set.
  2. Mask output spatial metadata (spacing, direction) matches native reference image.
  3. Provenance manifest strictly isolates human vs pseudo-approved labels.

#### [NEW] [run_pre_commit.py](file:///f:/Projects/Canvas/AcftiveLearningV1/scripts/run_pre_commit.py)
- Add standalone script to execute `pytest tests/test_imports.py`.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_imports.py` and `pytest tests/test_pipeline_ci.py`.

### Manual Verification
- Test interactive brush and eraser tools in Web UI by drawing on 2D slice and clicking Save Edits.
- Verify edited mask is updated in `.seg.nrrd` file on disk.
