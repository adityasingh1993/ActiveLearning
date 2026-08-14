# Implementation Plan — Interactive Lasso Cut, Undo, Redo, and Clear Mask

Implement full interactive **Lasso Cut (freehand polygon selection)** with real-time vector path preview, alongside **Undo (Ctrl+Z)**, **Redo (Ctrl+Y)**, and **Clear Mask (Delete)** for the 3D Slicer HASSL Annotation Studio.

---

## Proposed Changes

### Component: Web Frontend (`hassl/app/static/`)

#### [MODIFY] [index.html](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.html)
- Add ↩️ **Undo**, ↪️ **Redo**, and 🗑️ **Clear Mask** action buttons to the Slicer3D toolbar.

#### [MODIFY] [index.css](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/index.css)
- Add styling for Undo/Redo buttons and dashed yellow Lasso preview path.

#### [MODIFY] [app.js](file:///f:/Projects/Canvas/AcftiveLearningV1/hassl/app/static/app.js)
- **Lasso Tool**:
  - Track `state.lassoPoints = []`.
  - On `mousedown`: Begin path `state.lassoPoints = [pt]`.
  - On `mousemove`: Append points if dragging, render real-time yellow polygon outline on canvas.
  - On `mouseup`: Rasterize closed polygon into binary mask array using canvas 2D fill context, push history, and clear lasso path.
- **Undo / Redo / Delete**:
  - `state.undoStack`: Array of mask history snapshots per viewport.
  - `state.redoStack`: Array of redo snapshots per viewport.
  - `pushHistory(axis)`: Deep copy `state.masks2D[axis]` before any edit.
  - `undo(axis)`: Restore previous mask snapshot.
  - `redo(axis)`: Reapply undone mask snapshot.
  - `clearSlice(axis)`: Zero out all mask voxels on the current slice.
- **Hotkeys**:
  - `Ctrl + Z` / `Z`: Undo
  - `Ctrl + Y` / `Shift + Ctrl + Z` / `Y`: Redo
  - `Delete` / `Backspace`: Clear Slice

---

## Verification Plan

### Automated Tests
- Run `python scripts/run_pre_commit.py` to ensure module syntax and test suite pass 100%.

### Manual Verification
- Launch server `python -m hassl.app.server --port 8000`.
- Open browser at `http://localhost:8000`.
- Select **Lasso Cut** tool, click and drag a loop around a region, verify live yellow polygon outline appears while dragging, and fills upon release.
- Test **Undo** (`Ctrl+Z`) and verify previous state is restored.
- Test **Redo** (`Ctrl+Y`) and verify undone action is re-applied.
- Test **Clear Mask** (`Delete`) and verify current slice is cleared.
