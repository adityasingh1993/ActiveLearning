"""
HASSL Web UI Server (Option C).

A lightweight FastAPI application for reviewing and correcting AI pre-segmentation
masks directly in the browser — no 3D Slicer required.

Usage:
    python -m hassl.app.server --config config.yaml --port 8000
    # Then open http://localhost:8000 in your browser

Endpoints:
    GET  /                          → Main annotation UI
    GET  /api/volumes               → List all volumes with AL scores
    GET  /api/volume/{id}/slice     → Get a 2D slice as PNG
    POST /api/volume/{id}/accept    → Accept corrected mask
    POST /api/retrain               → Trigger model retraining
    GET  /api/metrics               → Get training metrics
"""

import argparse
import io
import os
import json
import glob
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required for the web UI. "
        "Install them with: pip install fastapi uvicorn python-multipart"
    )

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import nrrd
except ImportError:
    nrrd = None


app = FastAPI(title="HASSL Annotation UI", version="0.1.0")

# Global state (set during startup)
_state = {
    "config": None,
    "volumes": {},       # {id: {"image_path": ..., "label_path": ..., "preseg_path": ...}}
    "current_volume": None,
    "cached_images": {},  # {id: np.ndarray}
    "cached_labels": {},  # {id: np.ndarray}
    "cached_presegs": {}, # {id: np.ndarray}
}


def _load_volume(path: str) -> np.ndarray:
    """Load a 3D volume from .mha or .nrrd file."""
    if sitk is not None and path.endswith(".mha"):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        return arr.astype(np.float32)
    elif nrrd is not None and path.endswith(".nrrd"):
        data, _ = nrrd.read(path)
        return data.astype(np.float32)
    else:
        raise ValueError(f"Cannot load {path}: unsupported format or missing library")


def _load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask from .seg.nrrd or .nrrd file."""
    if nrrd is not None:
        data, _ = nrrd.read(path)
        return data.astype(np.uint8)
    elif sitk is not None:
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.uint8)
    raise ValueError(f"Cannot load mask {path}")


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to 0-255 uint8 for PNG encoding."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 0:
        arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)


def _array_to_png(arr: np.ndarray) -> bytes:
    """Convert a 2D numpy array to PNG bytes."""
    from PIL import Image
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _create_overlay(image_slice: np.ndarray, mask_slice: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Create an RGB overlay of image + mask."""
    from PIL import Image
    img_norm = _normalize_for_display(image_slice)
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)  # Grayscale -> RGB

    if mask_slice is not None and mask_slice.any():
        # Overlay mask in semi-transparent green
        overlay = rgb.copy().astype(np.float32)
        mask_bool = mask_slice > 0
        overlay[mask_bool, 0] = overlay[mask_bool, 0] * (1 - alpha)
        overlay[mask_bool, 1] = overlay[mask_bool, 1] * (1 - alpha) + 255 * alpha
        overlay[mask_bool, 2] = overlay[mask_bool, 2] * (1 - alpha)
        rgb = overlay.astype(np.uint8)

    return rgb


def scan_volumes(config):
    """Scan data directory and build volume registry."""
    data_dir = Path(config.data_dir)
    image_suffix = config.image_suffix
    label_suffix = config.label_suffix
    preseg_dir = Path(config.preseg_dir)

    image_files = sorted(glob.glob(str(data_dir / f"**/*{image_suffix}"), recursive=True))

    volumes = {}
    for img_path in image_files:
        vol_id = os.path.basename(img_path).replace(image_suffix, "")

        # Check for label
        label_path = None
        for lbl_candidate in [
            str(data_dir / "labels" / f"{vol_id}{label_suffix}"),
            str(Path(img_path).parent / f"{vol_id}{label_suffix}"),
        ]:
            if os.path.exists(lbl_candidate):
                label_path = lbl_candidate
                break

        # Check for pre-segmentation
        preseg_path = None
        for preseg_candidate in [
            str(preseg_dir / f"{vol_id}.seg.nrrd"),
            str(preseg_dir / f"{vol_id}{label_suffix}"),
        ]:
            if os.path.exists(preseg_candidate):
                preseg_path = preseg_candidate
                break

        volumes[vol_id] = {
            "id": vol_id,
            "image_path": img_path,
            "label_path": label_path,
            "preseg_path": preseg_path,
            "status": "labeled" if label_path else ("preseg" if preseg_path else "unlabeled"),
        }

    return volumes


# ─── API Endpoints ───────────────────────────────────────────────────

@app.get("/api/volumes")
async def list_volumes():
    """List all volumes with their status."""
    volumes = list(_state["volumes"].values())
    # Sort: preseg first, then unlabeled, then labeled
    priority = {"preseg": 0, "unlabeled": 1, "labeled": 2}
    volumes.sort(key=lambda v: priority.get(v["status"], 3))
    return {"volumes": volumes, "total": len(volumes)}


@app.get("/api/volume/{vol_id}/info")
async def get_volume_info(vol_id: str):
    """Get info about a specific volume."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]

    # Load and cache volume to get shape
    if vol_id not in _state["cached_images"]:
        _state["cached_images"][vol_id] = _load_volume(vol["image_path"])

    shape = _state["cached_images"][vol_id].shape
    return {
        "id": vol_id,
        "status": vol["status"],
        "shape": shape,
        "num_slices": {"axial": shape[0], "sagittal": shape[2], "coronal": shape[1]},
    }


@app.get("/api/volume/{vol_id}/slice")
async def get_slice(
    vol_id: str,
    axis: str = Query("axial", regex="^(axial|sagittal|coronal)$"),
    index: int = Query(0),
    overlay: bool = Query(True),
    alpha: float = Query(0.4),
):
    """Get a 2D slice as PNG image, optionally with mask overlay."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]

    # Load and cache image
    if vol_id not in _state["cached_images"]:
        _state["cached_images"][vol_id] = _load_volume(vol["image_path"])
    image = _state["cached_images"][vol_id]

    # Load mask (label or preseg)
    mask = None
    mask_path = vol.get("label_path") or vol.get("preseg_path")
    if mask_path and overlay:
        cache_key = vol_id
        if cache_key not in _state["cached_presegs"]:
            try:
                _state["cached_presegs"][cache_key] = _load_mask(mask_path)
            except Exception:
                _state["cached_presegs"][cache_key] = None
        mask = _state["cached_presegs"].get(cache_key)

    # Extract slice
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    ax = axis_map[axis]
    max_idx = image.shape[ax] - 1
    index = max(0, min(index, max_idx))

    if ax == 0:
        img_slice = image[index, :, :]
        mask_slice = mask[index, :, :] if mask is not None else None
    elif ax == 1:
        img_slice = image[:, index, :]
        mask_slice = mask[:, index, :] if mask is not None else None
    else:
        img_slice = image[:, :, index]
        mask_slice = mask[:, :, index] if mask is not None else None

    # Create overlay
    if overlay and mask_slice is not None:
        rgb = _create_overlay(img_slice, mask_slice, alpha=alpha)
    else:
        rgb_val = _normalize_for_display(img_slice)
        rgb = np.stack([rgb_val, rgb_val, rgb_val], axis=-1)

    png_bytes = _array_to_png(rgb)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/volume/{vol_id}/mask_slice")
async def get_mask_slice(
    vol_id: str,
    axis: str = Query("axial", regex="^(axial|sagittal|coronal)$"),
    index: int = Query(0),
):
    """Get raw 2D uint8 mask slice for interactive canvas editing."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]
    mask_path = vol.get("label_path") or vol.get("preseg_path")

    if vol_id not in _state["cached_images"]:
        _state["cached_images"][vol_id] = _load_volume(vol["image_path"])
    image = _state["cached_images"][vol_id]

    if mask_path and vol_id not in _state["cached_presegs"]:
        try:
            _state["cached_presegs"][vol_id] = _load_mask(mask_path)
        except Exception:
            _state["cached_presegs"][vol_id] = None

    mask = _state["cached_presegs"].get(vol_id)
    if mask is None:
        mask = np.zeros(image.shape, dtype=np.uint8)

    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    ax = axis_map[axis]
    index = max(0, min(index, image.shape[ax] - 1))

    if ax == 0:
        mask_slice = mask[index, :, :]
    elif ax == 1:
        mask_slice = mask[:, index, :]
    else:
        mask_slice = mask[:, :, index]

    return {"axis": axis, "index": index, "shape": list(mask_slice.shape), "mask": mask_slice.tolist()}


@app.post("/api/volume/{vol_id}/slice_edit")
async def edit_mask_slice(vol_id: str, payload: dict):
    """Save interactive 2D slice edits into the 3D volume mask and persist to disk (A-6 fix)."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]
    axis = payload.get("axis", "axial")
    index = int(payload.get("index", 0))
    slice_mask = payload.get("mask_data")

    if slice_mask is None:
        raise HTTPException(status_code=400, detail="Missing mask_data in payload")

    if vol_id not in _state["cached_images"]:
        _state["cached_images"][vol_id] = _load_volume(vol["image_path"])
    image = _state["cached_images"][vol_id]

    if vol_id not in _state["cached_presegs"] or _state["cached_presegs"][vol_id] is None:
        mask_path = vol.get("label_path") or vol.get("preseg_path")
        if mask_path and os.path.exists(mask_path):
            _state["cached_presegs"][vol_id] = _load_mask(mask_path)
        else:
            _state["cached_presegs"][vol_id] = np.zeros(image.shape, dtype=np.uint8)

    mask_3d = _state["cached_presegs"][vol_id]
    edited_2d = np.array(slice_mask, dtype=np.uint8)

    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    ax = axis_map[axis]
    index = max(0, min(index, image.shape[ax] - 1))

    if ax == 0:
        mask_3d[index, :, :] = edited_2d
    elif ax == 1:
        mask_3d[:, index, :] = edited_2d
    else:
        mask_3d[:, :, index] = edited_2d

    # Persist updated 3D volume mask to approved pseudo directory
    config = _state["config"]
    approved_dir = os.path.join(config.data_dir, "pseudo_approved")
    os.makedirs(approved_dir, exist_ok=True)
    out_mask_path = os.path.join(approved_dir, f"{vol_id}{config.label_suffix}")

    try:
        from hassl.data.nrrd_utils import write_mask_with_spatial_geometry
        write_mask_with_spatial_geometry(out_mask_path, mask_3d, reference_image_path=vol["image_path"])
    except Exception:
        if nrrd is not None:
            nrrd.write(out_mask_path, mask_3d)

    vol["label_path"] = out_mask_path
    vol["status"] = "pseudo_approved"

    return {"message": f"Slice {index} on {axis} axis saved successfully for {vol_id}", "path": out_mask_path}


@app.post("/api/volume/{vol_id}/accept")
async def accept_volume(vol_id: str):
    """Accept the pre-segmentation as a label (moves preseg -> labels directory & updates provenance C-1 fix)."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]
    preseg_path = vol.get("preseg_path")

    if not preseg_path or not os.path.exists(preseg_path):
        raise HTTPException(status_code=400, detail="No pre-segmentation available")

    config = _state["config"]
    approved_dir = os.path.join(config.data_dir, "pseudo_approved")
    os.makedirs(approved_dir, exist_ok=True)

    import shutil
    dest_approved = os.path.join(approved_dir, f"{vol_id}{config.label_suffix}")
    # V-4 (A-1) fix: Write strictly to pseudo_approved, NEVER to data/labels/ (gold directory)
    shutil.copy2(preseg_path, dest_approved)

    # Update state
    vol["label_path"] = dest_approved
    vol["status"] = "pseudo_approved"

    # Update manifest provenance (P-1 fix)
    manifest_path = os.path.join(config.log_dir, "pool_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if "provenance" not in manifest:
                manifest["provenance"] = {}
            manifest["provenance"][vol_id] = "pseudo_approved"
            if vol_id not in manifest.get("labeled_ids", []):
                manifest["labeled_ids"].append(vol_id)
            if vol_id in manifest.get("unlabeled_ids", []):
                manifest["unlabeled_ids"].remove(vol_id)
            if vol_id in manifest.get("pseudo_ids", []):
                manifest["pseudo_ids"].remove(vol_id)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
        except Exception as e:
            print(f"[WebUI Server] Warning: could not update manifest provenance: {e}")

    # Clear cache
    _state["cached_presegs"].pop(vol_id, None)

    return {"message": f"Volume {vol_id} accepted and saved to {dest}"}


@app.post("/api/volume/{vol_id}/reject")
async def reject_volume(vol_id: str):
    """Reject the pre-segmentation (flags for full manual review)."""
    if vol_id not in _state["volumes"]:
        raise HTTPException(status_code=404, detail=f"Volume {vol_id} not found")

    vol = _state["volumes"][vol_id]
    vol["status"] = "rejected"
    return {"message": f"Volume {vol_id} rejected. Flagged for manual review."}


@app.get("/api/metrics")
async def get_metrics():
    """Get training metrics if available."""
    config = _state["config"]
    manifest_path = os.path.join(config.log_dir, "pool_manifest.json")

    metrics = {"labeled_count": 0, "unlabeled_count": 0, "rounds": []}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        metrics["labeled_count"] = len(manifest.get("labeled_ids", []))
        metrics["unlabeled_count"] = len(manifest.get("unlabeled_ids", []))
        metrics["rounds"] = manifest.get("rounds", [])

    return metrics


@app.post("/api/retrain")
async def trigger_retrain():
    """Trigger model retraining with updated label pool."""
    return {
        "message": "Retrain triggered. Run in terminal: "
                   "python -m hassl.pipeline --phase al-round --round <N>",
        "status": "pending",
    }


# ─── Static Files & HTML ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main annotation UI."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>HASSL Web UI</h1><p>Static files not found.</p>")


# ─── Server Startup ──────────────────────────────────────────────────

def start_server(config, host: str = "0.0.0.0", port: int = 8000):
    """Start the HASSL web UI server."""
    _state["config"] = config
    _state["volumes"] = scan_volumes(config)

    print(f"\n{'=' * 60}")
    print(f"HASSL Web UI")
    print(f"  Volumes found: {len(_state['volumes'])}")
    labeled = sum(1 for v in _state["volumes"].values() if v["status"] == "labeled")
    preseg = sum(1 for v in _state["volumes"].values() if v["status"] == "preseg")
    unlabeled = sum(1 for v in _state["volumes"].values() if v["status"] == "unlabeled")
    print(f"  Labeled: {labeled} | Pre-segmented: {preseg} | Unlabeled: {unlabeled}")
    print(f"  Open: http://localhost:{port}")
    print(f"{'=' * 60}\n")

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HASSL Web Annotation UI")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from hassl.config import HASSLConfig
    config = HASSLConfig.from_yaml(args.config)
    start_server(config, host=args.host, port=args.port)
