/**
 * 3D Slicer HASSL Multi-Planar Annotation Studio (app.js)
 *
 * Implements 4-quadrant synchronized MPR viewports (Axial, Coronal, Sagittal)
 * with 3D crosshair navigation, flood fill, window/level contrast control,
 * distance measurement ruler in mm, brush, eraser, and active learning save workflow.
 */

const state = {
    volumes: [],
    currentVolume: null,
    status: 'unloaded',
    cursor: { z: 0, y: 0, x: 0 },          // 3D Voxel Coordinate (Axial=Z, Coronal=Y, Sagittal=X)
    maxSlices: { axial: 128, coronal: 128, sagittal: 128 },
    spacing: [1.0, 1.0, 1.0],               // [sp_z, sp_y, sp_x] in mm
    filter: 'all',
    currentTool: 'crosshair',               // 'crosshair', 'brush', 'eraser', 'fill', 'winlevel', 'ruler'
    brushRadius: 8,
    opacity: 0.4,
    showCrosshairs: true,
    winLevel: { window: 255, level: 128 },  // WW / WL contrast
    masks2D: { axial: null, coronal: null, sagittal: null },
    maskShapes: { axial: [0, 0], coronal: [0, 0], sagittal: [0, 0] },
    rulerPoints: [],                        // [{x, y}, {x, y}] for distance tool
    isDrawing: false,
    activeViewport: 'axial',
};

// ─── API Client ──────────────────────────────────────────────────────

async function api(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`/api${endpoint}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'API Error' }));
        throw new Error(err.detail || 'API Error');
    }
    return res.json();
}

// ─── Initialization ──────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadVolumes();
    await loadMetrics();
    initViewportCanvas('axial');
    initViewportCanvas('coronal');
    initViewportCanvas('sagittal');

    // Global Hotkeys
    document.addEventListener('keydown', (e) => {
        if (!state.currentVolume) return;
        switch (e.key) {
            case '1': setDrawTool('crosshair'); break;
            case '2': setDrawTool('brush'); break;
            case '3': setDrawTool('eraser'); break;
            case '4': setDrawTool('fill'); break;
            case '5': setDrawTool('winlevel'); break;
            case '6': setDrawTool('ruler'); break;
            case 'a': acceptVolume(); break;
            case 'r': rejectVolume(); break;
            case 'n': loadNextVolume(); break;
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    saveSliceEdit();
                }
                break;
        }
    });
});

// ─── Slicer Tool Selection ──────────────────────────────────────────

function setDrawTool(tool) {
    state.currentTool = tool;
    state.rulerPoints = [];
    document.querySelectorAll('.btn-tool').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`tool-${tool}`);
    if (activeBtn) activeBtn.classList.add('active');

    const names = {
        crosshair: '🎯 3D Crosshair Navigation',
        brush: '🖌️ Paint Brush',
        eraser: '🧹 Erase Brush',
        fill: '🪣 Flood Fill',
        winlevel: '🌗 Window / Level Contrast',
        ruler: '📏 Physical Distance Ruler (mm)'
    };
    document.getElementById('hud-tool-name').textContent = names[tool] || tool;
    updateAllViewports();
}

function updateRadiusVal(val) {
    state.brushRadius = parseInt(val);
    document.getElementById('radius-val').textContent = val;
}

// ─── Viewport Interaction & Canvas Handling ─────────────────────────

function initViewportCanvas(axis) {
    const canvas = document.getElementById(`canvas-${axis}`);
    if (!canvas) return;

    let dragStartWW = null;

    const getCoords = (e) => {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: Math.floor((clientX - rect.left) * scaleX),
            y: Math.floor((clientY - rect.top) * scaleY),
            clientX, clientY
        };
    };

    const handlePointerDown = (e) => {
        if (!state.currentVolume) return;
        state.activeViewport = axis;
        const pt = getCoords(e);

        if (state.currentTool === 'crosshair') {
            update3DCursorFromViewport(axis, pt.x, pt.y);
        } else if (state.currentTool === 'fill') {
            floodFill(axis, pt.x, pt.y);
        } else if (state.currentTool === 'ruler') {
            addRulerPoint(axis, pt);
        } else if (state.currentTool === 'winlevel') {
            dragStartWW = { x: pt.clientX, y: pt.clientY, ww: state.winLevel.window, wl: state.winLevel.level };
        } else if (state.currentTool === 'brush' || state.currentTool === 'eraser') {
            state.isDrawing = true;
            applyBrushStroke(axis, pt);
        }
    };

    const handlePointerMove = (e) => {
        if (!state.currentVolume) return;
        const pt = getCoords(e);

        if (state.currentTool === 'crosshair' && e.buttons === 1) {
            update3DCursorFromViewport(axis, pt.x, pt.y);
        } else if (state.currentTool === 'winlevel' && dragStartWW && e.buttons === 1) {
            const dx = pt.clientX - dragStartWW.x;
            const dy = pt.clientY - dragStartWW.y;
            state.winLevel.window = Math.max(1, Math.min(500, dragStartWW.ww + dx));
            state.winLevel.level = Math.max(0, Math.min(255, dragStartWW.wl - dy));
            updateAllViewports();
        } else if (state.isDrawing && (state.currentTool === 'brush' || state.currentTool === 'eraser')) {
            applyBrushStroke(axis, pt);
        }
    };

    const handlePointerUp = () => {
        state.isDrawing = false;
        dragStartWW = null;
    };

    canvas.addEventListener('mousedown', handlePointerDown);
    canvas.addEventListener('mousemove', handlePointerMove);
    canvas.addEventListener('mouseup', handlePointerUp);
    canvas.addEventListener('mouseleave', handlePointerUp);

    // Mousewheel slice stepping
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        stepSlice(axis, e.deltaY > 0 ? 1 : -1);
    }, { passive: false });
}

// ─── 3D Crosshair Navigation & MPR Sync ──────────────────────────────

function update3DCursorFromViewport(axis, ptX, ptY) {
    if (axis === 'axial') {
        // Axial slice is (Z). ptX maps to Sagittal (X), ptY maps to Coronal (Y)
        state.cursor.x = Math.max(0, Math.min(ptX, state.maxSlices.sagittal - 1));
        state.cursor.y = Math.max(0, Math.min(ptY, state.maxSlices.coronal - 1));
    } else if (axis === 'coronal') {
        // Coronal slice is (Y). ptX maps to Sagittal (X), ptY maps to Axial (Z)
        state.cursor.x = Math.max(0, Math.min(ptX, state.maxSlices.sagittal - 1));
        state.cursor.z = Math.max(0, Math.min(ptY, state.maxSlices.axial - 1));
    } else if (axis === 'sagittal') {
        // Sagittal slice is (X). ptX maps to Coronal (Y), ptY maps to Axial (Z)
        state.cursor.y = Math.max(0, Math.min(ptX, state.maxSlices.coronal - 1));
        state.cursor.z = Math.max(0, Math.min(ptY, state.maxSlices.axial - 1));
    }
    updateAllViewports();
}

function stepSlice(axis, delta) {
    const keyMap = { axial: 'z', coronal: 'y', sagittal: 'x' };
    const key = keyMap[axis];
    const maxVal = state.maxSlices[axis] || 128;
    state.cursor[key] = Math.max(0, Math.min(state.cursor[key] + delta, maxVal - 1));
    updateAllViewports();
}

function onViewportSliderChange(axis, value) {
    const keyMap = { axial: 'z', coronal: 'y', sagittal: 'x' };
    state.cursor[keyMap[axis]] = parseInt(value);
    updateAllViewports();
}

// ─── Brush & Flood Fill ──────────────────────────────────────────────

function applyBrushStroke(axis, pt) {
    const mask = state.masks2D[axis];
    const shape = state.maskShapes[axis];
    if (!mask || !shape[0]) return;

    const [h, w] = shape;
    const val = state.currentTool === 'brush' ? 1 : 0;
    const r = state.brushRadius;

    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx * dx + dy * dy <= r * r) {
                const py = pt.y + dy;
                const px = pt.x + dx;
                if (py >= 0 && py < h && px >= 0 && px < w) {
                    mask[py][px] = val;
                }
            }
        }
    }
    renderViewportCanvas(axis);
}

function floodFill(axis, startX, startY) {
    const mask = state.masks2D[axis];
    const shape = state.maskShapes[axis];
    if (!mask || !shape[0]) return;

    const [h, w] = shape;
    if (startX < 0 || startX >= w || startY < 0 || startY >= h) return;

    const targetVal = mask[startY][startX];
    const fillVal = targetVal === 1 ? 0 : 1;
    if (targetVal === fillVal) return;

    const stack = [[startX, startY]];
    while (stack.length > 0) {
        const [x, y] = stack.pop();
        if (x < 0 || x >= w || y < 0 || y >= h) continue;
        if (mask[y][x] === targetVal) {
            mask[y][x] = fillVal;
            stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
        }
    }
    renderViewportCanvas(axis);
}

// ─── Distance Ruler Tool ──────────────────────────────────────────────

function addRulerPoint(axis, pt) {
    state.rulerPoints.push(pt);
    if (state.rulerPoints.length > 2) state.rulerPoints = [pt];

    if (state.rulerPoints.length === 2) {
        const p1 = state.rulerPoints[0];
        const p2 = state.rulerPoints[1];
        const spY = state.spacing[1] || 1.0;
        const spX = state.spacing[2] || 1.0;
        const distMm = Math.sqrt(Math.pow((p2.x - p1.x) * spX, 2) + Math.pow((p2.y - p1.y) * spY, 2));
        document.getElementById('hud-measurement').textContent = `${distMm.toFixed(2)} mm`;
    } else {
        document.getElementById('hud-measurement').textContent = 'Point 1 set...';
    }
    renderViewportCanvas(axis);
}

// ─── Rendering Viewports & Crosshairs ────────────────────────────────

function updateAllViewports() {
    if (!state.currentVolume) return;

    // Update Sliders & HUD Text
    document.getElementById('slider-axial').value = state.cursor.z;
    document.getElementById('slider-coronal').value = state.cursor.y;
    document.getElementById('slider-sagittal').value = state.cursor.x;

    document.getElementById('info-axial').textContent = `Slice Z: ${state.cursor.z + 1} / ${state.maxSlices.axial}`;
    document.getElementById('info-coronal').textContent = `Slice Y: ${state.cursor.y + 1} / ${state.maxSlices.coronal}`;
    document.getElementById('info-sagittal').textContent = `Slice X: ${state.cursor.x + 1} / ${state.maxSlices.sagittal}`;

    document.getElementById('hud-axial').textContent = `Z: ${state.cursor.z}`;
    document.getElementById('hud-coronal').textContent = `Y: ${state.cursor.y}`;
    document.getElementById('hud-sagittal').textContent = `X: ${state.cursor.x}`;

    document.getElementById('coord-z').textContent = state.cursor.z;
    document.getElementById('coord-y').textContent = state.cursor.y;
    document.getElementById('coord-x').textContent = state.cursor.x;

    state.showCrosshairs = document.getElementById('crosshair-toggle').checked;
    state.opacity = parseInt(document.getElementById('alpha-slider').value) / 100;
    document.getElementById('opacity-val').textContent = document.getElementById('alpha-slider').value;

    fetchAndRenderSlice('axial', state.cursor.z);
    fetchAndRenderSlice('coronal', state.cursor.y);
    fetchAndRenderSlice('sagittal', state.cursor.x);
}

async function fetchAndRenderSlice(axis, index) {
    if (!state.currentVolume) return;
    const imgEl = document.getElementById(`img-${axis}`);
    const url = `/api/volume/${state.currentVolume}/slice?axis=${axis}&index=${index}&overlay=false`;
    imgEl.src = url;

    // Fetch mask slice
    try {
        const data = await api(`/volume/${state.currentVolume}/mask_slice?axis=${axis}&index=${index}`);
        if (data.encoding === 'rle' && data.mask_rle) {
            state.masks2D[axis] = rle_decode(data.mask_rle, data.shape);
        } else {
            state.masks2D[axis] = data.mask;
        }
        state.maskShapes[axis] = data.shape;
        renderViewportCanvas(axis);
    } catch (err) {
        state.masks2D[axis] = null;
        renderViewportCanvas(axis);
    }
}

function rle_decode(rle, shape) {
    const [h, w] = shape;
    const flat = new Uint8Array(h * w);
    let pos = 0;
    for (const [val, count] of rle) {
        flat.fill(val, pos, pos + count);
        pos += count;
    }
    const out = [];
    for (let r = 0; r < h; r++) {
        out.push(Array.from(flat.subarray(r * w, (r + 1) * w)));
    }
    return out;
}

function renderViewportCanvas(axis) {
    const canvas = document.getElementById(`canvas-${axis}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const mask = state.masks2D[axis];
    const shape = state.maskShapes[axis];
    if (!shape || !shape[0]) return;

    const [h, w] = shape;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    // Draw Mask
    if (mask) {
        const alpha = state.opacity;
        const imgData = ctx.createImageData(w, h);
        const data = imgData.data;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (mask[y][x] > 0) {
                    const idx = (y * w + x) * 4;
                    data[idx] = 63;       // Red
                    data[idx + 1] = 185;  // Green (MONAI green overlay)
                    data[idx + 2] = 80;   // Blue
                    data[idx + 3] = Math.floor(alpha * 255);
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    // Draw 3D Crosshairs
    if (state.showCrosshairs) {
        ctx.lineWidth = 1;
        let crossX = 0, crossY = 0;
        if (axis === 'axial') { crossX = state.cursor.x; crossY = state.cursor.y; }
        else if (axis === 'coronal') { crossX = state.cursor.x; crossY = state.cursor.z; }
        else if (axis === 'sagittal') { crossX = state.cursor.y; crossY = state.cursor.z; }

        ctx.strokeStyle = axis === 'axial' ? '#ff3b30' : (axis === 'coronal' ? '#34c759' : '#ffcc00');
        ctx.beginPath();
        ctx.moveTo(crossX, 0); ctx.lineTo(crossX, h);
        ctx.moveTo(0, crossY); ctx.lineTo(w, crossY);
        ctx.stroke();
    }

    // Draw Distance Ruler
    if (state.currentTool === 'ruler' && state.rulerPoints.length > 0 && state.activeViewport === axis) {
        ctx.strokeStyle = '#007aff';
        ctx.fillStyle = '#007aff';
        ctx.lineWidth = 2;

        for (const pt of state.rulerPoints) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }

        if (state.rulerPoints.length === 2) {
            const [p1, p2] = state.rulerPoints;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    }
}

// ─── Volume Selection & Queue Management ────────────────────────────

async function loadVolumes() {
    try {
        const data = await api('/volumes');
        state.volumes = data.volumes || [];
        renderVolumeList();
    } catch (err) {
        showToast('Failed to load volume queue: ' + err.message, 'error');
    }
}

function renderVolumeList() {
    const list = document.getElementById('volume-list');
    const filtered = state.filter === 'all'
        ? state.volumes
        : state.volumes.filter(v => v.status === state.filter);

    list.innerHTML = filtered.map((vol) => `
        <div class="volume-item ${state.currentVolume === vol.id ? 'active' : ''}"
             onclick="selectVolume('${vol.id}')">
            <div class="status-dot ${vol.status}"></div>
            <span class="vol-name">${vol.id}</span>
            <span class="vol-status">${vol.status}</span>
        </div>
    `).join('');
}

function filterVolumes(filter, btn) {
    state.filter = filter;
    document.querySelectorAll('.filter-tabs .tab').forEach(t => t.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderVolumeList();
}

async function selectVolume(volId) {
    state.currentVolume = volId;
    document.getElementById('viewer-placeholder').style.display = 'none';
    document.getElementById('mpr-grid').style.display = 'grid';

    try {
        const info = await api(`/volume/${volId}/info`);
        state.maxSlices = info.num_slices;
        state.spacing = info.spacing || [1.0, 1.0, 1.0];
        state.status = info.status;

        state.cursor = {
            z: Math.floor(info.num_slices.axial / 2),
            y: Math.floor(info.num_slices.coronal / 2),
            x: Math.floor(info.num_slices.sagittal / 2)
        };

        document.getElementById('slider-axial').max = info.num_slices.axial - 1;
        document.getElementById('slider-coronal').max = info.num_slices.coronal - 1;
        document.getElementById('slider-sagittal').max = info.num_slices.sagittal - 1;

        document.getElementById('vol-id-display').textContent = volId;
        document.getElementById('meta-dims').textContent = `${info.shape[0]} x ${info.shape[1]} x ${info.shape[2]}`;
        document.getElementById('meta-spacing').textContent = `${state.spacing[0]} x ${state.spacing[1]} x ${state.spacing[2]} mm`;
        document.getElementById('meta-status').textContent = info.status;

        updateAllViewports();
        renderVolumeList();
    } catch (err) {
        showToast('Failed to load volume: ' + err.message, 'error');
    }
}

// ─── Actions: Save, Accept, Reject, Retrain ──────────────────────────

async function saveSliceEdit() {
    if (!state.currentVolume) return;
    const axis = state.activeViewport;
    const mask = state.masks2D[axis];
    if (!mask) return;

    const sliceIdx = axis === 'axial' ? state.cursor.z : (axis === 'coronal' ? state.cursor.y : state.cursor.x);

    try {
        await api(`/volume/${state.currentVolume}/mask_slice`, 'POST', {
            axis,
            index: sliceIdx,
            mask: mask
        });
        showToast(`Saved slice edit on ${axis.toUpperCase()} slice ${sliceIdx + 1}`);
        updateAllViewports();
    } catch (err) {
        showToast('Failed to save slice edit: ' + err.message, 'error');
    }
}

async function acceptVolume() {
    if (!state.currentVolume) return;
    try {
        await api(`/volume/${state.currentVolume}/accept`, 'POST');
        showToast(`Accepted volume ${state.currentVolume}`);
        await loadVolumes();
        await loadMetrics();
        loadNextVolume();
    } catch (err) {
        showToast('Failed to accept volume: ' + err.message, 'error');
    }
}

async function rejectVolume() {
    if (!state.currentVolume) return;
    try {
        await api(`/volume/${state.currentVolume}/reject`, 'POST');
        showToast(`Rejected volume ${state.currentVolume}`);
        await loadVolumes();
        await loadMetrics();
        loadNextVolume();
    } catch (err) {
        showToast('Failed to reject volume: ' + err.message, 'error');
    }
}

function loadNextVolume() {
    const list = state.volumes;
    if (!list.length) return;
    const curIdx = list.findIndex(v => v.id === state.currentVolume);
    const nextVol = list[(curIdx + 1) % list.length];
    if (nextVol) selectVolume(nextVol.id);
}

async function triggerRetrain() {
    try {
        const res = await api('/retrain', 'POST');
        showToast(res.message || 'Retraining triggered');
    } catch (err) {
        showToast('Failed to trigger retrain: ' + err.message, 'error');
    }
}

async function loadMetrics() {
    try {
        const data = await api('/metrics');
        document.getElementById('count-labeled').textContent = data.labeled_count || 0;
        document.getElementById('count-preseg').textContent = data.unlabeled_count || 0;
        document.getElementById('count-unlabeled').textContent = state.volumes.filter(v => v.status === 'unlabeled').length;
    } catch (err) {
        // Silent metric fail fallback
    }
}

function showToast(msg, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast ${type}`;
    setTimeout(() => toast.classList.add('hidden'), 3000);
}
