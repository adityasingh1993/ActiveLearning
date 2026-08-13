const state = {
    volumes: [],
    currentVolume: null,
    currentAxis: 'axial',
    currentSlice: 0,
    maxSlices: { axial: 0, coronal: 0, sagittal: 0 },
    filter: 'all',
    currentTool: 'brush',  // 'brush' or 'eraser'
    brushRadius: 8,
    isDrawing: false,
    mask2D: null,          // 2D uint8 matrix [H, W]
    maskShape: [0, 0],
};

// ─── API Helpers ─────────────────────────────────────────

async function api(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`/api${endpoint}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || 'API Error');
    }
    return res.json();
}

// ─── Initialization ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadVolumes();
    await loadMetrics();
    initCanvas();

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (!state.currentVolume) return;
        switch (e.key) {
            case 'ArrowUp':
            case 'ArrowRight':
                e.preventDefault();
                changeSlice(1);
                break;
            case 'ArrowDown':
            case 'ArrowLeft':
                e.preventDefault();
                changeSlice(-1);
                break;
            case 'a':
                acceptVolume();
                break;
            case 'r':
                rejectVolume();
                break;
            case 'n':
                loadNextVolume();
                break;
            case 'b':
                setDrawTool('brush');
                break;
            case 'e':
                setDrawTool('eraser');
                break;
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    saveSliceEdit();
                }
                break;
            case '1':
                switchAxis('axial', document.querySelector('[data-axis="axial"]'));
                break;
            case '2':
                switchAxis('coronal', document.querySelector('[data-axis="coronal"]'));
                break;
            case '3':
                switchAxis('sagittal', document.querySelector('[data-axis="sagittal"]'));
                break;
        }
    });

    // Mouse wheel scroll on canvas & image
    const canvas = document.getElementById('draw-canvas');
    if (canvas) {
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            changeSlice(e.deltaY > 0 ? 1 : -1);
        }, { passive: false });
    }
});

// ─── Canvas & Interactive Drawing ────────────────────────

function initCanvas() {
    const canvas = document.getElementById('draw-canvas');
    if (!canvas) return;

    const getCanvasCoords = (e) => {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: Math.floor((clientX - rect.left) * scaleX),
            y: Math.floor((clientY - rect.top) * scaleY)
        };
    };

    const startDraw = (e) => {
        if (!state.currentVolume || !state.mask2D) return;
        state.isDrawing = true;
        drawStroke(getCanvasCoords(e));
    };

    const drawMove = (e) => {
        if (!state.isDrawing || !state.mask2D) return;
        drawStroke(getCanvasCoords(e));
    };

    const stopDraw = () => {
        state.isDrawing = false;
    };

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', drawMove);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);

    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); }, { passive: false });
    canvas.addEventListener('touchmove', (e) => { e.preventDefault(); drawMove(e); }, { passive: false });
    canvas.addEventListener('touchend', stopDraw);
}

function setDrawTool(tool) {
    state.currentTool = tool;
    document.getElementById('tool-brush').classList.toggle('active', tool === 'brush');
    document.getElementById('tool-eraser').classList.toggle('active', tool === 'eraser');
}

function updateRadiusVal(val) {
    state.brushRadius = parseInt(val);
    document.getElementById('radius-val').textContent = val;
}

function drawStroke(pt) {
    if (!state.mask2D || !state.maskShape[0]) return;
    const [h, w] = state.maskShape;
    const val = state.currentTool === 'brush' ? 1 : 0;
    const r = state.brushRadius;

    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx * dx + dy * dy <= r * r) {
                const py = pt.y + dy;
                const px = pt.x + dx;
                if (py >= 0 && py < h && px >= 0 && px < w) {
                    state.mask2D[py][px] = val;
                }
            }
        }
    }
    renderMaskToCanvas();
}

function renderMaskToCanvas() {
    const canvas = document.getElementById('draw-canvas');
    if (!canvas || !state.mask2D) return;
    const ctx = canvas.getContext('2d');
    const [h, w] = state.maskShape;

    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);
    const overlayChecked = document.getElementById('overlay-toggle').checked;
    if (!overlayChecked) return;

    const alpha = document.getElementById('alpha-slider').value / 100;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (state.mask2D[y][x] > 0) {
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

/**
 * Decode a [[value, count], ...] RLE sequence into a 2D array of shape [h, w].
 * Row-major order matches the server's numpy ravel() C-order (V6-9 fix).
 */
function rle_decode(rle, shape) {
    const [h, w] = shape;
    const flat = new Uint8Array(h * w);
    let pos = 0;
    for (const [val, count] of rle) {
        flat.fill(val, pos, pos + count);
        pos += count;
    }
    // Reshape into 2D array [h][w]
    const out = [];
    for (let r = 0; r < h; r++) {
        out.push(Array.from(flat.subarray(r * w, (r + 1) * w)));
    }
    return out;
}

async function fetchMaskSlice() {
    if (!state.currentVolume) return;
    try {
        const data = await api(`/volume/${state.currentVolume}/mask_slice?axis=${state.currentAxis}&index=${state.currentSlice}`);
        if (data.encoding === 'rle' && data.mask_rle) {
            state.mask2D = rle_decode(data.mask_rle, data.shape);
        } else {
            state.mask2D = data.mask;  // legacy fallback
        }
        state.maskShape = data.shape;
        renderMaskToCanvas();
    } catch (err) {
        state.mask2D = null;
    }
}

// ─── Volume List ─────────────────────────────────────────

async function loadVolumes() {
    try {
        const data = await api('/volumes');
        state.volumes = data.volumes || [];
        renderVolumeList();
    } catch (err) {
        showToast('Failed to load volumes: ' + err.message, 'error');
    }
}

function renderVolumeList() {
    const list = document.getElementById('volume-list');
    const filtered = state.filter === 'all'
        ? state.volumes
        : state.volumes.filter(v => v.status === state.filter);

    list.innerHTML = filtered.map((vol, i) => `
        <div class="volume-item ${state.currentVolume === vol.id ? 'active' : ''}"
             style="animation-delay: ${i * 30}ms"
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

// ─── Volume Selection ────────────────────────────────────

async function selectVolume(volId) {
    state.currentVolume = volId;
    state.currentSlice = 0;

    // Show viewer, hide placeholder
    document.getElementById('viewer-placeholder').style.display = 'none';
    document.getElementById('viewer-container').style.display = 'flex';

    try {
        const info = await api(`/volume/${volId}/info`);
        state.maxSlices = info.num_slices;

        // Reset to middle slice
        const maxForAxis = state.maxSlices[state.currentAxis] || 128;
        state.currentSlice = Math.floor(maxForAxis / 2);

        const slider = document.getElementById('slice-slider');
        slider.max = maxForAxis - 1;
        slider.value = state.currentSlice;

        updateSlice();
        renderVolumeList();
    } catch (err) {
        showToast('Failed to load volume: ' + err.message, 'error');
    }
}

// ─── Slice Viewing ───────────────────────────────────────

function updateSlice() {
    if (!state.currentVolume) return;

    const overlay = false;  // Canvas handles interactive mask drawing directly
    const alpha = document.getElementById('alpha-slider').value / 100;

    const url = `/api/volume/${state.currentVolume}/slice`
        + `?axis=${state.currentAxis}`
        + `&index=${state.currentSlice}`
        + `&overlay=${overlay}`
        + `&alpha=${alpha}`;

    const img = document.getElementById('slice-image');
    img.src = url;

    const maxForAxis = state.maxSlices[state.currentAxis] || 0;
    document.getElementById('slice-info').textContent =
        `${state.currentAxis} — Slice ${state.currentSlice + 1} / ${maxForAxis}`;

    fetchMaskSlice();
}

function onSliderChange(value) {
    state.currentSlice = parseInt(value);
    updateSlice();
}

function changeSlice(delta) {
    const maxForAxis = state.maxSlices[state.currentAxis] || 128;
    state.currentSlice = Math.max(0, Math.min(state.currentSlice + delta, maxForAxis - 1));

    document.getElementById('slice-slider').value = state.currentSlice;
    updateSlice();
}

function switchAxis(axis, btn) {
    state.currentAxis = axis;

    document.querySelectorAll('.axis-tab').forEach(t => t.classList.remove('active'));
    if (btn) btn.classList.add('active');

    const maxForAxis = state.maxSlices[axis] || 128;
    state.currentSlice = Math.floor(maxForAxis / 2);

    const slider = document.getElementById('slice-slider');
    slider.max = maxForAxis - 1;
    slider.value = state.currentSlice;

    updateSlice();
}

// ─── Actions ─────────────────────────────────────────────

async function saveSliceEdit() {
    if (!state.currentVolume || !state.mask2D) return;
    try {
        const res = await api(`/volume/${state.currentVolume}/slice_edit`, 'POST', {
            axis: state.currentAxis,
            index: state.currentSlice,
            mask_data: state.mask2D,
        });
        showToast(`💾 Saved slice edit for ${state.currentVolume}`, 'success');
        await loadVolumes();
        await loadMetrics();
    } catch (err) {
        showToast('Failed to save slice edit: ' + err.message, 'error');
    }
}

// ─── Actions ─────────────────────────────────────────────

async function acceptVolume() {
    if (!state.currentVolume) return;
    try {
        const res = await api(`/volume/${state.currentVolume}/accept`, 'POST');
        showToast(`✅ ${state.currentVolume} accepted!`, 'success');
        await loadVolumes();
        await loadMetrics();
        loadNextVolume();
    } catch (err) {
        showToast('Failed: ' + err.message, 'error');
    }
}

async function rejectVolume() {
    if (!state.currentVolume) return;
    try {
        await api(`/volume/${state.currentVolume}/reject`, 'POST');
        showToast(`❌ ${state.currentVolume} rejected`, 'error');
        await loadVolumes();
        loadNextVolume();
    } catch (err) {
        showToast('Failed: ' + err.message, 'error');
    }
}

function loadNextVolume() {
    const presegVolumes = state.volumes.filter(v => v.status === 'preseg');
    if (presegVolumes.length > 0) {
        const next = presegVolumes.find(v => v.id !== state.currentVolume) || presegVolumes[0];
        selectVolume(next.id);
    } else {
        showToast('No more pre-segmented volumes to review!', 'success');
    }
}

async function triggerRetrain() {
    try {
        const res = await api('/retrain', 'POST');
        showToast('🔄 ' + res.message, 'success');
    } catch (err) {
        showToast('Failed: ' + err.message, 'error');
    }
}

// ─── Metrics ─────────────────────────────────────────────

async function loadMetrics() {
    try {
        const metrics = await api('/metrics');
        document.getElementById('count-labeled').textContent = metrics.labeled_count || 0;

        const presegCount = state.volumes.filter(v => v.status === 'preseg').length;
        const unlabeledCount = state.volumes.filter(v => v.status === 'unlabeled').length;
        document.getElementById('count-preseg').textContent = presegCount;
        document.getElementById('count-unlabeled').textContent = unlabeledCount;
    } catch (err) {
        // Silently fail metrics
    }
}

// ─── Toast Notifications ─────────────────────────────────

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;

    // Force reflow for animation
    toast.offsetHeight;

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}
