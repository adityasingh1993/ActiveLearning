/* ═══════════════════════════════════════════════════════════
   HASSL Web Annotation UI — Application Logic
   ═══════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────

const state = {
    volumes: [],
    currentVolume: null,
    currentAxis: 'axial',
    currentSlice: 0,
    maxSlices: { axial: 0, coronal: 0, sagittal: 0 },
    filter: 'all',
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

    // Mouse wheel scroll on image
    const img = document.getElementById('slice-image');
    if (img) {
        img.addEventListener('wheel', (e) => {
            e.preventDefault();
            changeSlice(e.deltaY > 0 ? 1 : -1);
        });
    }
});

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

    const overlay = document.getElementById('overlay-toggle').checked;
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
