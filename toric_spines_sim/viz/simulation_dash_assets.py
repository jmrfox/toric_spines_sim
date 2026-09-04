"""String assets for the synchronized simulation Dash dashboard."""

from __future__ import annotations


DASHBOARD_LAYOUT_CSS = """
.dashboard-stack {
    display: flex;
    flex-direction: column;
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
    padding-bottom: 24px;
}
.dashboard-header {
    margin-bottom: 0;
}
.dashboard-panel {
    overflow: hidden;
    position: relative;
    z-index: 0;
}
.dashboard-panel-3d {
    padding: 12px 12px 4px 12px;
    margin-bottom: 4px;
}
.dashboard-panel-3d .dashboard-graph-3d {
    height: 480px;
    min-height: 480px;
    max-height: 480px;
}
.dashboard-panel-3d .js-plotly-plot,
.dashboard-panel-3d .plot-container,
.dashboard-panel-3d .main-svg {
    height: 480px !important;
}
.dashboard-plots-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px;
    align-items: start;
}
.dashboard-panel-lower {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.dashboard-panel-lower .themed-dropdown {
    margin-bottom: 4px;
}
.dashboard-graph-voltage {
    height: 280px;
    min-height: 280px;
    max-height: 280px;
}
.dashboard-graph-raster {
    height: 392px;
    min-height: 392px;
    max-height: 392px;
}
.dashboard-panel-voltage .js-plotly-plot,
.dashboard-panel-voltage .plot-container {
    height: 280px !important;
}
.dashboard-panel-raster .js-plotly-plot,
.dashboard-panel-raster .plot-container {
    height: 392px !important;
}
.dashboard-timeline {
    padding: 12px 16px 20px 16px;
}
.dashboard-timeline .rc-slider {
    margin-top: 24px;
    margin-bottom: 4px;
    padding-left: 8px;
    padding-right: 8px;
}
.dashboard-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
"""


CLIENTSIDE_PAUSE_OR_RESET = """
function(_pause, _reset) {
    const NU = dash_clientside.no_update;
    const ctx = dash_clientside.callback_context;
    if (!ctx.triggered || ctx.triggered.length === 0) {
        return [NU, NU];
    }
    const id = ctx.triggered[0].prop_id.split('.')[0];
    if (id === 'pause-btn' || id === 'reset-btn') {
        return [false, true];
    }
    return [NU, NU];
}
"""


CLIENTSIDE_PLAYBACK_TRANSPORT = """
function(_play, _pause, _reset, dragValue, _nIntervals,
         playing, frameIdx, bundle) {
    const NU = dash_clientside.no_update;
    const ctx = dash_clientside.callback_context;
    if (!ctx.triggered || ctx.triggered.length === 0) {
        return [NU, NU, NU];
    }
    const trig = ctx.triggered[0].prop_id.split('.')[0];
    const maxIdx = bundle ? bundle.n_frames - 1 : 0;
    frameIdx = parseInt(frameIdx || 0, 10);
    playing = Boolean(playing);

    if (trig === 'play-btn') {
        if (frameIdx >= maxIdx) { frameIdx = 0; }
        return [frameIdx, true, false];
    }
    if (trig === 'pause-btn') {
        return [frameIdx, false, true];
    }
    if (trig === 'reset-btn') {
        return [0, false, true];
    }
    if (trig === 'time-slider') {
        if (dragValue === undefined || dragValue === null) {
            return [NU, NU, NU];
        }
        const target = parseInt(dragValue, 10);
        if (target === frameIdx) { return [NU, NU, NU]; }
        return [target, false, true];
    }
    if (trig === 'tick') {
        if (!playing) { return [NU, NU, NU]; }
        if (frameIdx >= maxIdx) {
            return [maxIdx, false, true];
        }
        return [frameIdx + 1, NU, NU];
    }
    return [NU, NU, NU];
}
"""


CLIENTSIDE_FRAME_UPDATE = """
function(frameIdx, bundle, traceMap) {
    function plotlyDiv(id) {
        const el = document.getElementById(id);
        if (!el) { return null; }
        if (el.classList &&
                el.classList.contains('js-plotly-plot')) {
            return el;
        }
        return el.querySelector('.js-plotly-plot') || el;
    }
    function liveSceneCamera(gd) {
        const scene = gd._fullLayout && gd._fullLayout.scene;
        const obj = scene && scene._scene;
        if (obj && typeof obj.getCamera === 'function') {
            const cam = obj.getCamera();
            if (cam && cam.eye) {
                const proj = (scene.camera && scene.camera.projection
                    && scene.camera.projection.type) || 'perspective';
                return {
                    eye: {x: +cam.eye.x, y: +cam.eye.y, z: +cam.eye.z},
                    center: {
                        x: +cam.center.x, y: +cam.center.y, z: +cam.center.z
                    },
                    up: {x: +cam.up.x, y: +cam.up.y, z: +cam.up.z},
                    projection: {type: proj}
                };
            }
        }
        return null;
    }
    if (!bundle || frameIdx === undefined || frameIdx === null) {
        return dash_clientside.no_update;
    }
    const idx = Math.max(
        0, Math.min(frameIdx, bundle.n_frames - 1)
    );
    const t = bundle.time_ms[idx];

    function pinLiveCamera(gd) {
        const cam = liveSceneCamera(gd);
        if (!cam) { return; }
        if (!gd.layout) { gd.layout = {}; }
        if (!gd.layout.scene) { gd.layout.scene = {}; }
        gd.layout.scene.camera = cam;
        if (gd._fullLayout && gd._fullLayout.scene) {
            gd._fullLayout.scene.camera = cam;
        }
    }

    const gd3d = plotlyDiv('graph-3d');
    if (gd3d) {
        pinLiveCamera(gd3d);
        Plotly.restyle(gd3d, {
            intensity: [bundle.mesh_intensity[idx]]
        }, bundle.mesh_trace_idx);
        if (bundle.synapse_trace_idx !== null &&
                bundle.synapse_trace_idx !== undefined) {
            let facecolor;
            if (bundle.synapse_facecolor !== undefined) {
                facecolor = bundle.synapse_facecolor[idx];
            } else if (
                bundle.synapse_facecolor_indices !== undefined
            ) {
                const palette = bundle.synapse_color_palette;
                const indices = (
                    bundle.synapse_facecolor_indices[idx]
                );
                facecolor = indices.map(
                    function(i) { return palette[i]; }
                );
            }
            if (facecolor !== undefined) {
                Plotly.restyle(gd3d, {
                    facecolor: [facecolor]
                }, bundle.synapse_trace_idx);
            }
        }
    }

    const gdV = plotlyDiv('graph-voltage');
    if (gdV && bundle.voltage_labels &&
            bundle.voltage_marker_values && traceMap) {
        const values = bundle.voltage_marker_values[idx];
        const xUpdate = [];
        const yUpdate = [];
        const traceIndices = [];
        for (let i = 0; i < bundle.voltage_labels.length; i++) {
            const label = bundle.voltage_labels[i];
            const indices = traceMap[label];
            if (!indices) continue;
            const markerIdx = indices[1];
            xUpdate.push([t]);
            yUpdate.push([values[i]]);
            traceIndices.push(markerIdx);
        }
        if (traceIndices.length) {
            Plotly.restyle(gdV, {
                x: xUpdate,
                y: yUpdate
            }, traceIndices);
        }
        Plotly.relayout(gdV, {
            'shapes[0].x0': t,
            'shapes[0].x1': t,
            'datarevision': idx
        });
    }

    const gdR = plotlyDiv('graph-raster');
    if (gdR) {
        Plotly.relayout(gdR, {
            'shapes[0].x0': t,
            'shapes[0].x1': t,
            'datarevision': idx
        });
    }

    const readout = (
        't = ' + t.toFixed(2) + ' ms  (frame ' + (idx + 1) +
        '/' + bundle.n_frames + ')'
    );
    return [readout, idx];
}
"""


CLIENTSIDE_PROBE_VISIBILITY = """
function(selected, traceMap) {
    if (!selected || !traceMap) {
        return dash_clientside.no_update;
    }
    const gd = document.getElementById('graph-voltage');
    if (!gd) {
        return dash_clientside.no_update;
    }
    const selectedSet = new Set(selected);
    const visibility = [];
    const traceIndices = [];
    for (const label in traceMap) {
        const indices = traceMap[label];
        const visible = selectedSet.has(label);
        visibility.push(visible);
        traceIndices.push(indices[0]);
        visibility.push(visible);
        traceIndices.push(indices[1]);
    }
    Plotly.restyle(gd, {visible: visibility}, traceIndices);
    return dash_clientside.no_update;
}
"""


CLIENTSIDE_INSTALL_CAMERA_GUARD = """
function(figuresReady) {
    const NU = dash_clientside.no_update;
    if (!figuresReady) { return NU; }

    function plotlyDiv() {
        const el = document.getElementById('graph-3d');
        if (!el) { return null; }
        if (el.classList && el.classList.contains('js-plotly-plot')) {
            return el;
        }
        return el.querySelector('.js-plotly-plot');
    }
    function liveSceneCamera(gd) {
        const scene = gd._fullLayout && gd._fullLayout.scene;
        const obj = scene && scene._scene;
        if (!(obj && typeof obj.getCamera === 'function')) { return null; }
        const cam = obj.getCamera();
        if (!(cam && cam.eye)) { return null; }
        const proj = (scene.camera && scene.camera.projection
            && scene.camera.projection.type) || 'perspective';
        return {
            eye: {x: +cam.eye.x, y: +cam.eye.y, z: +cam.eye.z},
            center: {
                x: +cam.center.x, y: +cam.center.y, z: +cam.center.z
            },
            up: {x: +cam.up.x, y: +cam.up.y, z: +cam.up.z},
            projection: {type: proj}
        };
    }
    function pinLiveCamera(gd) {
        const cam = liveSceneCamera(gd);
        if (!cam) { return; }
        if (!gd.layout) { gd.layout = {}; }
        if (!gd.layout.scene) { gd.layout.scene = {}; }
        gd.layout.scene.camera = cam;
        if (gd._fullLayout && gd._fullLayout.scene) {
            gd._fullLayout.scene.camera = cam;
        }
    }
    function attach(triesLeft) {
        const gd = plotlyDiv();
        if (!gd) {
            if (triesLeft > 0) {
                window.setTimeout(function() { attach(triesLeft - 1); }, 50);
            }
            return;
        }
        if (gd._tsCameraGuard) { return; }
        gd._tsCameraGuard = true;
        gd.addEventListener('pointerdown', function(ev) {
            const target = ev.target;
            if (target && target.closest && target.closest('.modebar-btn')) {
                pinLiveCamera(gd);
            }
        }, true);
    }
    attach(40);
    return true;
}
"""


CLIENTSIDE_SERVER_SLIDER_SYNC = """
function(frameIdx) {
    if (frameIdx === undefined || frameIdx === null) {
        return 0;
    }
    return frameIdx;
}
"""


CLIENTSIDE_SERVER_DISABLE_TICK = """
function(_pause, _reset, dragValue) {
    const triggered =
        dash_clientside.callback_context.triggered;
    if (!triggered || triggered.length === 0) {
        return dash_clientside.no_update;
    }
    const id = triggered[0].prop_id.split(".")[0];
    if (id === "pause-btn" || id === "reset-btn") {
        return true;
    }
    if (id === "time-slider" &&
            dragValue !== undefined && dragValue !== null) {
        return true;
    }
    return dash_clientside.no_update;
}
"""
