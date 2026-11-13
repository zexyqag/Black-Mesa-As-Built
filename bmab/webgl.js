"use strict";
// Shared 3D viewer for bmab's HTML pages: packed faces (int16 positions, int8 normals, uint16 item, uint8
// kind 0 floor / 1 wall / 2 ceiling, uint32 indices; optional uint8 RGB per vertex, then float32 texture
// coordinates and uint16 texture layer per vertex), coloured per item from a texture or textured from a
// sheet, orbit camera, height slice, lines and translucent planes, click to pick an item or the point hit on it.
// No outside code.
function createViewer(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true });
  if (!gl) throw new Error("This browser has no WebGL2.");
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => scale(a, 1 / Math.hypot(...a));
  function mul(a, b) { const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; }
    return o; }
  function persp(fov, asp, n, f) { const t = 1 / Math.tan(fov / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; }
  function lookAt(e, c, u) { const z = norm(sub(e, c)), x = norm(cross(u, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]); }
  function prog(vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  const MESH = prog(`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNrm; layout(location=2) in uint aItem; layout(location=3) in uint aKind;
layout(location=4) in vec3 aCol; layout(location=5) in vec2 aUV; layout(location=6) in uint aLayer;
layout(location=7) in vec4 aM0; layout(location=8) in vec4 aM1; layout(location=9) in vec4 aM2; layout(location=10) in float aKeep;
uniform int uInst; uniform mat4 uMVP; uniform highp sampler2D uColors; uniform vec3 uShow; uniform float uTex; uniform int uSel; uniform vec3 uEye;
out vec3 vCol; out vec3 vItemCol; out float vLight; out float vZ; flat out uint vItem; out float vHide; out float vFacing;
out vec2 vUV; flat out uint vLayer; flat out float vSel; flat out float vDim; flat out float vKeep; out vec3 vAt;
void main() {
  vec4 c = texelFetch(uColors, ivec2(int(aItem) % 256, int(aItem) / 256), 0);
  float show = aKind == 0u ? uShow.x : (aKind == 1u ? uShow.y : uShow.z);
  vHide = (c.a < 0.5 || show < 0.5) ? 1.0 : 0.0;
  vec3 at = aPos, nr = aNrm;
  if (uInst == 1) { vec4 p = vec4(aPos, 1.0); at = vec3(dot(aM0, p), dot(aM1, p), dot(aM2, p)); nr = vec3(dot(aM0.xyz, aNrm), dot(aM1.xyz, aNrm), dot(aM2.xyz, aNrm)); }
  vec3 n = normalize(nr);
  vFacing = dot(n, uEye - at);
  vLight = 0.45 + 0.4 * max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0) + 0.15 * abs(n.z);
  vCol = mix(c.rgb, mix(aCol, c.rgb, 0.3), uTex); vItemCol = c.rgb; vZ = at.z; vItem = aItem;
  vUV = aUV; vLayer = aLayer; vSel = int(aItem) == uSel ? 1.0 : 0.0; vKeep = aKeep; vAt = at;
  vDim = c.a < 0.9 ? 0.5 : 1.0;  // alpha under 255 but shown: a dimmed item (its colour is dimmed already)
  gl_Position = uMVP * vec4(at, 1.0);
}`, `#version 300 es
precision highp float;
in vec3 vCol; in vec3 vItemCol; in float vLight; in float vZ; flat in uint vItem; in float vHide; in float vFacing;
in vec2 vUV; flat in uint vLayer; flat in float vSel; flat in float vDim; flat in float vKeep; in vec3 vAt;
uniform vec2 uSlice; uniform int uPick; uniform int uReal; uniform int uFront; uniform highp sampler2DArray uSheet;
uniform int uAlone; uniform int uBoxes; uniform vec3 uBoxC[8]; uniform mat3 uBox[8];  // one thing shown alone: its vertices, cut at its boxes (axes over half sizes)
uniform int uHide;  // leave out the vertices marked hidden
out vec4 o;
void main() {
  if (vHide > 0.5 || vZ < uSlice.x || vZ > uSlice.y || (uFront == 1 && vFacing < 0.0)) discard;
  int keep = int(vKeep + 0.5);  // 1: kept when one thing is shown alone, 2: hidden
  if (uHide == 1 && (keep & 2) != 0) discard;
  if (uAlone == 1) {  // a little slack: a wall on a box's side stays whole
    bool in_ = false;
    for (int k = 0; k < 8; k++) { if (k >= uBoxes) break; vec3 q = abs((vAt - uBoxC[k]) * uBox[k]); in_ = in_ || max(q.x, max(q.y, q.z)) <= 1.01; }
    if ((keep & 1) == 0 || !in_) discard;
  }
  vec3 col = vCol;
  bool real = uReal == 1 && vLayer != 65535u;
  if (real) {  // the face's own texture, lightly tinted by its item's colour
    vec4 t = texture(uSheet, vec3(vUV, float(vLayer)));
    if (t.a < 0.5) discard;  // see-through texels ({ textures)
    col = mix(t.rgb, vItemCol, 0.25) * vDim;
  }
  if (uPick == 1) { uint m = vItem + 1u; o = vec4(float(m & 255u) / 255.0, float((m >> 8) & 255u) / 255.0, 0.0, 1.0); return; }
  col *= vLight;
  if (vSel > 0.5) col = mix(col, vec3(1.0, 0.95, 0.6), real ? 0.25 : 0.6);  // the selected item stands out
  o = vec4(col, 1.0);
}`);
  const FLAT = prog(`#version 300 es
layout(location=0) in vec3 aPos; uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`, `#version 300 es
precision highp float; uniform vec4 uCol; out vec4 o; void main() { o = uCol; }`);

  const V = { lo: [0, 0, 0], hi: [1, 1, 1], show: [1, 1, 0], slice: [-1e9, 1e9], lines: [], planes: null,
              onPick: null, onClick: null, cam: null, textures: false, hasColours: false, real: false, hasSheet: false, selected: -1 };
  let vao = null, nIdx = 0, items = 1, pickFb = null, pickW = 0, pickH = 0, queued = false, geo = null;
  const colourTex = gl.createTexture();
  let sheetTex = null;
  const buffers = [];

  V.setGeometry = function (buf, layout, lo, hi, keepCamera) {
    for (const b of buffers) gl.deleteBuffer(b);
    buffers.length = 0;
    if (vao) gl.deleteVertexArray(vao);
    const [P, N, I, K, X] = layout;
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const attr = (loc, arr, size, type, int, normalized) => {
      const b = gl.createBuffer(); buffers.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0); else gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
    };
    geo = { pos: new Int16Array(buf, P[0], P[1]), nrm: new Int8Array(buf, N[0], N[1]), item: new Uint16Array(buf, I[0], I[1]), kind: new Uint8Array(buf, K[0], K[1]), idx: new Uint32Array(buf, X[0], X[1]),
            uv: layout.length > 7 ? new Float32Array(buf, layout[6][0], layout[6][1]) : null, layer: layout.length > 7 ? new Uint16Array(buf, layout[7][0], layout[7][1]) : null };
    attr(0, geo.pos, 3, gl.SHORT, false, false);
    attr(1, new Int8Array(buf, N[0], N[1]), 3, gl.BYTE, false, true);
    attr(2, new Uint16Array(buf, I[0], I[1]), 1, gl.UNSIGNED_SHORT, true);
    attr(3, new Uint8Array(buf, K[0], K[1]), 1, gl.UNSIGNED_BYTE, true);
    V.hasColours = layout.length > 5;  // optional sixth array: texture colour per vertex
    if (V.hasColours) attr(4, new Uint8Array(buf, layout[5][0], layout[5][1]), 3, gl.UNSIGNED_BYTE, false, true);
    else { gl.disableVertexAttribArray(4); gl.vertexAttrib3f(4, 0.5, 0.5, 0.5); }
    if (layout.length > 7) {  // optional seventh and eighth: texture coordinates and layer per vertex
      attr(5, new Float32Array(buf, layout[6][0], layout[6][1]), 2, gl.FLOAT, false, false);
      attr(6, new Uint16Array(buf, layout[7][0], layout[7][1]), 1, gl.UNSIGNED_SHORT, true);
    } else {
      gl.disableVertexAttribArray(5); gl.vertexAttrib2f(5, 0, 0);
      gl.disableVertexAttribArray(6); gl.vertexAttribI4ui(6, 65535, 0, 0, 0);
    }
    const ib = gl.createBuffer(); buffers.push(ib); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(buf, X[0], X[1]), gl.STATIC_DRAW);
    nIdx = X[1];
    gl.disableVertexAttribArray(10);
    gl.bindVertexArray(null);
    if (flags) gl.deleteBuffer(flags);
    V.lo = lo; V.hi = hi; grid = null; alone = null; hidden = null; flags = null;
    if (!keepCamera || !V.cam) V.resetCam();
  };
  V.geometry = () => geo;
  V.setItems = function (item, models) {  // which item each vertex is, changed (as many as built), and the item the models are drawn as
    geo.item.set(item);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[2]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, geo.item);
    if (shapes) shapes.item = models;
    V.draw();
  };
  let alone = null, hidden = null, flags = null;  // the thing shown alone {keep, n, c, m}, the vertices left out, and the buffer of both
  gl.vertexAttrib1f(10, 1.0);  // everything drawn without a flags array is kept and shown
  function setFlags() {  // one byte a vertex: 1 kept when one thing is shown alone, 2 hidden
    gl.bindVertexArray(vao);
    if (flags) { gl.deleteBuffer(flags); gl.disableVertexAttribArray(10); }
    flags = null;
    if (alone || hidden) {
      const n = (alone || hidden).keep.length, f = new Uint8Array(n);
      if (alone) f.set(alone.keep); else f.fill(1);
      if (hidden) for (let i = 0; i < n; i++) if (hidden.keep[i]) f[i] |= 2;
      flags = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, flags); gl.bufferData(gl.ARRAY_BUFFER, f, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(10); gl.vertexAttribPointer(10, 1, gl.UNSIGNED_BYTE, false, 0, 0);
    }
    gl.bindVertexArray(null);
    V.draw();
  }
  V.alone = function (keep, boxes) {  // show only the vertices keep (one byte each, 1: kept) marks, cut at up to 8 boxes [{c, axes, half}] (local units); null: everything
    alone = null;
    if (keep) {
      const kept = boxes.slice(0, 8), m = kept.map(box => box.axes.flatMap((a, k) => a.map(v => v / box.half[k])));  // as columns: (p - c) * M dots p - c with each axis over its half size
      alone = { keep, n: kept.length, c: kept.flatMap(box => box.c), m: m.flat() };
    }
    setFlags();
  };
  V.hide = function (marked) {  // leave out the vertices marked (one byte each, 1: left out); null: none
    hidden = marked ? { keep: marked } : null;
    setFlags();
  };
  V.offsetVertices = function (offsetOf) {  // slide vertices: offsetOf(x, y, z, item, vertex) gives [dx, dy, dz] in position units, or null; null: back as built
    if (!geo || (!offsetOf && !geo.base)) return;
    if (!geo.base) geo.base = geo.pos.slice();
    const out = new Int16Array(geo.base.length), b = geo.base, clamp = v => Math.max(-32767, Math.min(32767, Math.round(v)));
    for (let i = 0; i < geo.item.length; i++) {
      const o = offsetOf ? offsetOf(b[i * 3], b[i * 3 + 1], b[i * 3 + 2], geo.item[i], i) : null;
      for (let a = 0; a < 3; a++) out[i * 3 + a] = o ? clamp(b[i * 3 + a] + o[a]) : b[i * 3 + a];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, out); V.draw();
  };
  let extra = null;
  V.setExtra = function (pos, nrm, item, kind, uv, layer, idx, col) {  // more triangles drawn with the mesh, replaced on every call
    if (extra) { gl.deleteVertexArray(extra.vao); extra.bufs.forEach(b => gl.deleteBuffer(b)); extra = null; }
    if (!idx.length) { V.draw(); return; }
    const vao = gl.createVertexArray(), bufs = []; gl.bindVertexArray(vao);
    const attr = (loc, arr, size, type, int) => {
      const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0); else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
    };
    attr(0, pos, 3, gl.FLOAT, false); attr(1, nrm, 3, gl.FLOAT, false); attr(2, item, 1, gl.UNSIGNED_SHORT, true); attr(3, kind, 1, gl.UNSIGNED_BYTE, true);
    attr(4, col, 3, gl.FLOAT, false);
    attr(5, uv, 2, gl.FLOAT, false); attr(6, layer, 1, gl.UNSIGNED_SHORT, true);
    const ib = gl.createBuffer(); bufs.push(ib); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    extra = { vao, bufs, n: idx.length };
    V.draw();
  };
  let shapes = null;
  V.showModels = false;
  V.solo = null;  // {part, copy}: draw that one copy of one model and nothing else
  V.setModels = function (m) {  // the 3D models a map places, drawn as many copies of a few shapes. m: {pos (whole numbers), nrm, uv (half floats), layer, idx, inst (12 numbers a copy: its matrix, scale included), parts [[first index, indices, first copy, copies]], item}; null: none
    if (shapes) { gl.deleteVertexArray(shapes.vao); shapes.bufs.forEach(b => gl.deleteBuffer(b)); shapes = null; }
    if (!m) { V.draw(); return; }
    const vao = gl.createVertexArray(), bufs = []; gl.bindVertexArray(vao);
    const attr = (loc, arr, size, type, int, normalized) => {
      const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0); else gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
    };
    attr(0, m.pos, 3, gl.SHORT); attr(1, m.nrm, 3, gl.BYTE, false, true); attr(5, m.uv, 2, gl.HALF_FLOAT); attr(6, m.layer, 1, gl.UNSIGNED_SHORT, true);
    for (const loc of [2, 3, 4]) gl.disableVertexAttribArray(loc);
    const ib = gl.createBuffer(); bufs.push(ib); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.STATIC_DRAW);
    const inst = gl.createBuffer(); bufs.push(inst); gl.bindBuffer(gl.ARRAY_BUFFER, inst); gl.bufferData(gl.ARRAY_BUFFER, m.inst, gl.STATIC_DRAW);
    for (const loc of [7, 8, 9]) { gl.enableVertexAttribArray(loc); gl.vertexAttribDivisor(loc, 1); }
    gl.bindVertexArray(null);
    shapes = { vao, bufs, inst, parts: m.parts, item: m.item };
    V.draw();
  };
  V.setColours = function (rgba, n) {  // 4 bytes per item: colour, alpha 255 = shown, 128 to 254 = shown dimmed
    items = n;
    const rows = Math.max(1, Math.ceil(n / 256)), data = new Uint8Array(256 * rows * 4);
    data.set(rgba.subarray(0, Math.min(rgba.length, data.length)));
    gl.bindTexture(gl.TEXTURE_2D, colourTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    V.draw();
  };
  V.setSheet = function (img, rects, more, moreRects) {  // textures: rects [x, y, w, h] of a loaded image, one layer each, and those of a second image after them
    if (sheetTex) gl.deleteTexture(sheetTex);
    sheetTex = null; V.hasSheet = false;
    if (!img || !rects.length) return;
    const every = [...rects.map(r => [img, r]), ...(more ? moreRects.map(r => [more, r]) : [])], L = every.length > 300 ? 128 : 256;  // every layer one size; the coordinates are in repeats
    const cv = document.createElement("canvas"); cv.width = cv.height = L;
    const cx = cv.getContext("2d"); cx.imageSmoothingEnabled = false;
    sheetTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D_ARRAY, sheetTex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, Math.log2(L) + 1, gl.RGBA8, L, L, every.length);
    every.forEach(([from, r], k) => {
      if (!r) return;
      const [x, y, w, h] = r;
      cx.clearRect(0, 0, L, L); cx.drawImage(from, x, y, w, h, 0, 0, L, L);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, k, L, L, 1, gl.RGBA, gl.UNSIGNED_BYTE, cx.getImageData(0, 0, L, L));
    });
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
    V.hasSheet = true; V.draw();
  };
  function flatBuffer(arr) { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); return b; }
  function ribbons(pairs, width) {  // line segments as flat strips that many CSS pixels wide, facing the camera
    const e = eyeAt(), out = [], perPixel = 2 * Math.tan(0.45) / innerHeight;
    for (let i = 0; i + 5 < pairs.length; i += 6) {
      const a = [pairs[i], pairs[i + 1], pairs[i + 2]], b = [pairs[i + 3], pairs[i + 4], pairs[i + 5]], side = cross(sub(b, a), sub(a, e)), l = Math.hypot(...side);
      if (l < 1e-9) continue;
      const u = scale(side, 1 / l), ha = scale(u, width / 2 * perPixel * Math.hypot(...sub(a, e))), hb = scale(u, width / 2 * perPixel * Math.hypot(...sub(b, e)));
      out.push(...add(a, ha), ...sub(a, ha), ...add(b, hb), ...add(b, hb), ...sub(a, ha), ...sub(b, hb));
    }
    return out;
  }
  V.setLines = function (sets) {  // [{pairs: segment ends as x, y, z, ..., colour: [r, g, b, a], width: CSS pixels, optional}], drawn over everything; call again when the camera moves if any has a width
    for (const s of V.lines) gl.deleteBuffer(s.buf);
    V.lines = sets.filter(s => s.pairs.length).map(s => {
      const pts = s.width ? ribbons(s.pairs, s.width) : s.pairs;
      return { buf: flatBuffer(new Float32Array(pts)), n: pts.length / 3, colour: s.colour, mode: s.width ? gl.TRIANGLES : gl.LINES };
    });
  };
  V.setPlanes = function (zs, colour) {  // horizontal planes over the bounds at these heights
    const q = z => [V.lo[0], V.lo[1], z, V.hi[0], V.lo[1], z, V.hi[0], V.hi[1], z, V.lo[0], V.lo[1], z, V.hi[0], V.hi[1], z, V.lo[0], V.hi[1], z];
    V.planes = zs.length ? { buf: flatBuffer(new Float32Array(zs.flatMap(q))), n: zs.length * 6, colour } : null;
  };
  V.setPatch = function (corners, colour) {  // translucent triangles (x, y, z per corner) drawn over the mesh; none: nothing
    V.patch = corners.length ? { buf: flatBuffer(new Float32Array(corners)), n: corners.length / 3, colour } : null;
    V.draw();
  };
  V.showLines = true; V.showPlanes = true;
  V.resetCam = function () {
    const c = [(V.lo[0] + V.hi[0]) / 2, (V.lo[1] + V.hi[1]) / 2, (V.lo[2] + V.hi[2]) / 2];
    V.cam = { target: c, dist: Math.max(V.hi[0] - V.lo[0], V.hi[1] - V.lo[1], V.hi[2] - V.lo[2]) * 1.1, yaw: -0.7, pitch: 0.6 };
  };
  function eyeAt() {
    const cam = V.cam;
    return add(cam.target, scale([Math.cos(cam.pitch) * Math.cos(cam.yaw), Math.cos(cam.pitch) * Math.sin(cam.yaw), Math.sin(cam.pitch)], cam.dist));
  }
  function mvp() {
    const cam = V.cam, w = canvas.width, h = canvas.height, eye = eyeAt();
    return mul(persp(0.9, w / h, cam.dist / 200, cam.dist * 20), lookAt(eye, cam.target, [0, 0, 1]));
  }
  function drawMesh(pick) {
    gl.useProgram(MESH);
    gl.uniformMatrix4fv(gl.getUniformLocation(MESH, "uMVP"), false, mvp());
    gl.uniform3fv(gl.getUniformLocation(MESH, "uShow"), V.show);
    gl.uniform3fv(gl.getUniformLocation(MESH, "uEye"), eyeAt());
    gl.uniform1i(gl.getUniformLocation(MESH, "uInst"), 0);
    gl.uniform1i(gl.getUniformLocation(MESH, "uFront"), V.front ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(MESH, "uTex"), V.textures && V.hasColours ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(MESH, "uSel"), V.selected);
    gl.uniform2fv(gl.getUniformLocation(MESH, "uSlice"), V.slice);
    gl.uniform1i(gl.getUniformLocation(MESH, "uPick"), pick ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(MESH, "uAlone"), alone ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(MESH, "uHide"), hidden ? 1 : 0);
    if (alone) { gl.uniform1i(gl.getUniformLocation(MESH, "uBoxes"), alone.n); gl.uniform3fv(gl.getUniformLocation(MESH, "uBoxC"), alone.c); gl.uniformMatrix3fv(gl.getUniformLocation(MESH, "uBox"), false, alone.m); }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, colourTex);
    gl.uniform1i(gl.getUniformLocation(MESH, "uColors"), 0);
    gl.uniform1i(gl.getUniformLocation(MESH, "uReal"), V.real && V.hasSheet ? 1 : 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D_ARRAY, sheetTex);
    gl.uniform1i(gl.getUniformLocation(MESH, "uSheet"), 1);
    gl.activeTexture(gl.TEXTURE0);
    const solo = shapes && V.solo ? V.solo : null;
    if (!solo) {
      gl.bindVertexArray(vao); gl.drawElements(gl.TRIANGLES, nIdx, gl.UNSIGNED_INT, 0);
      if (extra && !pick) { gl.bindVertexArray(extra.vao); gl.drawElements(gl.TRIANGLES, extra.n, gl.UNSIGNED_INT, 0); }
    }
    if (shapes && (V.showModels || solo) && !pick) {  // clicks go through them to the map
      gl.uniform1i(gl.getUniformLocation(MESH, "uInst"), 1);
      gl.bindVertexArray(shapes.vao);
      gl.vertexAttribI4ui(2, shapes.item, 0, 0, 0); gl.vertexAttribI4ui(3, 1, 0, 0, 0); gl.vertexAttrib3f(4, 0.5, 0.5, 0.5);
      gl.bindBuffer(gl.ARRAY_BUFFER, shapes.inst);
      shapes.parts.forEach(([first, count, copy, copies], part) => {
        if (solo && part !== solo.part) return;
        const from = solo ? copy + solo.copy : copy;
        [7, 8, 9].forEach((loc, k) => gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 48, (from * 12 + k * 4) * 4));
        gl.drawElementsInstanced(gl.TRIANGLES, count, gl.UNSIGNED_INT, first * 4, solo ? 1 : copies);
      });
      gl.uniform1i(gl.getUniformLocation(MESH, "uInst"), 0);
    }
    gl.bindVertexArray(null);
  }
  function flat(o, mode) {
    gl.useProgram(FLAT);
    gl.uniformMatrix4fv(gl.getUniformLocation(FLAT, "uMVP"), false, mvp());
    gl.uniform4fv(gl.getUniformLocation(FLAT, "uCol"), o.colour);
    gl.bindBuffer(gl.ARRAY_BUFFER, o.buf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(mode, 0, o.n); gl.disableVertexAttribArray(0);
  }
  V.render = function () {
    if (!vao) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.043, 0.043, 0.063, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    drawMesh(false);
    if (V.patch) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); flat(V.patch, gl.TRIANGLES); gl.depthMask(true); gl.disable(gl.BLEND); }
    if (V.showLines) { gl.disable(gl.DEPTH_TEST); for (const s of V.lines) flat(s, s.mode); gl.enable(gl.DEPTH_TEST); }
    if (V.planes && V.showPlanes) {
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      flat(V.planes, gl.TRIANGLES); gl.depthMask(true); gl.disable(gl.BLEND);
    }
  };
  V.draw = function () {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; V.render(); });
  };
  V.pick = function (x, y) {
    const w = canvas.width, h = canvas.height;
    if (!pickFb || pickW !== w || pickH !== h) {
      pickFb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, pickFb);
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      const d = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, d); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, d);
      pickW = w; pickH = h;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFb); gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    drawMesh(true);
    const px = new Uint8Array(4); gl.readPixels(x, h - y - 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return px[0] + px[1] * 256 - 1;  // item, or -1
  };

  V.ray = function (x, y) {  // CSS pixels: the camera's position and the unit direction through them
    const cam = V.cam, eye = eyeAt();
    const fwd = norm(sub(cam.target, eye)), right = norm(cross(fwd, [0, 0, 1])), up = cross(right, fwd), t = Math.tan(0.9 / 2);
    return { eye, d: norm(add(fwd, add(scale(right, (2 * x / innerWidth - 1) * t * innerWidth / innerHeight), scale(up, (1 - 2 * y / innerHeight) * t)))) };
  };
  V.project = function (p) {  // a point to CSS pixels, or null if it is behind the camera
    const m = mvp(), w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (w <= 0) return null;
    return [((m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w * 0.5 + 0.5) * innerWidth, (0.5 - (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w * 0.5) * innerHeight];
  };
  let grid = null;  // triangles by cubic cell, so that a ray only tests those along its way
  function buildGrid() {
    const P = geo.pos, I = geo.idx, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let a = 0; a < 3; a++) { if (P[i + a] < lo[a]) lo[a] = P[i + a]; if (P[i + a] > hi[a]) hi[a] = P[i + a]; }
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1) / 96, dims = [0, 1, 2].map(a => Math.floor((hi[a] - lo[a]) / size) + 1), cells = new Map();
    const cell = (v, a) => Math.min(dims[a] - 1, Math.floor((v - lo[a]) / size));
    for (let k = 0; k < I.length; k += 3) {
      const r = [0, 1, 2].map(a => { const v = [P[I[k] * 3 + a], P[I[k + 1] * 3 + a], P[I[k + 2] * 3 + a]]; return [cell(Math.min(...v), a), cell(Math.max(...v), a)]; });
      for (let x = r[0][0]; x <= r[0][1]; x++) for (let y = r[1][0]; y <= r[1][1]; y++) for (let z = r[2][0]; z <= r[2][1]; z++) {
        const key = (x * dims[1] + y) * dims[2] + z;
        let list = cells.get(key);
        if (!list) cells.set(key, list = []);
        list.push(k);
      }
    }
    grid = { lo, hi, size, dims, cells };
  }
  V.cast = function (o, d, far) {  // the nearest surface along a ray, shown or hidden: {t, normal towards where the ray came from}, or null within far
    if (!grid) buildGrid();
    const P = geo.pos, I = geo.idx, [ox, oy, oz] = o, [dx, dy, dz] = d, { lo, hi, size, dims, cells } = grid;
    let t0 = 0, t1 = far;
    for (let a = 0; a < 3; a++) {
      if (Math.abs(d[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return null; continue; }
      let ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
      if (t0 > t1) return null;
    }
    const at0 = [0, 1, 2].map(a => Math.max(0, Math.min(dims[a] - 1, Math.floor((o[a] + d[a] * t0 - lo[a]) / size))));
    const step = [0, 1, 2].map(a => d[a] > 0 ? 1 : -1), delta = [0, 1, 2].map(a => Math.abs(d[a]) < 1e-12 ? Infinity : size / Math.abs(d[a]));
    const next = [0, 1, 2].map(a => Math.abs(d[a]) < 1e-12 ? Infinity : ((at0[a] + (d[a] > 0 ? 1 : 0)) * size + lo[a] - o[a]) / d[a]);
    let [x, y, z] = at0, best = far, hit = -1;
    for (;;) {
      const list = cells.get((x * dims[1] + y) * dims[2] + z);
      if (list) for (const k of list) {
        const a = I[k] * 3, b = I[k + 1] * 3, c = I[k + 2] * 3, ax = P[a], ay = P[a + 1], az = P[a + 2];
        const e1x = P[b] - ax, e1y = P[b + 1] - ay, e1z = P[b + 2] - az, e2x = P[c] - ax, e2y = P[c + 1] - ay, e2z = P[c + 2] - az;
        const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x, det = e1x * hx + e1y * hy + e1z * hz;
        if (det > -1e-9 && det < 1e-9) continue;
        const sx = ox - ax, sy = oy - ay, sz = oz - az, u = (sx * hx + sy * hy + sz * hz) / det;
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x, v = (dx * qx + dy * qy + dz * qz) / det;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) / det;
        if (t > 1e-3 && t < best) { best = t; hit = k; }
      }
      const axis = next[0] < next[1] ? (next[0] < next[2] ? 0 : 2) : (next[1] < next[2] ? 1 : 2), leave = next[axis];
      if (best <= leave || leave > t1) break;
      if (axis === 0) { x += step[0]; if (x < 0 || x >= dims[0]) break; }
      else if (axis === 1) { y += step[1]; if (y < 0 || y >= dims[1]) break; }
      else { z += step[2]; if (z < 0 || z >= dims[2]) break; }
      next[axis] += delta[axis];
    }
    if (hit < 0) return null;
    const a = I[hit] * 3, b = I[hit + 1] * 3, c = I[hit + 2] * 3;
    let n = norm(cross([P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]], [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]]));
    if (dot(n, d) > 0) n = scale(n, -1);
    return { t: best, normal: n, tri: hit };
  };

  V.hit = function (x, y) {  // CSS pixels: the item under them, the point hit on it and the surface's normal towards the camera, or null
    const r = devicePixelRatio || 1, item = V.pick(Math.round(x * r), Math.round(y * r));
    if (item < 0 || !geo) return null;
    const { eye, d } = V.ray(x, y);
    const P = geo.pos, vtx = i => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
    let best = Infinity, normal = null;
    for (let k = 0; k < geo.idx.length; k += 3) {  // the nearest triangle of that item along the ray that can be seen
      const i0 = geo.idx[k];
      if (geo.item[i0] !== item || !V.show[geo.kind[i0]] || (V.front && geo.nrm[i0 * 3] * d[0] + geo.nrm[i0 * 3 + 1] * d[1] + geo.nrm[i0 * 3 + 2] * d[2] > 0)) continue;
      const a = vtx(geo.idx[k]), e1 = sub(vtx(geo.idx[k + 1]), a), e2 = sub(vtx(geo.idx[k + 2]), a);
      const h = cross(d, e2), det = dot(e1, h);
      if (Math.abs(det) < 1e-9) continue;
      const s = sub(eye, a), u = dot(s, h) / det, q = cross(s, e1), v = dot(d, q) / det, tt = dot(e2, q) / det;
      if (u >= 0 && v >= 0 && u + v <= 1 && tt > 0 && tt < best) { best = tt; normal = norm(cross(e1, e2)); if (dot(normal, d) > 0) normal = scale(normal, -1); }
    }
    return best < Infinity ? { item, point: add(eye, scale(d, best)), normal } : null;
  };

  // flying, like noclip: the mouse looks, keys (V.flyKeys) move along the view and up and down, the wheel sets the speed
  V.flying = false;
  V.flyKeys = { forward: "w", back: "s", left: "a", right: "d", up: "e", down: "q", fast: "shift" };
  const keys = new Set();
  let flyTime = 0, flySpeed = 0, locked = false, cursor = [0, 0];  // cursor: where a handle being dragged has got to
  const centre = () => [innerWidth / 2, innerHeight / 2];
  function aim(eye) {
    const c = V.cam;
    c.target = sub(eye, scale([Math.cos(c.pitch) * Math.cos(c.yaw), Math.cos(c.pitch) * Math.sin(c.yaw), Math.sin(c.pitch)], c.dist));
  }
  function flyFrame(now) {
    if (!V.flying) return;
    const dt = Math.min(0.1, (now - flyTime) / 1000), c = V.cam, look = norm(sub(c.target, eyeAt())), right = norm(cross(look, [0, 0, 1])), up = [0, 0, 1];
    flyTime = now;
    let way = [0, 0, 0];
    const K = V.flyKeys;
    for (const [k, v] of [[K.forward, look], [K.back, scale(look, -1)], [K.right, right], [K.left, scale(right, -1)], [K.up, up], [K.down, scale(up, -1)]]) if (keys.has(k)) way = add(way, v);
    if (Math.hypot(...way) > 1e-6) {
      c.target = add(c.target, scale(norm(way), flySpeed * (keys.has(K.fast) ? 3 : 1) * dt));
      moved();
    }
    requestAnimationFrame(flyFrame);
  }
  function moved() {
    if (V.onCamera) V.onCamera();
    if (V.onHover) V.onHover(...centre());
    V.draw();
  }
  V.setFly = function (on, speed) {  // speed: position units a second
    if (on === V.flying) return;
    V.flying = on; keys.clear();
    if (on) {
      const eye = eyeAt();
      V.cam.dist = Math.max(Math.max(V.hi[0] - V.lo[0], V.hi[1] - V.lo[1], V.hi[2] - V.lo[2]) / 16, 64);  // what the near and far planes follow
      aim(eye); flySpeed = speed; flyTime = performance.now();
      const asked = canvas.requestPointerLock();
      if (asked && asked.catch) asked.catch(() => V.setFly(false));
      requestAnimationFrame(flyFrame);
    } else if (document.pointerLockElement === canvas) document.exitPointerLock();
    if (V.onFly) V.onFly(on);
    V.draw();
  };
  document.addEventListener("pointerlockchange", () => {
    const now = document.pointerLockElement === canvas;
    if (locked && !now) V.setFly(false);  // Esc
    locked = now;
  });
  document.addEventListener("pointerlockerror", () => V.setFly(false));
  addEventListener("blur", () => keys.clear());
  addEventListener("keydown", e => { if (V.flying) { keys.add(e.key.toLowerCase()); e.preventDefault(); } });
  addEventListener("keyup", e => keys.delete(e.key.toLowerCase()));

  function resize() { const r = devicePixelRatio || 1; canvas.width = innerWidth * r; canvas.height = innerHeight * r; V.draw(); }
  addEventListener("resize", resize);
  let drag = null, held = false;
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("mousedown", e => {
    if (V.flying) {  // the crosshair clicks
      if (e.button !== 0) return;
      const [x, y] = centre(), r = devicePixelRatio || 1;
      if (V.onGrab && V.onGrab(x, y)) { held = true; cursor = [x, y]; return; }
      if (V.onClick) V.onClick(x, y); else if (V.onPick) V.onPick(V.pick(Math.round(x * r), Math.round(y * r)), x, y);
      return;
    }
    if (e.button === 0 && V.onGrab && V.onGrab(e.clientX, e.clientY)) { held = true; return; }  // a handle: the camera stays put
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2, moved: false };
  });
  addEventListener("mouseup", e => {
    if (held) { held = false; V.onDrop(); return; }
    if (drag && !drag.moved && e.button === 0 && V.onClick) V.onClick(e.clientX, e.clientY);
    else if (drag && !drag.moved && e.button === 0 && V.onPick) {
      const r = devicePixelRatio || 1;
      V.onPick(V.pick(Math.round(e.clientX * r), Math.round(e.clientY * r)), e.clientX, e.clientY);
    }
    drag = null;
  });
  addEventListener("mousemove", e => {
    if (V.flying) {
      if (held) { cursor = [cursor[0] + e.movementX, cursor[1] + e.movementY]; return V.onDrag(cursor[0], cursor[1], e.shiftKey, e.ctrlKey); }
      if (locked) {
        const eye = eyeAt(), c = V.cam;
        c.yaw -= e.movementX * 0.0025; c.pitch = Math.max(-1.55, Math.min(1.55, c.pitch + e.movementY * 0.0025));
        aim(eye); moved();
      }
      return;
    }
    if (held) return V.onDrag(e.clientX, e.clientY, e.shiftKey, e.ctrlKey);
    if (!drag && V.onHover && e.target === canvas) return V.onHover(e.clientX, e.clientY);
    if (!drag || !V.cam) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y, cam = V.cam;
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const f = cam.dist / innerHeight * 1.1;
      const right = [-Math.sin(cam.yaw), Math.cos(cam.yaw), 0];
      const up = [-Math.sin(cam.pitch) * Math.cos(cam.yaw), -Math.sin(cam.pitch) * Math.sin(cam.yaw), Math.cos(cam.pitch)];
      cam.target = add(cam.target, add(scale(right, -dx * f), scale(up, dy * f)));
    } else {
      cam.yaw -= dx * 0.005; cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch + dy * 0.005));
    }
    if (V.onCamera) V.onCamera();
    V.draw();
  });
  canvas.addEventListener("wheel", e => { e.preventDefault(); if (V.flying) { flySpeed *= Math.pow(1.0015, -e.deltaY); return; } if (V.cam) { V.cam.dist *= Math.pow(1.0015, e.deltaY); if (V.onCamera) V.onCamera(); V.draw(); } }, { passive: false });
  resize();
  return V;
}

function decodeBase64(b64) {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64).buffer;
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u.buffer;
}
