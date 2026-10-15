'use strict';

// One SSE connection carries the whole state object at 5 Hz. EventSource reconnects by
// itself, so a restart of the server (or of the drone) just resumes the page.
let lastState = null;
// The state arrives at 5 Hz, but rebuilding a table that often destroys the buttons under
// the pointer mid-click and makes the list flicker. Both lists are rebuilt only when their
// content actually changed.
let unitsSignature = '';
let eventsSignature = '';

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) =>
  (value === null || value === undefined || Number.isNaN(value)) ? '--' : Number(value).toFixed(digits);

function pill(element, text, kind) {
  element.textContent = text;
  const parent = element.parentElement;
  parent.classList.remove('ok', 'warnp', 'badp');
  if (kind) parent.classList.add(kind);
}

function render(state) {
  lastState = state;
  const flight = state.flight, board = state.board, link = state.link;

  $('domain').textContent = link.domain_id;
  $('host').textContent = link.host;

  pill($('link-state'), link.live ? 'live' : 'no data', link.live ? 'ok' : 'badp');
  pill($('board-state'), board.online ? (board.uptime || 'online') : 'unreachable',
       board.online ? 'ok' : 'badp');

  if (flight.armed === true) pill($('armed-state'), 'ARMED', 'badp');
  else if (flight.armed === false) pill($('armed-state'), 'disarmed', 'ok');
  else pill($('armed-state'), 'unknown', 'warnp');

  // --- banner: what is in charge right now
  const banner = $('control-banner');
  banner.className = 'banner';
  if (!link.live) {
    banner.textContent = board.online
      ? 'No telemetry — start drone-control below'
      : 'No telemetry — board unreachable';
    banner.classList.add('bad');
  } else if (flight.in_control) {
    banner.textContent = 'RB3 is commanding the drone — ' + (flight.reason || '');
    banner.classList.add('warn');
  } else {
    banner.textContent = 'Radio has control — ' + (flight.reason || 'idle');
    banner.classList.add('ok');
  }

  $('roll').textContent = fmt(flight.roll_deg, 1);
  $('pitch').textContent = fmt(flight.pitch_deg, 1);
  $('yaw').textContent = fmt(flight.yaw_deg, 0);
  $('alt').textContent = fmt(flight.altitude_m, 2);
  $('baro').textContent = fmt(flight.baro_altitude_m, 2);
  $('batt').textContent = fmt(flight.battery_v, 2);

  const flags = $('flags');
  flags.innerHTML = '';
  (flight.arming_flags || []).forEach((name) => {
    const span = document.createElement('span');
    span.className = 'flag';
    span.textContent = name;
    flags.appendChild(span);
  });

  for (let i = 0; i < 4; i++) {
    $('s' + i).textContent = flight.sticks_sent ? flight.sticks_sent[i] : '-';
    $('f' + i).textContent = flight.sticks_fc ? flight.sticks_fc[i] : '-';
  }

  renderSensors(flight, link);
  renderPosition(state.position);
  updateTrack(state.track);

  renderCameras(state.cameras);

  // --- board facts
  const kv = $('board-kv');
  const rows = board.online ? [
    ['uptime', board.uptime || '-'],
    ['load', board.load || '-'],
    ['temperature', board.temp_c ? board.temp_c + ' °C' : '-'],
    ['memory', board.mem_percent !== null ? board.mem_percent + ' %' : '-'],
    ['disk /', board.disk_percent !== null ? board.disk_percent + ' %' : '-'],
    ['FC port', board.fc_port ? '/dev/ttyACM0 present' : 'missing'],
    ['port held by', board.fc_port_holder || '(free)'],
    ['sensor streams', board.see_streams + ' / 4'],
    ['checked', board.age !== null ? board.age + ' s ago' : '-'],
  ] : [['state', 'unreachable'], ['reason', board.reason || '-']];
  kv.innerHTML = rows.map(([key, value]) =>
    '<dt>' + key + '</dt><dd>' + String(value) + '</dd>').join('');

  renderUnits(state.units);
  renderEvents(state.events);
  drawPlots(state.plots, state.window_seconds);
  const label = $('history-window');
  if (label) label.textContent = 'last ' + Math.round((state.window_seconds || 120) / 60 * 10) / 10 + ' minutes';
}

// --- cameras: one view per tile, each an MJPEG stream of the topic picked in its menu.
// The server only receives a topic from the board while some stream shows it, so a view
// is what makes its frames cross the link. Browsers allow 6 connections per site and the
// state stream takes one, hence the cap.
const MAX_VIEWS = 4;
const VIEWS_KEY = 'drone_gui.cameraViews';
let views = null;                 // [{topic, wide, ...elements}], built on the first state
let cameraTopics = [];
let topicsSignature = '';

function saveViews() {
  try {
    localStorage.setItem(VIEWS_KEY, JSON.stringify(views.map((v) => ({topic: v.topic, wide: v.wide}))));
  } catch (error) { /* private window or blocked storage: the layout just is not kept */ }
}

function loadViews(defaultTopic) {
  try {
    const saved = JSON.parse(localStorage.getItem(VIEWS_KEY) || 'null');
    if (Array.isArray(saved) && saved.length) {
      return saved.slice(0, MAX_VIEWS).filter((v) => v && typeof v.topic === 'string');
    }
  } catch (error) { /* fall through to the default */ }
  return [{topic: defaultTopic, wide: false}];
}

function startStream(view) {
  view.img.src = '/camera/stream?topic=' + encodeURIComponent(view.topic) + '&t=' + Date.now();
}

function addView(topic, wide) {
  const view = {topic, wide: !!wide, removed: false};
  view.el = document.createElement('div');
  view.el.className = 'view';
  view.el.innerHTML =
    '<div class="view-bar"><select></select><span class="view-info"></span>' +
    '<button class="icon" title="full width">&#x2922;</button>' +
    '<button class="icon danger" title="remove this view">&times;</button></div>' +
    '<div class="camera-wrap"><img alt=""><div class="camera-placeholder"></div></div>';
  view.select = view.el.querySelector('select');
  view.info = view.el.querySelector('.view-info');
  view.img = view.el.querySelector('img');
  view.placeholder = view.el.querySelector('.camera-placeholder');
  const [wideButton, removeButton] = view.el.querySelectorAll('button');

  view.select.onchange = () => {
    view.topic = view.select.value;
    view.img.classList.remove('live');
    startStream(view);
    saveViews();
    updateView(view);
  };
  wideButton.onclick = () => {
    view.wide = !view.wide;
    view.el.classList.toggle('wide', view.wide);
    saveViews();
  };
  removeButton.onclick = () => {
    view.removed = true;
    view.img.removeAttribute('src');          // closes the stream, so the topic is unwatched
    view.el.remove();
    views = views.filter((v) => v !== view);
    saveViews();
    updateAddButton();
  };
  // A stream that drops (panel restarted, network blip) is reopened.
  view.img.onerror = () => {
    if (!view.removed && view.img.getAttribute('src')) setTimeout(() => { if (!view.removed) startStream(view); }, 2000);
  };
  view.el.classList.toggle('wide', view.wide);
  $('camera-grid').appendChild(view.el);
  views.push(view);
  fillTopicMenu(view, true);
  startStream(view);
  updateView(view);
  updateAddButton();
  return view;
}

function updateAddButton() {
  const button = $('camera-add');
  button.disabled = views.length >= MAX_VIEWS;
  button.title = button.disabled
    ? 'at most ' + MAX_VIEWS + ' views: browsers open only 6 connections to the panel'
    : 'show another camera topic';
}

function topicLabel(entry) {
  return entry.topic + (entry.kind === 'raw' ? '  (raw)' : '') +
    (entry.publishers === 0 ? '  - no publisher' : '');
}

// Rebuilt only when the topic list changes, and never while the menu is open.
function fillTopicMenu(view, force) {
  if (!force && document.activeElement === view.select) return;
  const entries = cameraTopics.slice();
  if (!entries.some((e) => e.topic === view.topic)) {
    entries.unshift({topic: view.topic, kind: null, publishers: null, missing: true});
  }
  view.select.innerHTML = '';
  entries.forEach((entry) => {
    const option = document.createElement('option');
    option.value = entry.topic;
    option.textContent = entry.missing ? entry.topic + '  - not on the board' : topicLabel(entry);
    view.select.appendChild(option);
  });
  view.select.value = view.topic;
}

function updateView(view) {
  const entry = cameraTopics.find((e) => e.topic === view.topic);
  const live = !!entry && entry.age !== null && entry.age < 3;
  view.img.classList.toggle('live', live);
  view.placeholder.style.display = live ? 'none' : 'block';

  let info, dead = false;
  if (!entry) {
    info = 'not on the board';
    dead = true;
  } else if (live) {
    info = fmt(entry.fps, 1) + ' fps' +
      (entry.source_hz ? ' of ' + fmt(entry.source_hz, 1) + ' published' : '') +
      (entry.kind === 'raw' ? ' · JPEG made on the board' : '');
  } else if (entry.publishers === 0) {
    info = 'no publisher';
    dead = true;
  } else {
    info = 'waiting for frames';
  }
  view.info.textContent = info;
  view.info.classList.toggle('dead', dead);
  view.placeholder.textContent = !entry
    ? view.topic + ' is not in the board\'s ROS graph - is its node running?'
    : entry.publishers === 0 ? 'nothing publishes ' + view.topic + ' right now'
    : 'waiting for frames from ' + view.topic;
}

function renderCameras(cameras) {
  cameraTopics = cameras.topics || [];
  if (views === null) {
    views = [];
    loadViews(cameras.default).forEach((v) => addView(v.topic, v.wide));
  }

  const signature = JSON.stringify(cameraTopics.map((e) => [e.topic, e.kind, e.publishers === 0]));
  const changed = signature !== topicsSignature;
  if (changed) topicsSignature = signature;
  views.forEach((view) => {
    if (changed || view.select.options.length === 0) fillTopicMenu(view, false);
    updateView(view);
  });

  const published = cameraTopics.filter((e) => e.publishers > 0).length;
  $('camera-sub').textContent = cameraTopics.length
    ? '· ' + published + ' of ' + cameraTopics.length + ' image topics publishing'
    : '· no image topics on the board';
}

$('camera-add').onclick = () => {
  if (views === null || views.length >= MAX_VIEWS) return;
  // Prefer a publishing topic that no view shows yet.
  const shown = new Set(views.map((v) => v.topic));
  const next = cameraTopics.find((e) => e.publishers > 0 && !shown.has(e.topic))
    || cameraTopics.find((e) => !shown.has(e.topic))
    || cameraTopics[0];
  addView(next ? next.topic : (lastState ? lastState.cameras.default : '/stereo/preview/compressed'), false);
  saveViews();
};

// The RB3's own IMU, barometer and magnetometer. The sensors themselves are on the sensor
// core and owe nothing to the flight controller, but they are PUBLISHED by the rb3_control
// node, so they stop with it. Hence "not streaming" here whenever drone-control is down,
// rather than the board's own motion.
// Position is deliberately explicit about where each number came from. The flight
// controller supplies height only - Betaflight has no X/Y without GPS, and none is fitted -
// so X/Y can only come from the stereo estimator publishing odometry.
function renderPosition(position) {
  position = position || {};
  const odom = position.odom, vo = position.vo;

  const set = (id, value, digits) => {
    const cell = $(id);
    if (cell) cell.textContent = (value === null || value === undefined) ? '--' : fmt(value, digits);
  };

  // The server picks the position (tracks.displayed_position): odometry wins, otherwise the
  // control loop's fused IMU + visual estimate on /vo/status. The path below is recorded from
  // the very same value, so these numbers are always its newest point.
  const shown = position.shown || {};
  const pos = Array.isArray(shown.xyz) ? shown.xyz : null;
  set('pos-x', pos ? pos[0] : null, 2);
  set('pos-y', pos ? pos[1] : null, 2);
  set('pos-z', pos ? pos[2] : (position.fc_altitude_m ?? null), 2);
  const posSource = $('pos-source');
  if (posSource) {
    posSource.classList.toggle('dead', !pos);
    posSource.textContent = shown.source === 'odom'
      ? ('odom' + (shown.frame ? ' · frame ' + shown.frame : ''))
      : shown.source === 'vo' ? 'control loop, fused IMU + visual' + (shown.mode ? ' · ' + shown.mode : '')
      : 'no X/Y — nothing publishes odometry yet (z is the FC height)';
  }

  const voVelocity = vo && Array.isArray(vo.velocity) ? vo.velocity : null;
  const velocity = odom ? odom.velocity
    : voVelocity || (vo ? [vo.vx, vo.vy, null] : null);
  set('vel-x', velocity ? velocity[0] : null, 2);
  set('vel-y', velocity ? velocity[1] : null, 2);
  set('vel-z', velocity ? velocity[2] : null, 2);
  const velSource = $('vel-source');
  if (velSource) {
    velSource.classList.toggle('dead', !velocity);
    velSource.textContent = odom ? 'odom twist'
      : voVelocity ? 'control loop, fused · visual vx ' + fmt(vo.vx, 2) + ', vy ' + fmt(vo.vy, 2)
      : vo ? 'stereo estimator (vx, vy only)'
      : 'no velocity source';
  }

  set('vo-tracked', vo ? vo.tracked : null, 0);
  set('vo-depth', vo && vo.depth_m !== null ? vo.depth_m : null, 2);
  set('vo-fps', vo ? vo.fps : null, 1);
  const voSource = $('vo-source');
  if (voSource) {
    voSource.classList.toggle('dead', !vo);
    voSource.textContent = vo
      ? ((vo.estimator_ms ? vo.estimator_ms + ' ms/frame · ' : '') +
         (typeof vo.depth_valid === 'number' ? 'depth valid ' + Math.round(vo.depth_valid * 100) + '% · ' : '') +
         (vo.mode || ''))
      : 'estimator not running (no VO node publishing /vo)';
  }

  set('pos-alt-fc', position.fc_altitude_m, 2);
  set('pos-alt-baro', position.baro_altitude_m, 2);
}

function renderSensors(flight, link) {
  const axes = (prefix, values, digits) => {
    ['x', 'y', 'z'].forEach((axis, i) => {
      const cell = $(prefix + '-' + axis);
      if (cell) cell.textContent = values ? fmt(values[i], digits) : '--';
    });
  };
  axes('accel', flight.accel, 2);
  axes('gyro', flight.gyro, 3);

  $('baro-p').textContent = fmt(flight.pressure_hpa, 2);
  $('baro-h').textContent = fmt(flight.baro_altitude_m, 2);

  // Update rate the panel measures itself (no dependency on /drone/status).
  const rates = (link && link.rates) || {};
  const label = (topic) => (rates[topic] ? rates[topic].toFixed(1) + ' Hz' : 'no data');
  const setStream = (id, text) => {
    const cell = $(id);
    if (!cell) return;
    cell.textContent = text;
    cell.classList.toggle('dead', text === 'no data');
  };
  setStream('stream-accel', label('imu'));
  setStream('stream-gyro', label('imu'));
  setStream('stream-pressure', rates.pressure ? label('pressure') : label('baro_altitude'));
}

function renderUnits(units) {
  const signature = JSON.stringify(units);
  if (signature === unitsSignature) return;
  unitsSignature = signature;
  const body = $('units');
  body.innerHTML = '';
  units.forEach((unit) => {
    const tr = document.createElement('tr');
    const stateClass = ['active', 'inactive', 'failed', 'activating'].includes(unit.active)
      ? unit.active : 'inactive';
    tr.innerHTML =
      '<td>' + unit.name + '<span class="desc">' + unit.description + '</span></td>' +
      '<td><span class="state ' + stateClass + '">' + unit.active +
        (unit.sub && unit.sub !== unit.active ? ' / ' + unit.sub : '') + '</span></td>' +
      '<td>' + (unit.enabled || '-') + '</td>';

    const actions = document.createElement('td');
    actions.className = 'right';
    const buttons = unit.controllable
      ? [['start', 'start'], ['stop', 'stop'], ['restart', 'restart']]
      : [['restart', 'restart']];
    buttons.forEach(([label, action]) => {
      const button = document.createElement('button');
      button.textContent = label;
      if (action === 'stop') button.className = 'danger';
      button.onclick = () => act(unit.name, action, button, unit.warning);
      if (action === 'start' && unit.active === 'active') button.disabled = true;
      if (action === 'stop' && unit.active !== 'active') button.disabled = true;
      actions.appendChild(button);
    });
    const journal = document.createElement('button');
    journal.textContent = 'log';
    journal.onclick = () => showJournal(unit.name);
    actions.appendChild(journal);
    tr.appendChild(actions);
    body.appendChild(tr);
  });
}

function renderEvents(events) {
  const signature = events.length ? (events.length + ':' + events[0].t) : '0';
  if (signature === eventsSignature) return;
  eventsSignature = signature;
  const list = $('events');
  list.innerHTML = events.map((event) => {
    const when = new Date(event.t * 1000).toLocaleTimeString([], {hour12: false}) +
      ' ' + new Date(event.t * 1000).toLocaleDateString([], {day: '2-digit', month: '2-digit'});
    return '<li class="' + event.level + '"><time>' + when + '</time><span>' +
      escapeHtml(event.message) + '</span></li>';
  }).join('');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function act(unit, action, button, warning) {
  if (action === 'stop' && !confirm('Stop ' + unit + ' on the drone?')) return;
  if (warning && action !== 'stop' && !confirm(action + ' ' + unit + ': ' + warning)) return;
  const hint = $('action-hint');
  hint.textContent = action + ' ' + unit + '...';
  button.disabled = true;
  try {
    const response = await fetch('/api/unit/' + unit + '/' + action, {method: 'POST'});
    const data = await response.json();
    hint.textContent = data.ok ? (action + ' ' + unit + ': ok')
                               : (action + ' ' + unit + ' failed: ' + (data.err || data.error || '?'));
  } catch (error) {
    hint.textContent = 'request failed: ' + error;
  }
  setTimeout(() => { button.disabled = false; }, 1200);
}

async function showJournal(unit) {
  $('modal').classList.add('open');
  $('modal-title').textContent = 'journalctl -u ' + unit;
  $('modal-body').textContent = 'loading...';
  try {
    const response = await fetch('/api/journal/' + unit);
    const data = await response.json();
    $('modal-body').textContent = data.text || '(empty)';
  } catch (error) {
    $('modal-body').textContent = String(error);
  }
}

// --- plots: small enough that a charting library would be more code than this
function drawPlots(plots, windowSeconds) {
  // minSpan keeps autoscaling from magnifying quantisation into a square wave: Betaflight
  // reports attitude in 0.1 deg steps and the battery in 0.01 V, so a resting signal that
  // flips between two adjacent codes would otherwise fill the whole chart.
  const span = windowSeconds || 120;
  draw($('plot-alt'), [[plots.altitude, '#4da3ff'], [plots.baro_altitude, '#3ecf8e']], 1.0, span);
  draw($('plot-att'), [[plots.roll, '#4da3ff'], [plots.pitch, '#3ecf8e']], 4.0, span);
  draw($('plot-batt'), [[plots.battery, '#f0a93b']], 1.0, span);
}

// Each series brings its own x (seconds ago, newest at 0): they are sampled at very
// different rates, so they cannot share one index axis.
function draw(canvas, series, minSpan, windowSeconds) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const present = series.filter(([s]) => s && s.y && s.y.length > 1);
  if (!present.length) {
    context.fillStyle = '#8b93a7';
    context.font = '12px system-ui';
    context.fillText('no data', 10, height / 2);
    return;
  }

  // x is always the full window (seconds ago, newest at 0). Scaling to the oldest sample
  // present instead would make a half-full buffer fill the width and then compress as it
  // filled, rather than the trace sliding leftwards at a constant speed.
  const xMin = -Math.abs(windowSeconds || 120);
  let min = Infinity, max = -Infinity;
  present.forEach(([s]) => {
    s.y.forEach((value) => {
      if (value < min) min = value;
      if (value > max) max = value;
    });
  });
  if (!isFinite(min)) { min = 0; max = 1; }
  const span = minSpan || 0;
  if (max - min < span) {
    const middle = (max + min) / 2;
    min = middle - span / 2;
    max = middle + span / 2;
  }
  const pad = (max - min) * 0.12;
  min -= pad; max += pad;

  const left = 40, right = 6, top = 8, bottom = 14;
  const plotW = width - left - right, plotH = height - top - bottom;
  const sx = (value) => left + (plotW * (value - xMin)) / Math.max(-xMin, 1e-6);
  const clampX = (value) => Math.max(value, xMin);
  const sy = (value) => top + plotH - (plotH * (value - min)) / (max - min);

  context.strokeStyle = '#2a2f3a';
  context.fillStyle = '#8b93a7';
  context.font = '10px ui-monospace, monospace';
  context.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const value = min + ((max - min) * i) / 2;
    const y = Math.round(sy(value)) + 0.5;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(width - right, y);
    context.stroke();
    context.fillText(value.toFixed(Math.abs(value) < 10 ? 2 : 1), 2, y + 3);
  }

  present.forEach(([s, color]) => {
    context.strokeStyle = color;
    context.lineWidth = 1.6;
    context.beginPath();
    s.y.forEach((value, index) => {
      const px = sx(clampX(s.x[index])), py = sy(value);
      if (index === 0) context.moveTo(px, py); else context.lineTo(px, py);
    });
    context.stroke();
  });
}

// --- wiring
$('modal-close').onclick = () => $('modal').classList.remove('open');
$('modal').onclick = (event) => { if (event.target === $('modal')) $('modal').classList.remove('open'); };
$('reboot').onclick = async () => {
  if (!confirm('Reboot the drone board now?')) return;
  $('action-hint').textContent = 'rebooting...';
  await fetch('/api/reboot', {method: 'POST'});
};

let reconnecting = false;

function connect() {
  const source = new EventSource('/api/stream');
  source.onopen = () => {
    // The panel came back: an MJPEG stream from before keeps showing its last frame
    // instead of failing, so reopen them all.
    if (reconnecting && views) views.forEach(startStream);
    reconnecting = false;
  };
  source.onmessage = (event) => {
    document.body.classList.remove('stale');
    try { render(JSON.parse(event.data)); } catch (error) { /* ignore a torn frame */ }
  };
  source.onerror = () => {
    document.body.classList.add('stale');
    source.close();
    reconnecting = true;
    setTimeout(connect, 2000);        // the server may be restarting
  };
}
connect();
window.addEventListener('resize', () => {
  if (lastState) drawPlots(lastState.plots, lastState.window_seconds);
  drawTrack();
});

// --- position track: the path from above and in 3D -------------------------------------
// tracks.py records one track per flight run on the server. The page follows the live one
// (fetching only the samples it does not have yet, when the count in the state grows) or
// shows an earlier run picked from the list. Samples: [t, x, y, z, roll, pitch, yaw, mode].
const T = 0, X = 1, Y = 2, Z = 3, ROLL = 4, PITCH = 5, YAW = 6;
const VIEW3D = {azimuth: -0.7, elevation: 0.55, zoom: 1};
const track = {
  follow: true,           // showing the live run
  liveId: null,
  id: null,               // the track on screen
  source: null,
  samples: [],
  loading: false,
  view: {...VIEW3D},
  view2d: {zoom: 1, cx: null, cy: null},   // cx/cy null = centred on the whole path
};

function updateTrack(status) {
  status = status || {live: null, samples: 0};
  if (status.live !== track.liveId) {
    track.liveId = status.live;
    refreshTrackList();
  }
  if (!track.follow) return;
  if (track.id !== status.live) {
    track.id = status.live;
    track.samples = [];
    track.source = null;
    drawTrack();
  }
  if (track.id && status.samples > track.samples.length && !track.loading) {
    loadTrack(track.id, track.samples.length);
  }
  if (!track.id) setTrackInfo();
}

async function loadTrack(id, from) {
  track.loading = true;
  try {
    const response = await fetch('/api/tracks/' + id + '?from=' + from);
    if (!response.ok) return;
    const data = await response.json();
    if (track.id !== id) return;                  // switched while this was loading
    if (data.from === track.samples.length) track.samples.push(...data.samples);
    track.source = data.source;
    drawTrack();
  } catch (error) {
    /* the next state tick retries */
  } finally {
    track.loading = false;
  }
}

async function refreshTrackList() {
  try {
    const data = await (await fetch('/api/tracks')).json();
    const select = $('track-select');
    const chosen = track.follow ? 'live' : track.id;
    select.innerHTML = '<option value="live">live run' +
      (data.live ? ' (' + trackLabel(data.live) + ')' : ' (waiting for a position)') + '</option>' +
      data.tracks.filter((t) => t.id !== data.live).map((t) =>
        '<option value="' + t.id + '">' + trackLabel(t.id) + ' · ' + duration(t.seconds) +
        ' · ' + t.samples + ' pts</option>').join('');
    select.value = [...select.options].some((o) => o.value === chosen) ? chosen : 'live';
  } catch (error) {
    /* list stays as it was */
  }
}

function trackLabel(id) {          // 20261015-140312 -> 15.10 14:03:12
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(id || '');
  return m ? m[3] + '.' + m[2] + ' ' + m[4] + ':' + m[5] + ':' + m[6] : id;
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  return seconds >= 60 ? Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's' : seconds + 's';
}

function setTrackInfo() {
  const info = $('track-info');
  const samples = track.samples;
  if (!samples.length) {
    info.textContent = track.follow
      ? 'no position yet: the path starts when the control loop publishes /vo/status'
      : 'empty track';
    return;
  }
  let distance = 0;
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    distance += Math.hypot(b[X] - a[X], b[Y] - a[Y], b[Z] - a[Z]);
  }
  const last = samples[samples.length - 1];
  info.textContent = (track.follow ? 'live · ' : 'replay · ') + duration(last[T]) + ' · ' +
    samples.length + ' samples · ' + distance.toFixed(1) + ' m flown' +
    (track.source ? ' · from ' + (track.source === 'odom' ? '/drone/odom' : '/vo/status') : '') +
    (last[YAW] === null ? ' · no attitude from the FC, drone drawn unrotated' : '');
}

$('track-select').onchange = (event) => {
  const value = event.target.value;
  track.follow = value === 'live';
  track.id = track.follow ? track.liveId : value;
  track.samples = [];
  track.source = null;
  drawTrack();
  if (track.id) loadTrack(track.id, 0);
};
$('track-select').onfocus = refreshTrackList;
$('track-new').onclick = async () => {
  await fetch('/api/tracks/split', {method: 'POST'});
  track.follow = true;
  $('track-select').value = 'live';
};

function drawTrack() {
  setTrackInfo();
  drawTrack2D($('track-2d'), track.samples, track.view2d);
  drawTrack3D($('track-3d'), track.samples, track.view);
  drawCameraMotion();
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return {context, width, height};
}

function noTrack(context, height, text) {
  context.fillStyle = '#8b93a7';
  context.font = '12px system-ui';
  context.fillText(text, 12, height / 2);
}

// A "nice" grid step (1, 2 or 5 x 10^n) giving about `lines` lines across `span`.
function niceStep(span, lines) {
  const raw = Math.max(span, 1e-6) / lines;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const unit = raw / power;
  return (unit < 1.5 ? 1 : unit < 3.5 ? 2 : unit < 7.5 ? 5 : 10) * power;
}

// Height as colour: blue (lowest) through green to orange (highest).
function heightColour(z, zMin, zMax) {
  const t = zMax > zMin ? (z - zMin) / (zMax - zMin) : 0.5;
  return 'hsl(' + Math.round(210 - 180 * t) + ', 80%, 60%)';
}

function bounds(samples) {
  const b = {x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity, z0: Infinity, z1: -Infinity};
  samples.forEach((s) => {
    b.x0 = Math.min(b.x0, s[X]); b.x1 = Math.max(b.x1, s[X]);
    b.y0 = Math.min(b.y0, s[Y]); b.y1 = Math.max(b.y1, s[Y]);
    b.z0 = Math.min(b.z0, s[Z]); b.z1 = Math.max(b.z1, s[Z]);
  });
  return b;
}

function heightLegend(context, width, b) {
  context.font = '10px ui-monospace, monospace';
  context.textAlign = 'right';
  context.fillStyle = heightColour(b.z1, b.z0, b.z1);
  context.fillText('z max ' + b.z1.toFixed(2) + ' m', width - 8, 14);
  context.fillStyle = heightColour(b.z0, b.z0, b.z1);
  context.fillText('z min ' + b.z0.toFixed(2) + ' m', width - 8, 26);
  context.textAlign = 'left';
}

function drawTrack2D(canvas, samples, view) {
  const {context, width, height} = prepareCanvas(canvas);
  canvas.map = null;
  if (!samples.length) return noTrack(context, height, 'no position yet');
  const b = bounds(samples);

  // Fitted: equal scale on both axes, at least 2 m across, so a hover does not fill the view
  // with noise. Zoom multiplies that scale; a pan replaces the centre (null = follow the fit).
  const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, 2) * 1.2;
  const cx = view.cx ?? (b.x0 + b.x1) / 2, cy = view.cy ?? (b.y0 + b.y1) / 2;
  const scale = Math.min(width, height) / span * view.zoom;
  canvas.map = {cx, cy, scale, width, height};
  const px = (x) => width / 2 + (x - cx) * scale;
  const py = (y) => height / 2 - (y - cy) * scale;          // y up on screen

  // grid, labelled in metres
  const step = niceStep(Math.max(width, height) / scale, 8);
  context.lineWidth = 1;
  context.font = '10px ui-monospace, monospace';
  context.fillStyle = '#8b93a7';
  const left = cx - width / 2 / scale, right = cx + width / 2 / scale;
  const bottom = cy - height / 2 / scale, top = cy + height / 2 / scale;
  for (let x = Math.ceil(left / step) * step; x <= right; x += step) {
    context.strokeStyle = Math.abs(x) < step / 2 ? '#3a4150' : '#2a2f3a';
    context.beginPath(); context.moveTo(px(x), 0); context.lineTo(px(x), height); context.stroke();
    context.fillText(+x.toFixed(3) + '', px(x) + 3, height - 4);
  }
  for (let y = Math.ceil(bottom / step) * step; y <= top; y += step) {
    context.strokeStyle = Math.abs(y) < step / 2 ? '#3a4150' : '#2a2f3a';
    context.beginPath(); context.moveTo(0, py(y)); context.lineTo(width, py(y)); context.stroke();
    context.fillText(+y.toFixed(3) + '', 3, py(y) - 3);
  }
  context.fillText('x →', width - 28, height - 16);
  context.fillText('y ↑', 6, 14);

  // the path: a straight line between consecutive positions, coloured by height, and a dot
  // on every recorded position
  context.lineWidth = 1.5;
  context.lineCap = 'round';
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], c = samples[i];
    context.strokeStyle = heightColour((a[Z] + c[Z]) / 2, b.z0, b.z1);
    context.beginPath(); context.moveTo(px(a[X]), py(a[Y])); context.lineTo(px(c[X]), py(c[Y]));
    context.stroke();
  }
  drawDots(context, samples.map((s) => [px(s[X]), py(s[Y])]), samples, b);
  if (view.zoom !== 1 || view.cx !== null) {
    context.fillStyle = '#8b93a7';
    context.font = '10px ui-monospace, monospace';
    context.fillText('zoom ' + view.zoom.toFixed(1) + '× · double-click or "fit" to see all', 30, 14);
  }

  // start: hollow circle
  const first = samples[0];
  context.strokeStyle = '#3ecf8e';
  context.lineWidth = 2;
  context.beginPath(); context.arc(px(first[X]), py(first[Y]), 5, 0, 2 * Math.PI); context.stroke();

  // the drone now: a rectangle turned by yaw, front marked with a triangle
  const last = samples[samples.length - 1];
  const yaw = last[YAW] === null ? 0 : last[YAW];
  context.save();
  context.translate(px(last[X]), py(last[Y]));
  context.rotate(-yaw);                                     // screen y points down
  context.fillStyle = 'rgba(77, 163, 255, .25)';
  context.strokeStyle = '#4da3ff';
  context.lineWidth = 2;
  context.fillRect(-12, -8, 24, 16);
  context.strokeRect(-12, -8, 24, 16);
  context.fillStyle = '#4da3ff';
  context.beginPath(); context.moveTo(16, 0); context.lineTo(9, -5); context.lineTo(9, 5);
  context.closePath(); context.fill();
  context.restore();

  heightLegend(context, width, b);
}

// Body-to-world rotation from roll, pitch, yaw (Z-Y-X, as Betaflight reports attitude).
function rotate(point, roll, pitch, yaw) {
  let [x, y, z] = point;
  [y, z] = [y * Math.cos(roll) - z * Math.sin(roll), y * Math.sin(roll) + z * Math.cos(roll)];
  [x, z] = [x * Math.cos(pitch) + z * Math.sin(pitch), -x * Math.sin(pitch) + z * Math.cos(pitch)];
  [x, y] = [x * Math.cos(yaw) - y * Math.sin(yaw), x * Math.sin(yaw) + y * Math.cos(yaw)];
  return [x, y, z];
}

function drawTrack3D(canvas, samples, view) {
  const {context, width, height} = prepareCanvas(canvas);
  if (!samples.length) return noTrack(context, height, 'no position yet');
  const b = bounds(samples);
  const ground = Math.min(0, b.z0);
  const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, cz = (ground + b.z1) / 2;
  const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1 - ground, 2) * 1.1;
  const scale = Math.min(width, height) / span / 1.5 * view.zoom;

  // Orthographic: turn around z by the azimuth, then tilt by the elevation
  // (0 = from the side, 90 degrees = from above).
  const ca = Math.cos(view.azimuth), sa = Math.sin(view.azimuth);
  const ce = Math.cos(view.elevation), se = Math.sin(view.elevation);
  const project = (x, y, z) => {
    x -= cx; y -= cy; z -= cz;
    const x1 = x * ca - y * sa, y1 = x * sa + y * ca;
    return [width / 2 + x1 * scale, height / 2 - (z * ce + y1 * se) * scale];
  };
  const line = (a, c, colour, widthPx, dash) => {
    const p = project(...a), q = project(...c);
    context.strokeStyle = colour;
    context.lineWidth = widthPx;
    context.setLineDash(dash || []);
    context.beginPath(); context.moveTo(p[0], p[1]); context.lineTo(q[0], q[1]); context.stroke();
    context.setLineDash([]);
  };

  // ground grid at z = 0 (or below the lowest point)
  const step = niceStep(span, 8);
  const gx0 = Math.floor((cx - span / 2) / step) * step, gx1 = Math.ceil((cx + span / 2) / step) * step;
  const gy0 = Math.floor((cy - span / 2) / step) * step, gy1 = Math.ceil((cy + span / 2) / step) * step;
  for (let x = gx0; x <= gx1 + 1e-9; x += step) line([x, gy0, ground], [x, gy1, ground], '#2a2f3a', 1);
  for (let y = gy0; y <= gy1 + 1e-9; y += step) line([gx0, y, ground], [gx1, y, ground], '#2a2f3a', 1);

  // axes from the origin, one grid step long
  const axes = [[[step, 0, 0], '#ff5f56', 'x'], [[0, step, 0], '#3ecf8e', 'y'], [[0, 0, step], '#4da3ff', 'z']];
  context.font = '11px ui-monospace, monospace';
  axes.forEach(([end, colour, name]) => {
    line([0, 0, ground], [end[0], end[1], ground + end[2]], colour, 2);
    const p = project(end[0] * 1.15, end[1] * 1.15, ground + end[2] * 1.15);
    context.fillStyle = colour;
    context.fillText(name, p[0] - 3, p[1] + 4);
  });
  context.fillStyle = '#8b93a7';
  context.font = '10px ui-monospace, monospace';
  context.fillText('grid ' + +step.toFixed(3) + ' m', 8, height - 6);

  // shadow on the ground, then the path coloured by height
  context.lineCap = 'round';
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], c = samples[i];
    line([a[X], a[Y], ground], [c[X], c[Y], ground], 'rgba(139, 147, 167, .35)', 1);
  }
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], c = samples[i];
    line([a[X], a[Y], a[Z]], [c[X], c[Y], c[Z]], heightColour((a[Z] + c[Z]) / 2, b.z0, b.z1), 1.5);
  }
  drawDots(context, samples.map((s) => project(s[X], s[Y], s[Z])), samples, b);

  // start marker
  const first = samples[0];
  const start = project(first[X], first[Y], first[Z]);
  context.strokeStyle = '#3ecf8e';
  context.lineWidth = 2;
  context.beginPath(); context.arc(start[0], start[1], 5, 0, 2 * Math.PI); context.stroke();

  // the drone now: a flat box turned by roll, pitch and yaw, with a line down to the ground
  const last = samples[samples.length - 1];
  const at = [last[X], last[Y], last[Z]];
  line([at[0], at[1], ground], at, 'rgba(77, 163, 255, .6)', 1, [4, 4]);
  const size = 24 / scale;                                  // about 24 px long whatever the zoom
  const [l, w, h] = [size / 2, size / 3, size / 10];
  const corners = [[l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h],
                   [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h]].map((corner) => {
    const r = rotate(corner, last[ROLL] || 0, last[PITCH] || 0, last[YAW] || 0);
    return project(at[0] + r[0], at[1] + r[1], at[2] + r[2]);
  });
  context.fillStyle = 'rgba(77, 163, 255, .25)';
  context.beginPath();
  [0, 1, 2, 3].forEach((i, n) => (n ? context.lineTo : context.moveTo).call(context, ...corners[i]));
  context.closePath(); context.fill();
  context.strokeStyle = '#4da3ff';
  context.lineWidth = 1.5;
  [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    .forEach(([i, j]) => {
      context.beginPath(); context.moveTo(...corners[i]); context.lineTo(...corners[j]); context.stroke();
    });
  const nose = rotate([l * 1.6, 0, 0], last[ROLL] || 0, last[PITCH] || 0, last[YAW] || 0);
  line(at, [at[0] + nose[0], at[1] + nose[1], at[2] + nose[2]], '#4da3ff', 2);

  heightLegend(context, width, b);
}

(() => {
  const canvas = $('track-3d');
  let drag = null;
  canvas.addEventListener('pointerdown', (event) => {
    drag = {x: event.clientX, y: event.clientY};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drag) return;
    track.view.azimuth += (event.clientX - drag.x) * 0.01;
    track.view.elevation = Math.max(0, Math.min(Math.PI / 2,
      track.view.elevation + (event.clientY - drag.y) * 0.01));
    drag = {x: event.clientX, y: event.clientY};
    drawTrack3D(canvas, track.samples, track.view);
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    track.view.zoom = Math.max(0.2, Math.min(20, track.view.zoom * Math.exp(-event.deltaY * 0.001)));
    drawTrack3D(canvas, track.samples, track.view);
  }, {passive: false});
  canvas.addEventListener('dblclick', () => {
    track.view = {...VIEW3D};
    drawTrack3D(canvas, track.samples, track.view);
  });
})();

// --- position and velocity from the camera, over the run -----------------------------
// The camera's own estimate, saved in the same track samples as the path (columns 8-13),
// so these plots follow the path dropdown: the live run, or a replayed one.
const CAM_X = 8, CAM_VX = 11;
const AXIS_COLOURS = ['#ff5f56', '#3ecf8e', '#4da3ff'];

function drawCameraMotion() {
  const samples = track.samples;
  drawRunPlot($('camera-position'), samples, CAM_X, 0.5,
    'no camera position yet: it appears once the visual estimator has usable tracks');
  drawRunPlot($('camera-velocity'), samples, CAM_VX, 0.2,
    'no camera velocity yet: it appears once the visual estimator has usable tracks');
}

// Three columns (x, y, z) against run time. A null value (no camera measurement on that axis)
// breaks the line instead of being drawn as zero.
function drawRunPlot(canvas, samples, first, minSpan, emptyText) {
  const {context, width, height} = prepareCanvas(canvas);
  const value = (s, axis) => {
    const v = s[first + axis];
    return typeof v === 'number' && isFinite(v) ? v : null;
  };
  let min = Infinity, max = -Infinity;
  samples.forEach((s) => {
    for (let axis = 0; axis < 3; axis++) {
      const v = value(s, axis);
      if (v !== null) { min = Math.min(min, v); max = Math.max(max, v); }
    }
  });
  if (!isFinite(min)) return noTrack(context, height, emptyText);
  if (max - min < minSpan) {
    const middle = (max + min) / 2;
    min = middle - minSpan / 2;
    max = middle + minSpan / 2;
  }
  const pad = (max - min) * 0.1;
  min -= pad; max += pad;

  const t0 = samples[0][T], t1 = Math.max(samples[samples.length - 1][T], t0 + 1);
  const left = 44, right = 8, top = 10, bottom = 20;
  const plotW = width - left - right, plotH = height - top - bottom;
  const sx = (t) => left + plotW * (t - t0) / (t1 - t0);
  const sy = (v) => top + plotH - plotH * (v - min) / (max - min);

  context.lineWidth = 1;
  context.font = '10px ui-monospace, monospace';
  context.fillStyle = '#8b93a7';
  const vStep = niceStep(max - min, 6);
  for (let v = Math.ceil(min / vStep) * vStep; v <= max; v += vStep) {
    context.strokeStyle = Math.abs(v) < vStep / 2 ? '#3a4150' : '#2a2f3a';
    context.beginPath(); context.moveTo(left, sy(v)); context.lineTo(width - right, sy(v)); context.stroke();
    context.textAlign = 'right';
    context.fillText(+v.toFixed(3) + '', left - 4, sy(v) + 3);
  }
  const tStep = niceStep(t1 - t0, 8);
  context.textAlign = 'center';
  for (let t = Math.ceil(t0 / tStep) * tStep; t <= t1; t += tStep) {
    context.strokeStyle = '#2a2f3a';
    context.beginPath(); context.moveTo(sx(t), top); context.lineTo(sx(t), top + plotH); context.stroke();
    context.fillText(+t.toFixed(1) + '', sx(t), height - 6);
  }
  context.textAlign = 'left';

  context.lineWidth = 1.6;
  context.lineJoin = 'round';
  for (let axis = 0; axis < 3; axis++) {
    context.strokeStyle = AXIS_COLOURS[axis];
    context.beginPath();
    let drawing = false;
    samples.forEach((s) => {
      const v = value(s, axis);
      if (v === null) { drawing = false; return; }
      if (drawing) context.lineTo(sx(s[T]), sy(v));
      else context.moveTo(sx(s[T]), sy(v));
      drawing = true;
    });
    context.stroke();
  }

  // latest values, top right
  const last = samples[samples.length - 1];
  context.font = '10px ui-monospace, monospace';
  context.textAlign = 'right';
  ['x', 'y', 'z'].forEach((name, axis) => {
    const v = value(last, axis);
    context.fillStyle = AXIS_COLOURS[axis];
    context.fillText(name + ' ' + (v === null ? '--' : v.toFixed(2)), width - right - 4 - (2 - axis) * 62, top + 10);
  });
  context.textAlign = 'left';
}

// A dot on every recorded position (each frame the path was sampled at). Skipped while they
// would be closer than ~3 px on average, where they only blur the line into a thick band.
function drawDots(context, points, samples, b) {
  if (points.length < 2) return;
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  if (length / (points.length - 1) < 3) return;
  points.forEach(([x, y], i) => {
    context.fillStyle = heightColour(samples[i][Z], b.z0, b.z1);
    context.beginPath(); context.arc(x, y, 2.2, 0, 2 * Math.PI); context.fill();
  });
}

// Zoom: the -, + and fit buttons work on both views; the wheel zooms the view under the
// mouse (the 2D one around the cursor); dragging pans the 2D view and turns the 3D one.
function zoomTrack(factor) {
  const view = track.view2d;
  const map = $('track-2d').map;
  if (map && view.cx === null) { view.cx = map.cx; view.cy = map.cy; }
  view.zoom = Math.max(0.2, Math.min(200, view.zoom * factor));
  track.view.zoom = Math.max(0.2, Math.min(200, track.view.zoom * factor));
  drawTrack();
}
function fitTrack() {
  track.view2d = {zoom: 1, cx: null, cy: null};
  track.view = {...VIEW3D};
  drawTrack();
}
$('track-zoom-in').onclick = () => zoomTrack(1.5);
$('track-zoom-out').onclick = () => zoomTrack(1 / 1.5);
$('track-fit').onclick = fitTrack;

(() => {
  const canvas = $('track-2d');
  const toWorld = (event) => {
    const map = canvas.map, rect = canvas.getBoundingClientRect();
    return [map.cx + (event.clientX - rect.left - map.width / 2) / map.scale,
            map.cy - (event.clientY - rect.top - map.height / 2) / map.scale];
  };
  let drag = null;
  canvas.addEventListener('pointerdown', (event) => {
    if (!canvas.map) return;
    drag = {x: event.clientX, y: event.clientY};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drag || !canvas.map) return;
    const map = canvas.map, view = track.view2d;
    view.cx = map.cx - (event.clientX - drag.x) / map.scale;
    view.cy = map.cy + (event.clientY - drag.y) / map.scale;
    drag = {x: event.clientX, y: event.clientY};
    drawTrack2D(canvas, track.samples, view);
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('wheel', (event) => {
    if (!canvas.map) return;
    event.preventDefault();
    const view = track.view2d, map = canvas.map;
    const factor = Math.exp(-event.deltaY * 0.0015);
    const zoom = Math.max(0.2, Math.min(200, view.zoom * factor));
    const [wx, wy] = toWorld(event);            // keep the point under the cursor in place
    const ratio = view.zoom / zoom;
    view.cx = wx + (map.cx - wx) * ratio;
    view.cy = wy + (map.cy - wy) * ratio;
    view.zoom = zoom;
    drawTrack2D(canvas, track.samples, view);
  }, {passive: false});
  canvas.addEventListener('dblclick', () => {
    track.view2d = {zoom: 1, cx: null, cy: null};
    drawTrack2D(canvas, track.samples, track.view2d);
  });
})();

refreshTrackList();
drawTrack();
