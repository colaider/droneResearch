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

  set('pos-x', odom ? odom.position[0] : null, 2);
  set('pos-y', odom ? odom.position[1] : null, 2);
  set('pos-z', odom ? odom.position[2] : (position.fc_altitude_m ?? null), 2);
  const posSource = $('pos-source');
  if (posSource) {
    const missing = !odom;
    posSource.classList.toggle('dead', missing);
    posSource.textContent = odom
      ? ('odom' + (odom.frame ? ' · frame ' + odom.frame : ''))
      : 'no X/Y — nothing publishes odometry yet (z is the FC height)';
  }

  // The estimator reports velocity, not position: /vo/status carries vx, vy and yaw rate.
  const velocity = odom ? odom.velocity
    : (vo ? [vo.vx, vo.vy, null] : null);
  set('vel-x', velocity ? velocity[0] : null, 2);
  set('vel-y', velocity ? velocity[1] : null, 2);
  set('vel-z', velocity ? velocity[2] : null, 2);
  const velSource = $('vel-source');
  if (velSource) {
    velSource.classList.toggle('dead', !velocity);
    velSource.textContent = odom ? 'odom twist'
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
      ? ((vo.estimator_ms ? vo.estimator_ms + ' ms/frame · ' : '') + (vo.mode || ''))
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
      button.onclick = () => act(unit.name, action, button);
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

async function act(unit, action, button) {
  if (action === 'stop' && !confirm('Stop ' + unit + ' on the drone?')) return;
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
});
