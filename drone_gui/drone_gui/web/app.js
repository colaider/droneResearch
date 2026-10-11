'use strict';

// One SSE connection carries the whole state object at 5 Hz. EventSource reconnects by
// itself, so a restart of the server (or of the drone) just resumes the page.
let lastState = null;
let cameraStarted = false;
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

  renderSensors(flight);

  // --- camera
  const sub = $('camera-sub');
  const stereo = state.camera.stereo;
  sub.textContent = state.camera.age === null ? ''
    : '· ' + state.camera.frames + ' frames' + (stereo && stereo.fps ? ', ' + fmt(stereo.fps, 1) + ' fps' : '');
  const haveFrames = state.camera.age !== null && state.camera.age < 5;
  $('camera').classList.toggle('live', haveFrames);
  $('camera-placeholder').style.display = haveFrames ? 'none' : 'block';
  if (haveFrames && !cameraStarted) {
    // Start the MJPEG stream only once there is something to show, so the <img> does not
    // sit on a dead connection from page load.
    $('camera').src = '/camera/stream?t=' + Date.now();
    cameraStarted = true;
  }
  if (!haveFrames && cameraStarted && state.camera.age === null) {
    $('camera').removeAttribute('src');
    cameraStarted = false;
  }

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

// The RB3's own IMU, barometer and magnetometer. These come from the sensor core and are
// independent of the flight controller, so they keep reading even when the FC link is down -
// which is exactly when you want to know whether the board itself is still healthy.
function renderSensors(flight) {
  const axes = (prefix, values, digits) => {
    ['x', 'y', 'z'].forEach((axis, i) => {
      const cell = $(prefix + '-' + axis);
      if (cell) cell.textContent = values ? fmt(values[i], digits) : '--';
    });
  };
  axes('gyro', flight.gyro, 3);
  axes('accel', flight.accel, 2);
  axes('mag', flight.mag_ut, 1);

  $('baro-p').textContent = fmt(flight.pressure_hpa, 2);
  $('baro-h').textContent = fmt(flight.baro_altitude_m, 2);
  // Field magnitude is the quick sanity check on the magnetometer: Earth's field is
  // 25-65 uT, and a much larger reading means something nearby is magnetising it.
  const mag = flight.mag_ut;
  $('mag-norm').textContent = mag
    ? fmt(Math.hypot(mag[0], mag[1], mag[2]), 1) : '--';

  const streams = flight.sensors || {};
  ['gyro', 'accel', 'mag', 'pressure'].forEach((name) => {
    const cell = $('stream-' + name);
    if (!cell) return;
    const info = streams[name];
    if (!info) {
      cell.textContent = 'not streaming';
      cell.classList.add('dead');
      return;
    }
    cell.classList.toggle('dead', !info.alive);
    cell.textContent = (info.alive ? '' : 'STALLED · ') + info.rate_hz + ' Hz · '
      + info.samples.toLocaleString() + ' samples'
      + (info.restarts ? ' · ' + info.restarts + ' restarts' : '');
  });
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

function connect() {
  const source = new EventSource('/api/stream');
  source.onmessage = (event) => {
    document.body.classList.remove('stale');
    try { render(JSON.parse(event.data)); } catch (error) { /* ignore a torn frame */ }
  };
  source.onerror = () => {
    document.body.classList.add('stale');
    source.close();
    setTimeout(connect, 2000);        // the server may be restarting
  };
}
connect();
window.addEventListener('resize', () => {
  if (lastState) drawPlots(lastState.plots, lastState.window_seconds);
});
