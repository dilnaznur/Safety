(() => {
  'use strict';

  const API_BASE = `${location.protocol}//${location.host}`;
  const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

  const el = {
    camera: document.getElementById('camera'),
    mode: document.getElementById('mode'),
    applyMode: document.getElementById('applyMode'),
    video: document.getElementById('video'),
    status: document.getElementById('status'),
    alerts: document.getElementById('alerts'),
    people: document.getElementById('statPeople'),
    ppe: document.getElementById('statPPE'),
    fire: document.getElementById('statFire'),
    active: document.getElementById('statAlerts'),
    spills: document.getElementById('statSpills'),
    falls: document.getElementById('statFalls'),
  };

  let ws = null;
  let reconnectTimer = null;
  let selectedCamera = '';

  function setClass(target, cls) {
    target.classList.remove('good', 'warn', 'bad');
    target.classList.add(cls);
  }

  function updateStats(stats) {
    const people = Number(stats.people_count || 0);
    const ppe = Number(stats.ppe_compliance || 0);
    const fire = String(stats.fire_risk || 'Safe');
    const active = Number(stats.active_alerts || 0);
    const spills = Number(stats.spill_count || 0);
    const falls = Number(stats.fall_count || 0);

    el.people.textContent = String(people);
    el.ppe.textContent = `${ppe.toFixed(1)}%`;
    el.fire.textContent = fire;
    el.active.textContent = String(active);
    el.spills.textContent = String(spills);
    el.falls.textContent = String(falls);

    setClass(el.people, people > 0 ? 'good' : 'warn');
    setClass(el.ppe, ppe >= 80 ? 'good' : ppe >= 50 ? 'warn' : 'bad');
    setClass(el.fire, fire === 'Safe' ? 'good' : fire === 'High' ? 'warn' : 'bad');
    setClass(el.active, active > 0 ? 'bad' : 'good');
    setClass(el.spills, spills > 0 ? 'bad' : 'good');
    setClass(el.falls, falls > 0 ? 'bad' : 'good');
  }

  function addAlerts(alerts) {
    if (!Array.isArray(alerts) || alerts.length === 0) {
      return;
    }

    alerts.slice().reverse().forEach((alert) => {
      const div = document.createElement('div');
      const sev = String(alert.severity || 'info');
      div.className = `alert ${sev === 'critical' ? 'alert-critical' : sev === 'high' ? 'alert-high' : ''}`;
      div.innerHTML = `<strong>${String(alert.type || 'ALERT')}</strong><br>${String(alert.message || '')}`;
      el.alerts.prepend(div);
      while (el.alerts.children.length > 40) {
        el.alerts.removeChild(el.alerts.lastChild);
      }
    });
  }

  function sendCameraSelection() {
    if (ws && ws.readyState === WebSocket.OPEN && selectedCamera) {
      ws.send(JSON.stringify({ camera_id: selectedCamera }));
    }
  }

  function connectWs() {
    if (ws) {
      try { ws.close(); } catch (_) {}
    }

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      el.status.textContent = 'Connected';
      sendCameraSelection();
    };

    ws.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (_) {
        return;
      }

      if (payload.type !== 'frame') {
        return;
      }

      if (payload.frame) {
        el.video.src = `data:image/jpeg;base64,${payload.frame}`;
      }

      if (payload.stats) {
        updateStats(payload.stats);
      }

      if (payload.alerts) {
        addAlerts(payload.alerts);
      }

      const camName = payload.camera_name || payload.camera_id || 'camera';
      if (payload.connected) {
        el.status.textContent = `${camName}: connected`;
      } else {
        el.status.textContent = `${camName}: ${payload.error || 'disconnected'}`;
      }
    };

    ws.onclose = () => {
      el.status.textContent = 'Disconnected. Reconnecting...';
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connectWs();
        }, 2000);
      }
    };

    ws.onerror = () => {
      el.status.textContent = 'WebSocket error';
    };
  }

  async function loadCameras() {
    const res = await fetch(`${API_BASE}/api/cameras`);
    const data = await res.json();
    const cams = data.cameras || [];

    el.camera.innerHTML = '';
    cams.forEach((cam) => {
      const option = document.createElement('option');
      option.value = cam.id;
      option.textContent = `${cam.name} (${cam.id})`;
      el.camera.appendChild(option);
    });

    if (cams.length > 0) {
      selectedCamera = cams[0].id;
      el.camera.value = selectedCamera;
    }
  }

  el.camera.addEventListener('change', () => {
    selectedCamera = el.camera.value;
    sendCameraSelection();
  });

  el.applyMode.addEventListener('click', async () => {
    if (!selectedCamera) return;
    const mode = el.mode.value;
    const res = await fetch(`${API_BASE}/api/cameras/${encodeURIComponent(selectedCamera)}/mode/${encodeURIComponent(mode)}`, { method: 'POST' });
    if (res.ok) {
      el.status.textContent = `Mode set to ${mode} for ${selectedCamera}`;
    } else {
      el.status.textContent = 'Failed to set mode';
    }
  });

  (async () => {
    try {
      await loadCameras();
      connectWs();
    } catch (err) {
      el.status.textContent = `Startup error: ${err.message}`;
    }
  })();
})();
