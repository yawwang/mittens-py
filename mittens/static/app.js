// Mittens Web UI — vanilla JS, no framework

const API = '';  // Same origin
let activeRunId = null;
let ws = null;

// -- API helpers --

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// -- Run management --

async function startRun() {
  const mission = document.getElementById('mission-input').value.trim();
  if (!mission) return;

  const tier = document.getElementById('tier-select').value || undefined;
  const body = { mission, workflow_id: 'autonomous-build' };
  if (tier) body.tier = tier;

  try {
    const run = await api('/api/runs', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    document.getElementById('mission-input').value = '';
    selectRun(run.run_id);
    refreshRunList();
  } catch (e) {
    console.error('Failed to start run:', e);
  }
}

async function cancelRun(runId) {
  try {
    await api(`/api/runs/${runId}`, { method: 'DELETE' });
    refreshRunList();
  } catch (e) {
    console.error('Failed to cancel:', e);
  }
}

// -- Run list --

async function refreshRunList() {
  try {
    const runs = await api('/api/runs');
    const list = document.getElementById('run-list');
    list.innerHTML = runs.map(r => `
      <li class="run-item ${r.id === activeRunId ? 'active' : ''}"
          onclick="selectRun('${r.id}')">
        <div class="mission">${escHtml(r.mission.slice(0, 60))}</div>
        <div class="meta">
          <span class="badge ${statusClass(r.status)}">${r.status}</span>
          ${r.tier} &middot; ${formatTime(r.started_at)}
        </div>
      </li>
    `).join('');
  } catch (e) {
    console.error('Failed to load runs:', e);
  }
}

function statusClass(s) {
  return (s || '').toLowerCase().replace('_', '-');
}

// -- Run detail --

async function selectRun(runId) {
  activeRunId = runId;
  refreshRunList();

  // Close existing WebSocket
  if (ws) { ws.close(); ws = null; }

  const panel = document.getElementById('main-panel');
  panel.innerHTML = '<div class="empty">Loading...</div>';

  try {
    const [run, events, artifacts] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/runs/${runId}/events`),
      api(`/api/runs/${runId}/artifacts`),
    ]);

    panel.innerHTML = `
      <h2>${escHtml(run.mission)}</h2>
      <div style="margin-bottom:16px">
        <span class="badge ${statusClass(run.status)}">${run.status}</span>
        ${run.tier} &middot; ${run.workflow_id}
        ${run.is_active ? `<button onclick="cancelRun('${runId}')" style="margin-left:12px;padding:4px 12px;background:#da3636;color:#fff;border:none;border-radius:4px;cursor:pointer">Cancel</button>` : ''}
      </div>
      <div class="timeline" id="timeline">
        ${events.map(renderEvent).join('')}
      </div>
      <div class="cost-panel" id="cost-panel"></div>
    `;

    // Load cost if available
    loadCost(runId);

    // Connect WebSocket for live events
    if (run.is_active) {
      connectWS(runId);
    }
  } catch (e) {
    panel.innerHTML = `<div class="empty">Error loading run: ${e.message}</div>`;
  }
}

function renderEvent(evt) {
  const cls = (evt.event_type || '').toLowerCase().replace('_', '-');
  const fields = evt.fields || {};
  const fieldHtml = Object.entries(fields)
    .map(([k, v]) => `<strong>${k}:</strong> ${escHtml(String(v))}`)
    .join(' &middot; ');

  return `
    <div class="event ${cls}">
      <span class="type">${evt.event_type}</span>
      <span class="time">${formatTime(evt.timestamp)}</span>
      <div class="fields">${fieldHtml}</div>
    </div>
  `;
}

async function loadCost(runId) {
  try {
    const cost = await api(`/api/runs/${runId}/cost`);
    const panel = document.getElementById('cost-panel');
    if (cost.total_input_tokens !== undefined) {
      let html = '<h3>Cost Breakdown</h3>';
      html += `<div class="cost-row"><span>Input tokens</span><span>${cost.total_input_tokens.toLocaleString()}</span></div>`;
      html += `<div class="cost-row"><span>Output tokens</span><span>${cost.total_output_tokens.toLocaleString()}</span></div>`;
      if (cost.per_talent) {
        html += '<h3 style="margin-top:8px">Per Talent</h3>';
        for (const [tid, data] of Object.entries(cost.per_talent)) {
          html += `<div class="cost-row"><span>${tid}</span><span>${data.input.toLocaleString()} in / ${data.output.toLocaleString()} out</span></div>`;
        }
      }
      panel.innerHTML = html;
    }
  } catch (e) {
    // Cost not available yet
  }
}

// -- WebSocket --

function connectWS(runId) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/runs/${runId}`);

  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      const timeline = document.getElementById('timeline');
      if (timeline) {
        timeline.insertAdjacentHTML('beforeend', renderEvent(data));
        timeline.scrollTop = timeline.scrollHeight;
      }

      // Refresh on completion
      if (data.event_type === 'PROJECT_COMPLETE') {
        setTimeout(() => selectRun(runId), 500);
      }
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    ws = null;
    // Refresh to show final state
    setTimeout(refreshRunList, 1000);
  };
}

// -- Utilities --

function escHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function formatTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

// -- Init --

refreshRunList();
setInterval(refreshRunList, 10000);
