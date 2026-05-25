const API = '';

let currentLiga = 'acb';

const selLocal       = document.getElementById('sel-local');
const selVisit       = document.getElementById('sel-visit');
const btnSwap        = document.getElementById('btn-swap');
const btnPredict     = document.getElementById('btn-predict');
const resultCard     = document.getElementById('result-card');
const resultContent  = document.getElementById('result-content');
const errorBox       = document.getElementById('error-box');
const loader         = document.getElementById('loader');
const headerTitle    = document.getElementById('header-title');
const headerSubtitle = document.getElementById('header-subtitle');
const footerText     = document.getElementById('footer-text');
const bblNotice      = document.getElementById('bbl-notice');

// ── League selector ───────────────────────────────────────────────
document.querySelectorAll('.league-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const liga = btn.dataset.liga;
    if (liga === currentLiga) return;
    currentLiga = liga;

    document.querySelectorAll('.league-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    hideAll();
    resetSelects();
    updateLeagueUI(liga);
    loadTeams(liga);
  });
});

function updateLeagueUI(liga) {
  if (liga === 'bbl') {
    headerTitle.textContent = 'Basketball Predictor';
    headerSubtitle.textContent = 'Predicción de partidos · Bundesliga (BBL) · Modelo ML (AUC 0.66)';
    footerText.textContent = 'Datos: 5 temporadas (2020-21 → 2024-25) · eurobasket.com';
  } else {
    headerTitle.textContent = 'Basketball Predictor';
    headerSubtitle.textContent = 'Predicción de partidos · Liga ACB (España) · Modelo ML (69.9% accuracy)';
    footerText.textContent = 'Datos: 1.663 partidos · 5 temporadas (2020-21 → 2024-25) · api2.acb.com';
  }
}

// ── Cargar equipos ────────────────────────────────────────────────
async function loadTeams(liga) {
  liga = liga || currentLiga;
  resetSelects();
  bblNotice.classList.add('hidden');

  try {
    const res = await fetch(`${API}/teams?liga=${liga}&_t=${Date.now()}`);
    if (res.status === 503) {
      bblNotice.classList.remove('hidden');
      return;
    }
    const teams = await res.json();
    if (!Array.isArray(teams) || teams.length === 0) {
      bblNotice.classList.remove('hidden');
      return;
    }
    [selLocal, selVisit].forEach(sel => {
      teams.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.nombre;
        opt.textContent = t.nombre;
        sel.appendChild(opt);
      });
    });
  } catch {
    showError('No se pudo conectar con la API. ¿Está corriendo el servidor?');
  }
}

function resetSelects() {
  [selLocal, selVisit].forEach(sel => {
    sel.innerHTML = '<option value="">— Elige equipo —</option>';
  });
  btnPredict.disabled = true;
}

// ── Habilitar botón solo cuando hay dos equipos distintos ─────────
function checkReady() {
  const ok = selLocal.value && selVisit.value && selLocal.value !== selVisit.value;
  btnPredict.disabled = !ok;
}
selLocal.addEventListener('change', checkReady);
selVisit.addEventListener('change', checkReady);

// ── Intercambiar equipos ──────────────────────────────────────────
btnSwap.addEventListener('click', () => {
  const tmp = selLocal.value;
  selLocal.value  = selVisit.value;
  selVisit.value  = tmp;
  checkReady();
});

// ── Predecir ─────────────────────────────────────────────────────
btnPredict.addEventListener('click', async () => {
  hideAll();
  loader.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        equipo_local:     selLocal.value,
        equipo_visitante: selVisit.value,
        liga:             currentLiga,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Error ${res.status}`);
    }

    const data = await res.json();
    loader.classList.add('hidden');
    renderResult(data);
  } catch (e) {
    loader.classList.add('hidden');
    showError(e.message);
  }
});

// ── Renderizar resultado ──────────────────────────────────────────
function renderResult(d) {
  const pL  = (d.prob_local   * 100).toFixed(1);
  const pV  = (d.prob_visitante * 100).toFixed(1);
  const isLocalWin = d.prediccion === 'local';
  const winner = isLocalWin ? d.equipo_local : d.equipo_visitante;
  const confClass = `conf-${d.confianza}`;
  const verdictClass = isLocalWin ? 'win-local' : 'win-visit';

  const f = d.features_usadas;
  const ligaLabel = currentLiga === 'bbl' ? 'Diferencial puntos' : 'Valoración ACB promedio';

  resultContent.innerHTML = `
    <!-- Cabecera del partido -->
    <div class="match-header">
      <span class="team-name local">${d.equipo_local}</span>
      <span class="vs-badge">VS</span>
      <span class="team-name visit">${d.equipo_visitante}</span>
    </div>

    <!-- Barras de probabilidad -->
    <div class="prob-section">
      <div class="prob-label-row">
        <span class="prob-label">🏠 ${d.equipo_local}</span>
        <span class="prob-label">✈️ ${d.equipo_visitante}</span>
      </div>
      <div class="prob-label-row">
        <span class="prob-value local">${pL}%</span>
        <span class="prob-value visit">${pV}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-local"  style="width:${pL}%"></div>
        <div class="prob-bar-visit"  style="width:${pV}%"></div>
      </div>
    </div>

    <!-- Veredicto -->
    <div class="verdict">
      <span class="verdict-text ${verdictClass}">
        ${isLocalWin ? '🏠' : '✈️'} Gana <strong>${winner}</strong>
        <span class="confidence-pill ${confClass}">${d.confianza}</span>
      </span>
    </div>

    <!-- Stats clave -->
    <p class="stats-title">📊 Estadísticas recientes (base del modelo)</p>
    <div class="stats-grid">
      ${statRow('% Victorias recientes',
          pct(f.home_win_rate_reciente), pct(f.away_win_rate_reciente))}
      ${statRow('Puntos promedio',
          f.home_pts_promedio.toFixed(1), f.away_pts_promedio.toFixed(1))}
      ${statRow(ligaLabel,
          f.home_val_promedio.toFixed(1), f.away_val_promedio.toFixed(1))}
      ${statRow('H2H enfrentamientos',
          f.h2h_enfrentamientos, f.h2h_enfrentamientos)}
      ${statRow('H2H tasa victoria local',
          pct(f.h2h_tasa_local), '—')}
    </div>
  `;

  resultCard.classList.remove('hidden');
}

function statRow(label, lv, vv) {
  return `
    <div class="stat-row">
      <span class="stat-name">${label}</span>
      <span class="stat-vals">
        <span class="lv">${lv}</span>
        <span style="color:#cbd5e1">|</span>
        <span class="vv">${vv}</span>
      </span>
    </div>`;
}

function pct(val) {
  return (val * 100).toFixed(0) + '%';
}

function showError(msg) {
  errorBox.textContent = '⚠️ ' + msg;
  errorBox.classList.remove('hidden');
}

function hideAll() {
  resultCard.classList.add('hidden');
  errorBox.classList.add('hidden');
  loader.classList.add('hidden');
}

// ── Init ─────────────────────────────────────────────────────────
updateLeagueUI('acb');
loadTeams('acb');
