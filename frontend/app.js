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
  const ligaLabel = currentLiga === 'bbl' ? 'Dif. puntos prom.' : 'Valoración ACB prom.';

  // Apply winner-colored top border
  resultCard.classList.remove('win-local', 'win-visit');
  resultCard.classList.add(verdictClass);

  resultContent.innerHTML = `
    <!-- Cabecera del partido -->
    <div class="match-header">
      <div class="team-name-block ${isLocalWin ? 'winner' : ''}">
        ${isLocalWin ? '<span class="crown">👑</span>' : ''}
        <span class="team-name local">${d.equipo_local}</span>
        <span class="team-role-tag local-tag">🏠 Local</span>
      </div>
      <div class="vs-center">
        <span class="vs-badge">VS</span>
      </div>
      <div class="team-name-block right ${!isLocalWin ? 'winner' : ''}">
        ${!isLocalWin ? '<span class="crown">👑</span>' : ''}
        <span class="team-name visit">${d.equipo_visitante}</span>
        <span class="team-role-tag visit-tag">✈️ Visitante</span>
      </div>
    </div>

    <!-- Probabilidades -->
    <div class="prob-section">
      <div class="prob-numbers-row">
        <span class="prob-value local" id="pval-local">0.0%</span>
        <div class="prob-center-label">Prob. victoria</div>
        <span class="prob-value visit" id="pval-visit">0.0%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-local" id="pbar-local" style="width:0%"></div>
        <div class="prob-bar-visit" id="pbar-visit" style="width:0%"></div>
      </div>
      <div class="prob-teams-row">
        <span class="prob-team-label local">${d.equipo_local}</span>
        <span class="prob-team-label visit">${d.equipo_visitante}</span>
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
    <p class="stats-title">📊 Estadísticas recientes</p>
    <div class="stats-grid">
      ${statBar('% Victorias rec.', f.home_win_rate_reciente, f.away_win_rate_reciente, 'pct')}
      ${statBar('Puntos promedio', f.home_pts_promedio, f.away_pts_promedio, 'raw')}
      ${statBar(ligaLabel, f.home_val_promedio, f.away_val_promedio, 'raw')}
      ${statBar('H2H victoria local', f.h2h_tasa_local, 1 - f.h2h_tasa_local, 'pct')}
    </div>

    <button class="predict-again-btn" onclick="scrollToTop()">↑ Nueva predicción</button>
  `;

  resultCard.classList.remove('hidden');

  // Animate probability counters
  animateCounter('pval-local', parseFloat(pL));
  animateCounter('pval-visit', parseFloat(pV));

  // Animate probability bars (short delay so CSS transition triggers)
  setTimeout(() => {
    const bl = document.getElementById('pbar-local');
    const bv = document.getElementById('pbar-visit');
    if (bl) bl.style.width = pL + '%';
    if (bv) bv.style.width = pV + '%';
  }, 80);
}

// Build a stat row with mini comparison bar
function statBar(label, localVal, visitVal, format) {
  const lNum = Number(localVal) || 0;
  const vNum = Number(visitVal) || 0;

  let lDisplay, vDisplay;
  if (format === 'pct') {
    lDisplay = (lNum * 100).toFixed(0) + '%';
    vDisplay = (vNum * 100).toFixed(0) + '%';
  } else {
    lDisplay = lNum.toFixed(1);
    vDisplay = vNum.toFixed(1);
  }

  const [lPct, vPct] = miniBarWidths(lNum, vNum);
  const localWins = lNum >= vNum;

  return `
    <div class="stat-row">
      <span class="stat-name">${label}</span>
      <div class="stat-values-row">
        <span class="stat-val lv ${localWins ? 'stat-winner' : ''}">${lDisplay}</span>
        <div class="stat-mini-bar-track">
          <div class="stat-mini-bar-local" style="width:${lPct.toFixed(1)}%"></div>
          <div class="stat-mini-bar-visit" style="width:${vPct.toFixed(1)}%"></div>
        </div>
        <span class="stat-val vv ${!localWins ? 'stat-winner' : ''}">${vDisplay}</span>
      </div>
    </div>`;
}

// Normalize two values to [0,100] proportions for the mini bar
function miniBarWidths(localVal, visitVal) {
  const min = Math.min(localVal, visitVal, 0);
  const lAdj = localVal - min;
  const vAdj = visitVal - min;
  const total = lAdj + vAdj;
  if (total <= 0) return [50, 50];
  return [(lAdj / total * 100), (vAdj / total * 100)];
}

// Count-up animation for probability display
function animateCounter(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const duration = 900;
  const startTime = performance.now();
  function frame(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = (eased * target).toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
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
