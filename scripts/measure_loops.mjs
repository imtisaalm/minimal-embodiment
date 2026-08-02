#!/usr/bin/env node
// scripts/measure_loops.mjs
//
// Reproduction script for §6.3 of the accompanying paper. Fires N reps of
// each named haptic effect, each named audio tone, and a haptic-baseline
// (no-motor) noise-floor probe, in randomized order interleaved across reps.
// Saves raw JSONL plus per-table summary CSVs.
//
// Haptic echoes are measured fires: each trial carries its own pre-fire
// control window, and the full statistics (AC peak/RMS for both windows,
// snr = rms / floor_rms, and the bridge's `felt` verdict) are pulled from
// /haptic/echo after the wait_echo confirms completion. felt_rate on the
// baseline row is the false-positive rate. The audio loop protocol is
// unchanged; note the firmware's mic chain includes a 200 Hz high-pass
// filter, so ambient floors are lower than raw-mic datasets.
//
// Usage:
//   export US_BRIDGE_TOKEN="..."          # bearer token the bridge accepts
//   export US_BRIDGE_HOST="https://..."   # public HTTPS endpoint of the bridge
//   node scripts/measure_loops.mjs
//
// Optional environment:
//   US_OUTPUT_DIR     Directory for output files (default ./data/).
//   US_REPS           Number of reps per effect/tone (default 30).
//   US_INTER_MS       Inter-trial gap in ms (default 2000).

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const TOKEN = process.env.US_BRIDGE_TOKEN;
if (!TOKEN) {
  process.stderr.write(
    'error: US_BRIDGE_TOKEN is not set.\n' +
    '       Set it to the bearer token the bridge is using:\n' +
    '         export US_BRIDGE_TOKEN="..."\n',
  );
  process.exit(1);
}
const HOST = process.env.US_BRIDGE_HOST;
if (!HOST) {
  process.stderr.write(
    'error: US_BRIDGE_HOST is not set.\n' +
    '       Set it to the public HTTPS endpoint of the bridge, e.g.:\n' +
    '         export US_BRIDGE_HOST="https://your-tunnel-host.example.com"\n',
  );
  process.exit(1);
}
const OUTPUT_DIR = path.resolve(ROOT, process.env.US_OUTPUT_DIR || 'data');
const N_REPS = Number.parseInt(process.env.US_REPS || '30', 10);
const INTER_MS = Number.parseInt(process.env.US_INTER_MS || '2000', 10);

const RAW_PATH = path.join(OUTPUT_DIR, 'loops_raw.jsonl');
const HAPTIC_CSV = path.join(OUTPUT_DIR, 'loops_haptic.csv');
const AUDIO_CSV = path.join(OUTPUT_DIR, 'loops_audio.csv');

// Mirrors HAPTIC_EFFECTS in src/http-bridge.ts.
const HAPTIC_EFFECTS = [
  'tap', 'soft_tap', 'double_tap', 'triple_tap',
  'buzz', 'hum', 'long_buzz',
  'heartbeat', 'knock', 'hello', 'alert',
];
// Mirrors SOUNDS in src/http-bridge.ts.
const AUDIO_TONES = [
  'hello', 'hi', 'hey', 'ping', 'hum',
  'call', 'alert', 'chirp', 'low', 'long_hum',
];

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function buildTrials() {
  // Per rep, randomize the 22 effect-types (haptic 11 + audio 10 + baseline 1).
  // This ensures every effect-type occurs once per rep, while exposure-
  // ordering is independent per rep — drift effects (motor heating, room
  // ambient, motor microposition) bias all effects equally.
  const trials = [];
  for (let rep = 0; rep < N_REPS; rep++) {
    const repTrials = [
      ...HAPTIC_EFFECTS.map((name) => ({ rep, kind: 'haptic', name })),
      ...AUDIO_TONES.map((name) => ({ rep, kind: 'audio', name })),
      { rep, kind: 'baseline', name: 'baseline' },
    ];
    shuffle(repTrials);
    trials.push(...repTrials);
  }
  return trials;
}

async function fireTrial(trial) {
  let url;
  if (trial.kind === 'haptic') {
    url = `${HOST}/haptic?effect=${encodeURIComponent(trial.name)}` +
      `&wait_echo=true&token=${TOKEN}`;
  } else if (trial.kind === 'audio') {
    url = `${HOST}/beep?name=${encodeURIComponent(trial.name)}` +
      `&wait_echo=true&token=${TOKEN}`;
  } else {
    url = `${HOST}/haptic/baseline?wait_echo=true&token=${TOKEN}`;
  }

  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
  const j = await r.json();

  const out = {
    ts: new Date().toISOString(),
    ...trial,
    echo_waited: j.echo_waited === true,
  };
  const room = j.room || {};
  // Room context for outlier forensics — a passing truck or a bump on
  // the desk shows up here instead of staying a mystery in the stats.
  if (Number.isFinite(room.noise_db)) out.ambient_db = room.noise_db;

  if (trial.kind === 'haptic') {
    out.effect_id = j.effect_id;
    await pullFullEcho(out, j.effect_id);
  } else if (trial.kind === 'audio') {
    out.frequency = j.frequency;
    out.duration_ms = j.duration_ms;
    out.ambient_db = room.noise_db;
    const e = room.recent_beep_echo;
    // Attribution guard: recent AND matching tone. The frequency+duration
    // pair is unique per named tone, so a stale echo from the previous
    // trial can't masquerade as this one's.
    if (
      e && e.age_seconds <= 1 &&
      e.frequency === j.frequency && e.duration_ms === j.duration_ms
    ) {
      out.echo_db = e.noise_db;
    }
  } else {
    await pullFullEcho(out, 0);
  }

  return out;
}

// The room snapshot deliberately carries only the percept (peak_g / snr /
// felt); the full instrument statistics live on /haptic/echo. Pull them
// from there after the wait_echo confirmed the measurement completed.
//
// Attribution guard: recent (≤1 s — the response can be built just after
// the age bucket rolls over) AND matching effect id, so a late echo from
// an earlier trial can never be credited to this one. The id match is
// what makes the relaxed age window safe: trials are 2 s apart, and
// within any 1 s window only one trial's id can match.
async function pullFullEcho(out, expectedEffectId) {
  const r = await fetch(`${HOST}/haptic/echo?token=${TOKEN}`);
  if (r.status !== 200) return; // 204 = no echo yet
  const j = await r.json();
  const e = j.echo;
  if (!e || j.age_seconds > 1 || e.effect_id !== expectedEffectId) return;
  out.peak_g = e.peak_g;
  out.rms = e.rms;
  out.floor_peak = e.floor_peak;
  out.floor_rms = e.floor_rms;
  out.snr = e.snr;
  out.felt = e.felt === true;
}

function quantile(sorted, q) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

function summarize(values) {
  const v = [...values].filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return {
    n: v.length,
    median: quantile(v, 0.5),
    iqr_lo: quantile(v, 0.25),
    iqr_hi: quantile(v, 0.75),
    min: v.length > 0 ? v[0] : NaN,
    max: v.length > 0 ? v[v.length - 1] : NaN,
  };
}

function fmt(x, dp = 3) {
  return Number.isFinite(x) ? x.toFixed(dp) : '';
}

async function writeSummaries(results) {
  // Haptic + baseline → loops_haptic.csv. Explicit statistic prefixes;
  // snr is the headline, felt_rate on the baseline row is the
  // false-positive rate.
  const hapticRows = [
    'kind,name,effect_id,n,' +
      'peak_median,peak_iqr_lo,peak_iqr_hi,' +
      'rms_median,floor_rms_median,' +
      'snr_median,snr_iqr_lo,snr_iqr_hi,snr_min,snr_max,' +
      'felt_rate',
  ];
  const hapticOrder = [...HAPTIC_EFFECTS, 'baseline'];
  for (const name of hapticOrder) {
    const kind = name === 'baseline' ? 'baseline' : 'haptic';
    const rows = results.filter((r) => r.kind === kind && r.name === name);
    const peak = summarize(rows.map((r) => r.peak_g));
    const rms = summarize(rows.map((r) => r.rms));
    const floor = summarize(rows.map((r) => r.floor_rms));
    const snr = summarize(rows.map((r) => r.snr));
    const feltN = rows.filter((r) => r.felt === true).length;
    const feltRate = snr.n > 0 ? feltN / snr.n : NaN;
    const eid = rows.find((r) => Number.isFinite(r.effect_id))?.effect_id ?? 0;
    hapticRows.push(
      `${kind},${name},${eid},${snr.n},` +
        `${fmt(peak.median)},${fmt(peak.iqr_lo)},${fmt(peak.iqr_hi)},` +
        `${fmt(rms.median)},${fmt(floor.median)},` +
        `${fmt(snr.median, 2)},${fmt(snr.iqr_lo, 2)},${fmt(snr.iqr_hi, 2)},` +
        `${fmt(snr.min, 2)},${fmt(snr.max, 2)},` +
        `${fmt(feltRate, 3)}`,
    );
  }
  await fs.writeFile(HAPTIC_CSV, hapticRows.join('\n') + '\n');

  // Audio → loops_audio.csv (separate echo + ambient summaries)
  const audioRows = [
    'name,frequency_hz,duration_ms,n,' +
      'echo_median,echo_iqr_lo,echo_iqr_hi,echo_min,echo_max,' +
      'ambient_median,ambient_iqr_lo,ambient_iqr_hi,ambient_min,ambient_max',
  ];
  for (const name of AUDIO_TONES) {
    const rows = results.filter((r) => r.kind === 'audio' && r.name === name);
    if (rows.length === 0) continue;
    const echoes = rows.map((r) => r.echo_db);
    const ambients = rows.map((r) => r.ambient_db);
    const e = summarize(echoes);
    const a = summarize(ambients);
    const freq = rows[0]?.frequency ?? '';
    const dur = rows[0]?.duration_ms ?? '';
    audioRows.push(
      `${name},${freq},${dur},${e.n},` +
        `${fmt(e.median, 1)},${fmt(e.iqr_lo, 1)},${fmt(e.iqr_hi, 1)},` +
        `${fmt(e.min, 1)},${fmt(e.max, 1)},` +
        `${fmt(a.median, 1)},${fmt(a.iqr_lo, 1)},${fmt(a.iqr_hi, 1)},` +
        `${fmt(a.min, 1)},${fmt(a.max, 1)}`,
    );
  }
  await fs.writeFile(AUDIO_CSV, audioRows.join('\n') + '\n');
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.writeFile(RAW_PATH, ''); // truncate; resume not supported on purpose

  const trials = buildTrials();
  const total = trials.length;
  // Measured haptic trials spend ~2.2 s inside the measurement (floor +
  // signal windows + echo round-trip); audio trials are unchanged.
  const estMin = Math.round((total * (INTER_MS / 1000 + 2.2)) / 60);
  process.stderr.write(
    `Running ${total} trials (${HAPTIC_EFFECTS.length} haptic + ` +
      `${AUDIO_TONES.length} audio + 1 baseline) × ${N_REPS} reps. ` +
      `Inter-trial ${INTER_MS}ms. ETA ~${estMin} min.\n`,
  );
  const t0 = Date.now();

  const results = [];
  for (let i = 0; i < trials.length; i++) {
    const t = trials[i];
    try {
      const result = { trial_index: i, ...(await fireTrial(t)) };
      results.push(result);
      await fs.appendFile(RAW_PATH, JSON.stringify(result) + '\n');
    } catch (err) {
      process.stderr.write(
        `[trial ${i}] ${t.kind} ${t.name} FAILED: ${err.message}\n`,
      );
    }

    // Progress beat at the end of each rep (every 22 trials).
    const trialsPerRep = HAPTIC_EFFECTS.length + AUDIO_TONES.length + 1;
    if ((i + 1) % trialsPerRep === 0) {
      const rep = (i + 1) / trialsPerRep;
      const pct = (((i + 1) / total) * 100).toFixed(1);
      const elapsedSec = ((Date.now() - t0) / 1000).toFixed(0);
      process.stderr.write(
        `[rep ${rep}/${N_REPS} done — ${pct}%, ${elapsedSec}s elapsed]\n`,
      );
    }

    if (i < trials.length - 1) {
      await new Promise((r) => setTimeout(r, INTER_MS));
    }
  }

  await writeSummaries(results);

  const elapsedSec = ((Date.now() - t0) / 1000).toFixed(0);
  process.stderr.write(
    `\nDone in ${elapsedSec}s. ${results.length}/${total} trials succeeded.\n` +
      `  raw → ${path.relative(ROOT, RAW_PATH)}\n` +
      `  haptic + baseline → ${path.relative(ROOT, HAPTIC_CSV)}\n` +
      `  audio → ${path.relative(ROOT, AUDIO_CSV)}\n`,
  );
}

main().catch((e) => {
  process.stderr.write(String(e?.stack || e) + '\n');
  process.exit(1);
});
