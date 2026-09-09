/**
 * Minion Airways — Seat Hold Worker
 * ----------------------------------
 * A tiny Cloudflare Worker that acts as the single source of truth for seat
 * availability across every user looking at a given flight+date. Backed by
 * Cloudflare KV (the "simple DB"). Seats a user selects are HELD for 10
 * minutes, not permanently booked — booking isn't live on the site yet — so
 * this only prevents two people from picking the same seat at the same time.
 *
 * Device recognition: each hold is tagged with BOTH a sessionId (from
 * sessionStorage, resets per-tab) and the requester's IP (via Cloudflare's
 * built-in `cf-connecting-ip` header — no setup needed). A hold is treated
 * as "yours" if EITHER matches, so reloading the page or opening a new tab
 * on the same device/network still recognizes your own held seats instead
 * of showing them as taken by someone else.
 *
 * KV key:   seats:<flightNo>:<date>            e.g. seats:MA328:2026-08-07
 * KV value: JSON { seats: { "12A": { status: "held"|"taken", sessionId, ip, expiresAt } } }
 *
 * Endpoints (all CORS-enabled, JSON in/out):
 *   GET  /seats?flight=MA328&date=2026-08-07&sessionId=...
 *        -> { seats: { "12A": {status, expiresAt, mine}, ... } }
 *        `mine` is true if this seat's hold matches the caller's sessionId or IP.
 *        Expired holds are lazily cleaned up on read.
 *
 *   POST /hold   { flight, date, seatId, sessionId }
 *        -> { ok:true, expiresAt } on success
 *        -> { ok:false, reason:"taken"|"held" } 409 if unavailable to this session/IP
 *        Holding a seat you already hold (by session OR IP) just refreshes the timer.
 *
 *   POST /release { flight, date, seatId, sessionId }
 *        -> { ok:true } always (idempotent) — only releases if this session/IP held it.
 *
 * Deploy:
 *   1. npm i -g wrangler   (if you don't have it)
 *   2. wrangler kv namespace create SEATS_KV
 *      -> paste the returned id into wrangler.toml
 *   3. wrangler deploy
 *   4. Put the resulting workers.dev URL into WORKER_URL at the top of the
 *      <script> block in booking.html.
 */

const HOLD_MS = 10 * 60 * 1000; // 10 minutes

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

function kvKey(flight, date) {
  return `seats:${flight}:${date}`;
}

function clientIp(req) {
  return req.headers.get("cf-connecting-ip") || "unknown";
}

// Same device = same sessionId (per-tab) OR same IP (per-network/device).
function isMine(seat, sessionId, ip) {
  return seat.sessionId === sessionId || (seat.ip && seat.ip === ip);
}

async function readRecord(env, flight, date) {
  const raw = await env.SEATS_KV.get(kvKey(flight, date));
  const record = raw ? JSON.parse(raw) : { seats: {} };
  const now = Date.now();
  let changed = false;
  for (const [seatId, s] of Object.entries(record.seats)) {
    if (s.status === "held" && s.expiresAt <= now) {
      delete record.seats[seatId];
      changed = true;
    }
  }
  return { record, changed };
}

async function writeRecord(env, flight, date, record) {
  await env.SEATS_KV.put(kvKey(flight, date), JSON.stringify(record), {
    expirationTtl: 60 * 60 * 24,
  });
}

async function handleGetSeats(req, env) {
  const url = new URL(req.url);
  const flight = url.searchParams.get("flight");
  const date = url.searchParams.get("date") || "no-date";
  const sessionId = url.searchParams.get("sessionId") || "";
  const ip = clientIp(req);
  if (!flight) return json({ error: "flight is required" }, 400);

  const { record, changed } = await readRecord(env, flight, date);
  if (changed) await writeRecord(env, flight, date, record);

  const seats = {};
  for (const [seatId, s] of Object.entries(record.seats)) {
    seats[seatId] = {
      status: s.status,
      expiresAt: s.expiresAt,
      mine: s.status === "held" ? isMine(s, sessionId, ip) : false,
    };
  }
  return json({ seats });
}

async function handleHold(req, env) {
  const body = await req.json().catch(() => ({}));
  const { flight, date, seatId, sessionId } = body;
  if (!flight || !seatId || !sessionId) {
    return json({ ok: false, reason: "missing fields" }, 400);
  }
  const ip = clientIp(req);
  const d = date || "no-date";
  const { record } = await readRecord(env, flight, d);
  const existing = record.seats[seatId];

  if (existing && existing.status === "taken") {
    return json({ ok: false, reason: "taken" }, 409);
  }
  if (existing && existing.status === "held" && !isMine(existing, sessionId, ip)) {
    return json({ ok: false, reason: "held" }, 409);
  }

  const expiresAt = Date.now() + HOLD_MS;
  record.seats[seatId] = { status: "held", sessionId, ip, expiresAt };
  await writeRecord(env, flight, d, record);
  return json({ ok: true, expiresAt });
}

async function handleRelease(req, env) {
  const body = await req.json().catch(() => ({}));
  const { flight, date, seatId, sessionId } = body;
  if (!flight || !seatId || !sessionId) {
    return json({ ok: false, reason: "missing fields" }, 400);
  }
  const ip = clientIp(req);
  const d = date || "no-date";
  const { record } = await readRecord(env, flight, d);
  const existing = record.seats[seatId];
  if (existing && existing.status === "held" && isMine(existing, sessionId, ip)) {
    delete record.seats[seatId];
    await writeRecord(env, flight, d, record);
  }
  return json({ ok: true });
}

export default {
  async fetch(req, env) {
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }
    const url = new URL(req.url);
    try {
      if (req.method === "GET" && url.pathname === "/seats") return await handleGetSeats(req, env);
      if (req.method === "POST" && url.pathname === "/hold") return await handleHold(req, env);
      if (req.method === "POST" && url.pathname === "/release") return await handleRelease(req, env);
      return json({ error: "not found" }, 404);
    } catch (err) {
      return json({ error: String(err) }, 500);
    }
  },
};

