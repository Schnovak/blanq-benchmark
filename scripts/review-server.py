#!/usr/bin/env python3
"""
BlanQ Benchmark — swipe review tool.
Serves on 0.0.0.0:7771 / accessed at blanqdev.izum.ch/rank/

Swipe right = Approve (saves ground truth)
Swipe left  = Flag    (saves to review_later for manual annotation)
Trash button = Delete  (removes from dataset)

Already-processed pages are skipped (deduplication by GT / review_later presence).
Done → commits + pushes to GitHub.
"""
import http.server, json, os, csv, fitz, base64, threading, urllib.request, subprocess, re

REPO     = "/tmp/blanq-benchmark-staging/blanq-benchmark"
MANIFEST = os.path.join(REPO, "dataset", "manifest.csv")
GT_DIR   = os.path.join(REPO, "ground_truth")
RL_DIR   = os.path.join(REPO, "review_later")
DET_FILE = os.path.join(REPO, "results", "blanq", "detections.json")
CACHE    = "/tmp/blanq-cache"
API_URL  = "http://172.25.0.5:8000/process-pdf"
PORT     = 7771

for d in [GT_DIR, RL_DIR, os.path.dirname(DET_FILE), CACHE]:
    os.makedirs(d, exist_ok=True)

# ── data helpers ──────────────────────────────────────────────────────────────

def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))

def page_status(rid):
    if os.path.exists(os.path.join(GT_DIR, f"{rid}.json")):   return "approved"
    if os.path.exists(os.path.join(RL_DIR, f"{rid}.json")):   return "flagged"
    return "unprocessed"

def cache_file(rid):
    return os.path.join(CACHE, f"{rid}.json")

def call_blanq(pdf_path, rid):
    cp = cache_file(rid)
    if os.path.exists(cp):
        return json.load(open(cp))
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        boundary = b"----BlanqBound"
        body = (b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="p.pdf"\r\n'
                b"Content-Type: application/pdf\r\n\r\n"
                + data + b"\r\n--" + boundary + b"--\r\n")
        req = urllib.request.Request(
            API_URL, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        with open(cp, "w") as f:
            json.dump(result, f)
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}

def render_overlay(pdf_path, page_data, scale=1.8):
    """Render PDF page with green rectangle overlay for detected blanks."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height
    sx = page_data["canvasW"] / pw
    sy = page_data["canvasH"] / ph
    for b in page_data.get("blanks", []):
        x, y = b["x"] / sx, b["y"] / sy
        x2, y2 = x + b["width"] / sx, y + b["height"] / sy
        page.draw_rect(fitz.Rect(x, y, x2, y2),
                       color=(0.0, 0.72, 0.36), fill=(0.0, 0.72, 0.36),
                       fill_opacity=0.22, width=1.8)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode()

def render_thumb(pdf_path, scale=1.8):
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode()

def blanks_to_gt(page_data, rid, pdf_path):
    doc = fitz.open(pdf_path)
    rect = doc[0].rect
    pw, ph = rect.width, rect.height
    doc.close()
    sx = page_data["canvasW"] / pw
    sy = page_data["canvasH"] / ph
    blanks = []
    for i, b in enumerate(page_data.get("blanks", []), 1):
        rows = len(b.get("mergedHeights", [1]))
        blanks.append({
            "id": f"b{i:03d}",
            "x": round(b["x"] / sx, 2), "y": round(b["y"] / sy, 2),
            "width": round(b["width"] / sx, 2), "height": round(b["height"] / sy, 2),
            "type": "multi_line" if rows > 1 else "single_line",
            "confidence": round(b.get("confidence", 1.0), 4),
            "rows": rows if rows > 1 else None,
        })
    return {"id": rid, "page_width": round(pw, 2), "page_height": round(ph, 2), "blanks": blanks}

def update_detections(rid, page_data, pdf_path):
    data = json.load(open(DET_FILE)) if os.path.exists(DET_FILE) else \
           {"tool": "blanq", "pages": {}, "system_info": {"source": "blanq-ai-detect"}}
    doc = fitz.open(pdf_path)
    rect = doc[0].rect
    pw, ph = rect.width, rect.height
    doc.close()
    sx = page_data["canvasW"] / pw
    sy = page_data["canvasH"] / ph
    dets = [{
        "x": round(b["x"] / sx, 2), "y": round(b["y"] / sy, 2),
        "width": round(b["width"] / sx, 2), "height": round(b["height"] / sy, 2),
        "type": "multi_line" if len(b.get("mergedHeights", [1])) > 1 else "single_line",
        "confidence": round(b.get("confidence", 1.0), 4),
        "rows": len(b.get("mergedHeights", [1])) if len(b.get("mergedHeights", [1])) > 1 else None,
    } for b in page_data.get("blanks", [])]
    data["pages"][rid] = {"detections": dets, "detection_time_ms": 0, "failed": False}
    with open(DET_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── background prefetch ───────────────────────────────────────────────────────

PREFETCH = {}   # rid → "pending" | "ready" | "error"
LOCK = threading.Lock()

def prefetch_worker(rows):
    for row in rows:
        rid = row["id"]
        if page_status(rid) != "unprocessed":
            with LOCK: PREFETCH[rid] = "ready"
            continue
        if os.path.exists(cache_file(rid)):
            with LOCK: PREFETCH[rid] = "ready"
            continue
        pdf_path = os.path.join(REPO, "dataset", row["file"])
        call_blanq(pdf_path, rid)
        with LOCK: PREFETCH[rid] = "ready"

# ── HTML ──────────────────────────────────────────────────────────────────────

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>BlanQ Rank</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  html,body{height:100%;overflow:hidden;background:#0a0a0a;color:#fff;
    font-family:system-ui,-apple-system,sans-serif;touch-action:none}

  /* ── top bar ── */
  #bar{position:fixed;top:0;left:0;right:0;background:#0a0a0a;
    display:flex;align-items:center;gap:12px;padding:0 16px;z-index:30;
    border-bottom:1px solid #1e1e1e;flex-wrap:wrap}
  #bar-main{display:flex;align-items:center;gap:12px;width:100%;height:52px}
  #all-banner{width:100%;background:#7c3aed;color:#fff;font-size:.72rem;font-weight:700;
    text-align:center;padding:4px 0;letter-spacing:.05em;display:none}
  #progress{flex:1;height:4px;background:#1e1e1e;border-radius:2px;overflow:hidden}
  #progress-fill{height:100%;background:#3b82f6;border-radius:2px;transition:width .3s}
  #counter{font-size:.78rem;color:#666;white-space:nowrap}
  #done-btn{background:#3b82f6;color:#fff;border:none;border-radius:8px;
    padding:7px 14px;font-size:.82rem;font-weight:700;cursor:pointer;white-space:nowrap}
  #done-btn:disabled{opacity:.5;cursor:default}

  /* ── card stack ── */
  #stack{position:fixed;inset:52px 0 80px;display:flex;align-items:center;
    justify-content:center;perspective:800px}
  .card{position:absolute;width:min(420px,94vw);background:#141414;
    border-radius:18px;overflow:hidden;cursor:grab;user-select:none;
    box-shadow:0 8px 32px rgba(0,0,0,.6);transition:transform .08s linear,opacity .08s linear;
    border:2px solid #222}
  .card:active{cursor:grabbing}
  .card-img{width:100%;display:block;max-height:calc(100vh - 200px);object-fit:contain}
  .card-meta{padding:10px 14px;display:flex;align-items:center;gap:8px}
  .card-id{font-size:.72rem;color:#666;flex:1}
  .card-blanks{font-size:.72rem;color:#3b82f6;font-weight:600}
  .card-status{font-size:.68rem;font-weight:700;padding:2px 7px;border-radius:6px}
  .card-status.approved{background:#14532d;color:#4ade80}
  .card-status.flagged{background:#450a0a;color:#f87171}
  .card-loading{display:flex;align-items:center;justify-content:center;
    min-height:300px;color:#444;font-size:.85rem}

  /* swipe feedback overlays */
  .stamp{position:absolute;top:24px;padding:6px 16px;border-radius:10px;
    font-size:1.4rem;font-weight:900;opacity:0;transition:opacity .1s;
    border:3px solid;letter-spacing:.05em;text-transform:uppercase;z-index:10}
  .stamp-yes{left:18px;color:#22c55e;border-color:#22c55e}
  .stamp-no {right:18px;color:#ef4444;border-color:#ef4444}

  /* ── bottom actions ── */
  #actions{position:fixed;bottom:0;left:0;right:0;height:80px;
    display:flex;align-items:center;justify-content:center;gap:20px;
    background:linear-gradient(to top,#0a0a0a 60%,transparent);z-index:20}
  .act{width:54px;height:54px;border:none;border-radius:50%;cursor:pointer;
    display:flex;align-items:center;justify-content:center;font-size:1.4rem;
    transition:transform .12s,filter .12s;box-shadow:0 4px 16px rgba(0,0,0,.4)}
  .act:active{transform:scale(.9)}
  .act-flag  {background:#1e1e1e;color:#ef4444}
  .act-del   {background:#1e1e1e;color:#888;font-size:1.1rem}
  .act-approve{background:#22c55e;color:#fff}

  /* ── result screen ── */
  #result{display:none;position:fixed;inset:0;background:#0a0a0a;
    align-items:center;justify-content:center;flex-direction:column;
    padding:32px;text-align:center;gap:16px}
  #result h2{font-size:1.8rem;color:#22c55e}
  #result p{color:#888;font-size:.9rem;line-height:1.7}
  #result pre{background:#141414;border-radius:10px;padding:14px 16px;
    text-align:left;font-size:.72rem;color:#666;max-height:40vh;overflow-y:auto;width:100%}
  #result a{color:#3b82f6;font-size:.85rem}

  /* ── empty state ── */
  #empty{display:none;position:fixed;inset:0;background:#0a0a0a;
    align-items:center;justify-content:center;flex-direction:column;gap:12px;color:#444}
  #empty p{font-size:.9rem}
  #empty a{color:#3b82f6;font-size:.82rem}
</style>
</head>
<body>

<div id="bar">
  <div id="all-banner">SHOW-ALL MODE — includes already reviewed pages · <a href="./" style="color:#ddd">exit</a></div>
  <div id="bar-main">
    <div id="progress"><div id="progress-fill" style="width:0%"></div></div>
    <span id="counter">— / —</span>
    <button id="done-btn" onclick="commitAll()" disabled>Done</button>
  </div>
</div>

<div id="stack"></div>

<div id="actions">
  <button class="act act-flag"   title="Flag — detection wrong" onclick="swipeCard('left')">✗</button>
  <button class="act act-del"    title="Delete page from dataset" onclick="deleteCard()">🗑</button>
  <button class="act act-approve" title="Approve — detection correct" onclick="swipeCard('right')">✓</button>
</div>

<div id="result">
  <h2>Done!</h2>
  <p id="result-stats"></p>
  <pre id="result-log"></pre>
  <a href="./">Review more</a>
</div>

<div id="empty">
  <p>All unreviewed pages done!</p>
</div>

<script>
const pages = [];        // [{id, source, blank_count}] — loaded from /pages
const decisions = {};    // id → "approve"|"flag"|"delete"
let current = 0;
let cardData = {};       // id → {overlay_b64, thumb_b64, blank_count}
let loadedAhead = new Set();

// ── bootstrap ──────────────────────────────────────────────────────────────
async function init() {
  const params = new URLSearchParams(location.search);
  const showAll = params.has('all');
  if (showAll) {
    const banner = document.getElementById('all-banner');
    banner.style.display = 'block';
    document.getElementById('bar').style.paddingBottom = '0';
  }
  const res = await fetch('pages' + (showAll ? '?all=1' : ''));
  const data = await res.json();
  pages.push(...data.pages);
  updateCounter();
  if (!pages.length) { document.getElementById('empty').style.display = 'flex'; return; }
  document.getElementById('done-btn').disabled = false;
  loadCard(0);
  loadCard(1);  // preload next
}

function updateCounter() {
  const done = Object.keys(decisions).length;
  const total = pages.length;
  document.getElementById('counter').textContent = `${done} / ${total}`;
  document.getElementById('progress-fill').style.width = total ? `${(done/total)*100}%` : '0%';
}

// ── card loading ────────────────────────────────────────────────────────────
async function loadCard(idx) {
  if (idx >= pages.length) return;
  const {id} = pages[idx];
  if (loadedAhead.has(id)) return;
  loadedAhead.add(id);

  const res = await fetch('page/' + id);
  cardData[id] = await res.json();
  if (idx === current) renderStack();
}

function renderStack() {
  const stack = document.getElementById('stack');
  stack.innerHTML = '';

  // render up to 2 cards (current + next peeking behind)
  for (let i = Math.min(current + 1, pages.length - 1); i >= current; i--) {
    const p = pages[i];
    const data = cardData[p.id];
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'card-' + i;

    const priorStatus = p.status !== 'unprocessed' ? p.status : null;
    if (!data) {
      card.innerHTML = `<div class="card-loading">Loading…</div>
        <div class="card-meta"><span class="card-id">${p.id}</span></div>`;
    } else {
      const statusBadge = priorStatus
        ? `<span class="card-status ${priorStatus}">${priorStatus}</span>` : '';
      card.innerHTML = `
        <div class="stamp stamp-yes">Approve</div>
        <div class="stamp stamp-no">Flag</div>
        <img class="card-img" src="data:image/png;base64,${data.overlay_b64}" draggable="false">
        <div class="card-meta">
          <span class="card-id">${p.id}</span>
          ${statusBadge}
          <span class="card-blanks">${data.blank_count} blanks</span>
        </div>`;
      if (i === current) attachSwipe(card, i);
    }

    // back card peeks slightly
    if (i > current) {
      card.style.transform = 'scale(0.94) translateY(10px)';
      card.style.zIndex = 0;
    } else {
      card.style.zIndex = 10;
    }
    stack.appendChild(card);
  }
}

// ── swipe mechanics ─────────────────────────────────────────────────────────
function attachSwipe(el, idx) {
  let startX, startY, dx = 0;

  function onStart(e) {
    const t = e.touches ? e.touches[0] : e;
    startX = t.clientX; startY = t.clientY; dx = 0;
    el.style.transition = 'none';
  }
  function onMove(e) {
    if (!startX) return;
    const t = e.touches ? e.touches[0] : e;
    dx = t.clientX - startX;
    const dy = t.clientY - startY;
    if (Math.abs(dx) < Math.abs(dy) && Math.abs(dx) < 10) return; // vertical scroll
    e.preventDefault();
    const rotate = dx * 0.06;
    el.style.transform = `translateX(${dx}px) rotate(${rotate}deg)`;
    // show stamps
    const yes = el.querySelector('.stamp-yes');
    const no  = el.querySelector('.stamp-no');
    if (yes && no) {
      yes.style.opacity = dx > 0 ? Math.min(dx / 80, 1) : 0;
      no.style.opacity  = dx < 0 ? Math.min(-dx / 80, 1) : 0;
    }
  }
  function onEnd() {
    if (!startX) return;
    startX = null;
    el.style.transition = 'transform .25s ease, opacity .25s ease';
    if (dx > 80)       commit(idx, 'approve', el);
    else if (dx < -80) commit(idx, 'flag', el);
    else {
      el.style.transform = '';
      const yes = el.querySelector('.stamp-yes');
      const no  = el.querySelector('.stamp-no');
      if (yes) yes.style.opacity = 0;
      if (no)  no.style.opacity  = 0;
    }
    dx = 0;
  }

  el.addEventListener('touchstart', onStart, {passive: true});
  el.addEventListener('touchmove',  onMove,  {passive: false});
  el.addEventListener('touchend',   onEnd,   {passive: true});
  el.addEventListener('mousedown',  onStart);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup',   onEnd);
}

function swipeCard(dir) {
  if (current >= pages.length) return;
  const el = document.getElementById('card-' + current);
  if (!el) return;
  el.style.transition = 'transform .3s ease, opacity .3s ease';
  const action = dir === 'right' ? 'approve' : 'flag';
  commit(current, action, el);
}

function deleteCard() {
  if (current >= pages.length) return;
  const el = document.getElementById('card-' + current);
  if (!el) return;
  el.style.transition = 'transform .3s ease, opacity .3s ease';
  el.style.transform = 'scale(0.8) translateY(40px)';
  el.style.opacity = '0';
  commit(current, 'delete', el, true);
}

function commit(idx, action, el, noSlide) {
  const id = pages[idx].id;
  decisions[id] = action;
  updateCounter();

  if (!noSlide) {
    const dir = action === 'approve' ? 1 : -1;
    el.style.transform = `translateX(${dir * 120}vw) rotate(${dir * 20}deg)`;
    el.style.opacity = '0';
  }

  setTimeout(() => {
    current++;
    renderStack();
    loadCard(current + 1);   // preload two ahead
  }, 280);
}

// ── commit to server ────────────────────────────────────────────────────────
async function commitAll() {
  if (!Object.keys(decisions).length) { alert('No decisions yet.'); return; }
  const btn = document.getElementById('done-btn');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const res = await fetch('apply', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(decisions)
    });
    const d = await res.json();
    document.getElementById('result').style.display = 'flex';
    document.getElementById('result-stats').textContent =
      `Approved: ${d.approved}  ·  Flagged: ${d.flagged}  ·  Deleted: ${d.deleted}`;
    document.getElementById('result-log').textContent = d.log;
  } catch(e) {
    alert('Error: ' + e); btn.disabled = false; btn.textContent = 'Done';
  }
}

init();
</script>
</body>
</html>"""

# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        qs   = self.path.split("?")[1] if "?" in self.path else ""

        if path in ("/", ""):
            self._send(200, "text/html", SHELL.encode())

        elif path == "/pages":
            rows = load_manifest()
            show_all = "all=1" in qs
            out = []
            for row in rows:
                rid = row["id"]
                status = page_status(rid)
                if not show_all and status != "unprocessed":
                    continue
                out.append({"id": rid, "status": status,
                             "source": row.get("source", ""),
                             "category": row.get("category", "")})
            self._send(200, "application/json", json.dumps({"pages": out}).encode())

        elif path.startswith("/page/"):
            rid = path[6:]
            rows = load_manifest()
            row = next((r for r in rows if r["id"] == rid), None)
            if not row:
                self._send(404, "application/json", b'{"error":"not found"}')
                return
            pdf_path = os.path.join(REPO, "dataset", row["file"])
            cached = call_blanq(pdf_path, rid)
            if cached.get("ok") and cached.get("pages"):
                pd = cached["pages"][0]
                overlay = render_overlay(pdf_path, pd)
                blank_count = pd.get("blankCount", 0)
            else:
                overlay = render_thumb(pdf_path)
                blank_count = 0
            result = {"id": rid, "overlay_b64": overlay,
                      "blank_count": blank_count, "status": page_status(rid)}
            self._send(200, "application/json", json.dumps(result).encode())

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/apply", "/rank/apply", ""):
            self._send(404, "text/plain", b"not found"); return

        decisions = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        log, approved, flagged, deleted = [], 0, 0, 0

        for rid, action in decisions.items():
            rows = load_manifest()
            row = next((r for r in rows if r["id"] == rid), None)
            if not row:
                log.append(f"[skip] {rid}: not in manifest"); continue
            pdf_path = os.path.join(REPO, "dataset", row["file"])

            if action == "approve":
                cached = json.load(open(cache_file(rid))) if os.path.exists(cache_file(rid)) else call_blanq(pdf_path, rid)
                if cached.get("ok") and cached.get("pages"):
                    pd = cached["pages"][0]
                    gt = blanks_to_gt(pd, rid, pdf_path)
                    with open(os.path.join(GT_DIR, f"{rid}.json"), "w") as f:
                        json.dump(gt, f, indent=2)
                    update_detections(rid, pd, pdf_path)
                    log.append(f"[approved] {rid}: {len(gt['blanks'])} blanks → ground_truth/")
                    approved += 1
                else:
                    log.append(f"[error] {rid}: API failed")

            elif action == "flag":
                cached = json.load(open(cache_file(rid))) if os.path.exists(cache_file(rid)) else None
                with open(os.path.join(RL_DIR, f"{rid}.json"), "w") as f:
                    json.dump({"id": rid, "reason": "detection_incorrect", "blanq_response": cached}, f, indent=2)
                log.append(f"[flagged] {rid} → review_later/")
                flagged += 1

            elif action == "delete":
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                all_rows = load_manifest()
                kept = [r for r in all_rows if r["id"] != rid]
                with open(MANIFEST, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                    w.writeheader(); w.writerows(kept)
                log.append(f"[deleted] {rid}")
                deleted += 1

        # git commit + push
        try:
            subprocess.run(["git", "-C", REPO, "add",
                            "ground_truth/", "review_later/",
                            "results/blanq/", "dataset/manifest.csv", "dataset/",
                            "docs/"], check=False, capture_output=True)
            status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip()
            if status:
                msg = f"Review: {approved} approved, {flagged} flagged, {deleted} deleted"
                subprocess.run(["git", "-C", REPO, "commit", "-m",
                                msg + "\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
                               check=False, capture_output=True)
                push = subprocess.run(["git", "-C", REPO, "push"],
                                      capture_output=True, text=True)
                log.append(f"[git] pushed ({'ok' if push.returncode == 0 else push.stderr.strip()})")
            else:
                log.append("[git] nothing to commit")
        except Exception as e:
            log.append(f"[git] {e}")

        self._send(200, "application/json",
                   json.dumps({"approved": approved, "flagged": flagged,
                               "deleted": deleted, "log": "\n".join(log)}).encode())

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    rows = load_manifest()
    t = threading.Thread(target=prefetch_worker, args=(rows,), daemon=True)
    t.start()
    unproc = sum(1 for r in rows if page_status(r["id"]) == "unprocessed")
    print(f"BlanQ review tool → http://localhost:{PORT}")
    print(f"Pre-fetching {unproc} unprocessed pages in background…")
    http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
