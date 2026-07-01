"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => {
  if (!r.ok) return r.json().then((e) => Promise.reject(e)).catch(() => Promise.reject({ detail: r.statusText }));
  return r.json();
});

let voices = [];
let pollTimer = null;

// --------------------------- initial load --------------------------------- //
async function init() {
  try {
    voices = await api("/api/voices");
    populateVoiceSelect();
    await loadCharacters();
    await loadLibrary();
  } catch (e) {
    console.error(e);
  }
  $("addCharacterBtn").addEventListener("click", addCharacter);
  $("startBtn").addEventListener("click", startStory);
  $("toggleScript").addEventListener("click", () => {
    const s = $("script");
    s.classList.toggle("hidden");
    $("toggleScript").textContent = s.classList.contains("hidden") ? "Show script" : "Hide script";
  });
}

function populateVoiceSelect() {
  const sel = $("ncSpeaker");
  sel.innerHTML = "";
  voices.forEach((v) => {
    const o = document.createElement("option");
    o.value = v.name;
    o.textContent = `${v.name} — ${v.desc}` + (v.recommended ? "  ⭐" : "");
    sel.appendChild(o);
  });
}

// ----------------------------- characters --------------------------------- //
async function loadCharacters() {
  const chars = await api("/api/characters");
  const wrap = $("characterList");
  wrap.innerHTML = chars.length ? "" : '<p class="hint">No saved characters yet.</p>';
  chars.forEach((c) => {
    const label = document.createElement("label");
    const voiceTxt = c.voice_mode === "clone" ? "cloned voice" : (c.speaker || "—");
    label.innerHTML =
      `<input type="checkbox" class="charChk" value="${c.id}" />` +
      `<span>${escapeHtml(c.name)}</span>` +
      `<span class="char-voice">· ${escapeHtml(voiceTxt)}</span>`;
    wrap.appendChild(label);
  });
}

async function addCharacter() {
  const name = $("ncName").value.trim();
  const msg = $("addCharMsg");
  if (!name) { msg.className = "msg err"; msg.textContent = "Name is required."; return; }
  try {
    await api("/api/characters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        persona: $("ncPersona").value.trim() || null,
        voice_mode: "preset",
        speaker: $("ncSpeaker").value,
      }),
    });
    msg.className = "msg ok"; msg.textContent = `Saved "${name}".`;
    $("ncName").value = ""; $("ncPersona").value = "";
    await loadCharacters();
  } catch (e) {
    msg.className = "msg err"; msg.textContent = e.detail || "Could not save.";
  }
}

// ------------------------------ start story ------------------------------- //
function splitList(v) {
  return v.split(",").map((s) => s.trim()).filter(Boolean);
}

async function startStory() {
  const name = $("subjectName").value.trim();
  if (!name) { alert("Please enter the child's name."); return; }

  const character_ids = Array.from(document.querySelectorAll(".charChk:checked"))
    .map((c) => parseInt(c.value, 10));

  const payload = {
    subject_name: name,
    subject_age: $("subjectAge").value ? parseInt($("subjectAge").value, 10) : null,
    location: $("location").value.trim() || null,
    themes: splitList($("themes").value),
    interests: splitList($("interests").value),
    rating: $("rating").value,
    genre: $("genre").value,
    extra_notes: $("extraNotes").value.trim() || null,
    character_ids,
  };

  $("startBtn").disabled = true;
  $("noStory").classList.add("hidden");
  $("player").classList.add("hidden");
  $("readiness").classList.remove("hidden");
  setProgress(2, "Queued…");

  try {
    const { id } = await api("/api/stories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    pollStatus(id);
  } catch (e) {
    setProgress(0, "Error: " + (e.detail || "could not start"));
    $("startBtn").disabled = false;
  }
}

function setProgress(pct, label) {
  $("barFill").style.width = Math.max(0, Math.min(100, pct)) + "%";
  $("stageLabel").textContent = label || "";
}

function pollStatus(id) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const s = await api(`/api/stories/${id}/status`);
      setProgress(s.progress, s.stage || s.status);
      if (s.status === "ready") {
        clearInterval(pollTimer);
        $("startBtn").disabled = false;
        await showStory(id);
        await loadLibrary();
      } else if (s.status === "error") {
        clearInterval(pollTimer);
        $("startBtn").disabled = false;
        setProgress(0, "⚠️ " + (s.error || "Generation failed"));
      }
    } catch (e) {
      clearInterval(pollTimer);
      $("startBtn").disabled = false;
      setProgress(0, "⚠️ lost connection to server");
    }
  }, 1200);
}

// ----------------------------- show a story ------------------------------- //
async function showStory(id) {
  const story = await api(`/api/stories/${id}`);
  $("readiness").classList.add("hidden");
  $("player").classList.remove("hidden");
  $("storyTitle").textContent = story.title || "Tonight's Story";
  const audio = $("audio");
  audio.src = `/api/audio/${id}?t=${Date.now()}`;

  const script = $("script");
  script.innerHTML = "";
  (story.lines || []).forEach((ln) => {
    const div = document.createElement("div");
    div.className = "line";
    div.innerHTML =
      `<span class="who">${escapeHtml(ln.character_name)}</span> ` +
      `<span class="emo">(${escapeHtml(ln.emotion || "")}, voice: ${escapeHtml(ln.speaker || "—")})</span><br />` +
      escapeHtml(ln.text);
    script.appendChild(div);
  });
  script.classList.add("hidden");
  $("toggleScript").textContent = "Show script";
}

// ------------------------------- library ---------------------------------- //
async function loadLibrary() {
  const stories = await api("/api/stories");
  const lib = $("library");
  lib.innerHTML = stories.length ? "" : '<p class="hint">No stories yet.</p>';
  stories.forEach((s) => {
    const item = document.createElement("div");
    item.className = "lib-item";
    const badge = s.status === "ready"
      ? '<span class="badge ready">ready</span>'
      : s.status === "error"
        ? '<span class="badge error">error</span>'
        : `<span class="badge working">${escapeHtml(s.stage || s.status)}</span>`;
    item.innerHTML =
      `<div class="t">${escapeHtml(s.title || "(untitled)")} ${badge}</div>` +
      `<div class="s">${escapeHtml(s.subject_name || "")}${s.summary ? " · " + escapeHtml(s.summary) : ""}</div>`;
    if (s.status === "ready") item.addEventListener("click", () => showStory(s.id));
    lib.appendChild(item);
  });
}

function escapeHtml(str) {
  return (str ?? "").toString().replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init();
