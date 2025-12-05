# Requirements:
# pip install deepgram-sdk murf groq websockets sounddevice numpy pywebview pywhatkit python-dotenv

import os
import sys
import time
import json
import queue
import threading
import sqlite3
import re
import random
import traceback
from dotenv import load_dotenv
import asyncio

# config
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY", "")
MURF_KEY = os.getenv("MURF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

WS_PORT = int(os.getenv("WS_PORT", "8765"))
WAKEWORD = os.getenv("WAKEWORD", "jarvis").lower()
CREATOR_NAME = os.getenv("CREATOR_NAME", "Venkat Teja")
AI_MODEL = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")
DISPLAY_NAME = "J.A.R.V.I.S"
VOCAL_NAME = "jarvis"
JARVIS_ABBREV = "Just A Rather Very Intelligent System"
TTS_BOOST = float(os.getenv("TTS_BOOST", "2.5"))

# optional imports
try:
    import sounddevice as sd
    import numpy as np
except Exception:
    sd = None; np = None

try:
    import websockets
except Exception:
    websockets = None

try:
    from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
except Exception:
    DeepgramClient = None; LiveOptions = None; LiveTranscriptionEvents = None

try:
    from murf import Murf, MurfRegion
except Exception:
    Murf = None; MurfRegion = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import webview
except Exception:
    webview = None

try:
    import pywhatkit
except Exception:
    pywhatkit = None

import webbrowser
import dateparser
import datetime

# Global state
audio_queue = queue.Queue()
connected_clients = set()
last_content_data = {"text": "", "title": "", "mode": None}  # snippet|research|teach
is_speaking = False                # True while TTS playback active
stop_speaking = threading.Event()  # set() to request TTS stop
shutdown_event = threading.Event()
speech_lock = threading.Lock()
gui_ready = threading.Event()
system_ready_flag = False
self_trigger_disabled_until = 0.0  # ignore STT briefly after TTS ends
murf_client = None
groq_client = None
_ws_loop = None

ACK_PHRASES = ["Yes, Sir?", "Listening.", "Ready.", "Standing by."]

# logging
import logging
logging.getLogger("websockets.client").setLevel(logging.CRITICAL)
logging.getLogger("websockets.protocol").setLevel(logging.CRITICAL)
def log(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

# Utils
def sanitize_for_speech(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    forbidden = [
        r"\bas an ai\b", r"\bas an ai model\b", r"\bi am a language model\b",
        r"\bi'?m a language model\b", r"\bi am a text-based ai\b", r"\bi'?m a text-based ai\b",
    ]
    s = text
    for pat in forbidden:
        s = re.sub(pat, "", s, flags=re.I)
    s = re.sub(r"```[\s\S]*?```", "", s, flags=re.DOTALL)
    s = re.sub(r"`[^`]*`", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\bjarvis\b", VOCAL_NAME, s, flags=re.I)
    if len(re.sub(r"[^a-zA-Z0-9]", "", s)) < 2:
        return ""
    return s

# WebSocket HUD
async def _ws_handler(ws):
    connected_clients.add(ws)
    log("✔ GUI Client Connected")
    gui_ready.set()
    try:
        if system_ready_flag:
            await ws.send(json.dumps({"status":"READY","text":"System Online","log":"Reconnected","progress":100}))
    except Exception:
        pass
    try:
        await ws.wait_closed()
    except Exception:
        pass
    finally:
        connected_clients.discard(ws)

def broadcast(status, text="", logmsg="", progress=0):
    payload = json.dumps({"status": status, "text": text, "log": logmsg, "progress": progress})
    stale = []
    for ws in list(connected_clients):
        try:
            if _ws_loop:
                asyncio.run_coroutine_threadsafe(ws.send(payload), _ws_loop)
        except Exception:
            stale.append(ws)
    for s in stale:
        connected_clients.discard(s)

def start_ws_server(port=WS_PORT):
    global _ws_loop
    if websockets is None:
        log("websockets missing; HUD disabled.")
        return
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    async def runner():
        log(f"✔ GUI Server running on ws://localhost:{port}")
        async with websockets.serve(_ws_handler, "localhost", port):
            await asyncio.Future()
    try:
        _ws_loop.run_until_complete(runner())
    except Exception as e:
        log("WS start error:", e)

# DB
def init_db():
    try:
        conn = sqlite3.connect("jarvis_mem.db")
        conn.execute("CREATE TABLE IF NOT EXISTS memory (fact TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS history (role TEXT, content TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, time INTEGER, text TEXT, triggered INTEGER DEFAULT 0)")
        conn.commit(); conn.close()
    except Exception as e:
        log("DB init error:", e)

def save_history(role, content):
    try:
        conn = sqlite3.connect("jarvis_mem.db")
        conn.execute("INSERT INTO history (role, content) VALUES (?, ?)", (role, content))
        conn.commit(); conn.close()
    except Exception:
        pass

def get_history(limit=10):
    try:
        conn = sqlite3.connect("jarvis_mem.db")
        rows = conn.execute("SELECT role, content FROM history ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        out = []
        for r in reversed(rows):
            out.append({"role": r[0], "content": r[1]})
        return out
    except Exception:
        return []

# Tools
def play_music(song):
    try:
        broadcast("PROCESSING", f"Playing {song}...", "YouTube")
        if pywhatkit:
            pywhatkit.playonyt(song); return f"Playing {song}."
        if webbrowser:
            webbrowser.open(f"https://www.youtube.com/results?search_query={song}"); return f"Opening search for {song}."
        return "No method to play music."
    except Exception as e:
        log("Play music error:", e); return f"Error: {e}"

def open_app(name):
    try:
        if not name: return "No app specified, Sir."
        if webbrowser:
            webbrowser.open(f"https://www.google.com/search?q={name}"); return "Opening web search for the app."
        return "App launcher unavailable."
    except Exception as e:
        log("Open app error:", e); return "Failed to open app."

# AI (Groq)
def get_ai_response(prompt):
    sys_prompt = (
        f"You are {DISPLAY_NAME} pronounced as {VOCAL_NAME} ({JARVIS_ABBREV}) created by {CREATOR_NAME}.\n"
        "Persona: concise, witty, and always in-character. Address the user as 'Sir'. Do not identify as a language model, or mention limitations.\n"
        "Just speak a little, Even if you are explaining or anything just speak few words, never break this rule, unless you are teaching or something else but don't speak alot"


    )
    if not groq_client:
        return "Neural Core offline. I cannot provide advanced summaries right now, Sir."
    try:
        messages = [{"role":"system","content":sys_prompt}]
        messages.extend(get_history(limit=6))
        messages.append({"role":"user","content":prompt})
        response = groq_client.chat.completions.create(model=AI_MODEL, messages=messages, temperature=0.25, max_tokens=700)
        out = response.choices[0].message.content.strip()
        return out if isinstance(out, str) else str(out)
    except Exception as e:
        log("Groq error:", e)
        return "Neural Core error. I cannot fetch that right now, Sir."

def groq_research_summary(topic):
    prompt = (
        f"Produce a concise research summary about: {topic}\n"
        "- Output in Markdown.\n"
        "- Start with a one-line summary.\n"
        "- Then 3-6 bullet points.\n"
        "- Keep it under 350 words."
    )
    return get_ai_response(prompt)

# ------------------ TTS (Murf Falcon) ------------------
# This speak() uses Murf Falcon streaming and supports immediate interruption.
def speak(text, interruptible=True, minimal=False):
    """
    Interruptible Murf Falcon TTS. Uses murf_client global if initialized;
    otherwise creates a local Murf client instance (MurfRegion.GLOBAL).
    """
    global is_speaking, self_trigger_disabled_until, murf_client

    if not text: return
    text = sanitize_for_speech(text)
    if not text:
        return

    # request stop of any current TTS immediately
    stop_speaking.set()
    time.sleep(0.02)
    stop_speaking.clear()

    # mark speaking state
    is_speaking = True
    # short suppression AFTER TTS finishes — we still allow STT audio to be captured for barge-in
    self_trigger_disabled_until = time.time() + 0.5

    broadcast("SPEAKING", text, f"TTS: {text[:80]}")

    def tts_worker():
        global is_speaking, murf_client, self_trigger_disabled_until

        # ensure murf_client
        try:
            if murf_client is None:
                if Murf is None:
                    raise RuntimeError("Murf SDK missing")
                if MurfRegion is not None:
                    murf_client = Murf(api_key=MURF_KEY, region=MurfRegion.GLOBAL)
                else:
                    murf_client = Murf(api_key=MURF_KEY)
        except Exception as e:
            log("Murf init error in speak:", e)
            print(text)
            is_speaking = False
            return

        try:
            # user-specified Falcon usage (Matthew used as example; swap to Ronnie if you prefer)
            stream = murf_client.text_to_speech.stream(
                text=text,
                voice_id="Matthew",
                model="FALCON",
                multi_native_locale="en-US",
                sample_rate=24000
            )

            if sd is None or np is None:
                print(text)
                return

            # Output stream
            out = sd.OutputStream(samplerate=24000, channels=1, dtype='int16', blocksize=2048)
            out.start()

            pcm_buffer = b""
            BOOST = float(TTS_BOOST) if TTS_BOOST else 1.0

            for packet in stream:
                # interruption requested -> break (immediate)
                if stop_speaking.is_set():
                    break

                if not packet or len(packet) < 2:
                    continue

                if not isinstance(packet, (bytes, bytearray)):
                    try:
                        packet = bytes(packet)
                    except Exception:
                        continue

                pcm_buffer += packet

                while len(pcm_buffer) >= 4096:
                    chunk = pcm_buffer[:4096]
                    pcm_buffer = pcm_buffer[4096:]

                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    # boost safely
                    arr *= BOOST
                    peak = np.max(np.abs(arr)) if arr.size else 0
                    if peak > 30000:
                        arr *= (30000.0 / peak)
                    arr = np.clip(arr, -32768, 32767).astype(np.int16)
                    out.write(arr)

            # flush leftovers
            if len(pcm_buffer) >= 2:
                arr = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
                arr *= BOOST
                peak = np.max(np.abs(arr)) if arr.size else 0
                if peak > 30000:
                    arr *= (30000.0 / peak)
                arr = np.clip(arr, -32768, 32767).astype(np.int16)
                out.write(arr)

        except Exception as e:
            log("TTS playback error:", e)
            # fallback: print text to console
            print(text)
        finally:
            # mark finished — re-enable short window for self-trigger suppression
            is_speaking = False
            self_trigger_disabled_until = time.time() + 0.6
            broadcast("IDLE", "Awaiting input")

    threading.Thread(target=tts_worker, daemon=True).start()

# ---------------- Snippet / Research / Teach flows ----------------
SNIPPET_TRIGGERS = ["show example","show me an example","show snippet","code example","show code","give me a snippet","example in code","in snippet","example"]
RESEARCH_TRIGGERS = ["research","search for","find info","find information","show me research","do research","search"]
TEACH_TRIGGERS = ["teach me","explain like","explain simply","explain to me","teach"]

def looks_like_trigger(text, triggers):
    t = text.lower()
    return any(ph in t for ph in triggers)

def close_snippet(reason="User"):
    global last_content_data
    if last_content_data.get("mode") == "snippet":
        last_content_data = {"text": "", "title": "", "mode": None}
        broadcast("HIDE_CONTENT", "", f"Snippet closed: {reason}")
        return True
    return False
def close_any_active_panel():
    """Close snippet / research / teach UI panels safely."""
    global last_content_data
    
    mode = last_content_data.get("mode")
    if mode in ["snippet", "research", "teach"]:
        last_content_data = {"text": "", "title": "", "mode": None}
        broadcast("HIDE_CONTENT", "", f"{mode.upper()} Panel Closed")
        return True

    return False


def update_snippet(display, title):
    global last_content_data
    last_content_data = {"text": display, "title": title, "mode": "snippet"}
    broadcast("SHOW_CONTENT", display, title)



# Action parsing
def execute_action_from_text(text):
    tl = text.lower()
    if "play" in tl and ("song" in tl or "music" in tl or tl.startswith("play ")):
        m = re.search(r"play (?:song |music )?(?:named )?(.+)", text, flags=re.I)
        if m:
            return play_music(m.group(1).strip())
        else:
            q = tl.replace("play","").strip()
            return play_music(q)
    if "open" in tl and ("app" in tl or "open " in tl):
        m = re.search(r"open (?:the )?(?:app )?(?:called )?(.+)", text, flags=re.I)
        if m:
            return open_app(m.group(1).strip())
    if tl.startswith("search ") or "search for" in tl:
        q = re.sub(r"^search (for )?","", text, flags=re.I).strip()
        def _do_search(q):
            speak("Searching and summarizing, Sir.", interruptible=False, minimal=True)
            res = groq_research_summary(q)
            last_content_data.update({"text": res, "title": f"RESEARCH: {q}", "mode": "research"})
            broadcast("SHOW_CONTENT", res, last_content_data["title"])
            first = (res.split("\n")[0] if isinstance(res, str) and res else "Search complete, Sir.")
            speak(first, interruptible=False, minimal=True)
        threading.Thread(target=_do_search, args=(q,), daemon=True).start()
        return f"Searching for {q}"
    return None

# Cinematic intro
def cinematic_intro():
    intro = (
        f"Allow me to introduce myself"
        f"I am {VOCAL_NAME} — {JARVIS_ABBREV}. "
        f"Designed and Created by {CREATOR_NAME}, Sir. "
        "I exist to assist, analyze, and execute with precision and discretion. "
        "and I'm here to assist you with a variety of tasks as best I can, 24 hours a day seven days a week."
    )
    broadcast("SHOW_CONTENT", intro, "IDENTITY: J.A.R.V.I.S")
    speak(intro, interruptible=False, minimal=True)
# Main process
def process(text):
    global last_content_data
    if not text:
        return

    t = text.strip()
    tl = t.lower()

    # Quick confirmations
    if "can you hear" in tl:
        speak("Yes, Sir. Loud and clear.", interruptible=False, minimal=True)
        return

    # Direct close requests
    if "close" in tl and ("panel" in tl or "snippet" in tl or "research" in tl or "teach" in tl):
        closed = close_any_active_panel()
        speak("Closed, Sir." if closed else "Nothing is open, Sir.", interruptible=False, minimal=True)
        return

    # Handle wakeword-only commands (jarvis ...)
    # (Handled elsewhere but safe fallback)

    # Detect mode-change → auto close previous panel
    if last_content_data.get("mode") in ["snippet", "research", "teach"]:
        if looks_like_trigger(t, SNIPPET_TRIGGERS) or \
           looks_like_trigger(t, RESEARCH_TRIGGERS) or \
           looks_like_trigger(t, TEACH_TRIGGERS):
            close_any_active_panel()

    # ------------ AUTOMATION (play/open/search) ------------
    action = execute_action_from_text(t)
    if action:
        speak(action, interruptible=False, minimal=True)
        save_history("user", t)
        save_history("assistant", action)
        return

    # ------------ IDENTITY ------------
    id_triggers = ["who are you", "your identity", "tell me about yourself", "who made you"]
    if any(p in tl for p in id_triggers):
        cinematic_intro()
        save_history("user", t)
        save_history("assistant", "Cinematic intro")
        return

    # ------------ SNIPPET MODE ------------
    if looks_like_trigger(t, SNIPPET_TRIGGERS):
        close_any_active_panel()

        prompt = (
            f"Provide a concise code example for: {t}. "
            "Return ONLY:\n"
            "1. One-line summary\n"
            "2. One triple-backtick code block"
        )

        resp = get_ai_response(prompt)

        # Extract code
        code_match = re.search(r"```[\s\S]*?```", resp)
        snippet = code_match.group(0) if code_match else f"```\n{resp}\n```"

        update_snippet(snippet, f"SNIPPET: {t[:40]}")

        summary = resp.split("\n")[0].strip()
        speak(summary, interruptible=False, minimal=True)

        save_history("user", t)
        save_history("assistant", resp)
        return

    # ------------ RESEARCH MODE ------------
    if looks_like_trigger(t, RESEARCH_TRIGGERS):
        close_any_active_panel()

        speak("Researching, Sir.", interruptible=False, minimal=True)

        def _do(topic):
            result = groq_research_summary(topic)
            last_content_data.update({
                "text": result, "title": f"RESEARCH: {topic}", "mode": "research"
            })
            broadcast("SHOW_CONTENT", result, f"RESEARCH: {topic}")
            speak(result.split("\n")[0], interruptible=False, minimal=True)

        threading.Thread(target=_do, args=(t,), daemon=True).start()

        save_history("user", t)
        return

    # ------------ TEACH MODE ------------
    if looks_like_trigger(t, TEACH_TRIGGERS):
        close_any_active_panel()

        prompt = (
            f"Teach {t} in simple terms. "
            "Return a one-line summary then bullet points."
        )

        resp = get_ai_response(prompt)

        last_content_data = {
            "text": resp,
            "title": f"TEACH: {t}",
            "mode": "teach"
        }

        broadcast("SHOW_CONTENT", resp, f"TEACH: {t}")
        speak(resp.split("\n")[0], interruptible=False, minimal=True)

        save_history("user", t)
        save_history("assistant", resp)
        return

    # ------------ DEFAULT CONVERSATION ------------
    broadcast("THINKING", "Processing...", "Neural Core")
    resp = get_ai_response(t)

    save_history("user", t)
    save_history("assistant", resp)

    post_action = execute_action_from_text(resp)
    if post_action:
        speak(post_action, interruptible=False, minimal=True)
        broadcast("PROCESSING", post_action, "Automation Executed")
        return

    speak(resp, interruptible=True, minimal=True)

# ---------------- Deepgram STT ----------------
def mic_callback(indata, frames, t, status):
    now = time.time()

    # IGNORE audio while TTS is speaking or in suppression window
    if is_speaking or now < self_trigger_disabled_until:
        return

    try:
        audio_queue.put(indata.tobytes())
    except Exception:
        pass


def process_final_stt(clean):
    if not clean: return
    now = time.time()
    if now < self_trigger_disabled_until:
        log("Final transcript ignored due to recent TTS (self-trigger suppression).")
        return
    clean = clean.strip().lower()
    if clean == WAKEWORD:
        speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True)
        broadcast("LISTENING", "Listening...", "Wake Detected")
        return
    if clean.startswith(WAKEWORD + " "):
        cmd = clean[len(WAKEWORD):].strip()
        if cmd:
            threading.Thread(target=process, args=(cmd,), daemon=True).start()
            return
        else:
            speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True); return
    if clean.endswith(" " + WAKEWORD):
        trimmed = clean[:-len(WAKEWORD)].strip()
        if trimmed:
            threading.Thread(target=process, args=(trimmed,), daemon=True).start(); return
        else:
            speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True); return
    # normal final utterance
    threading.Thread(target=process, args=(clean,), daemon=True).start()

async def deepgram_loop():
    if DeepgramClient is None:
        log("Deepgram SDK missing; STT disabled.")
        return

    while not shutdown_event.is_set():
        try:
            log("⏳ Connecting to Deepgram...")
            dg = DeepgramClient(DG_KEY)

            options = LiveOptions(
                model="nova-2",
                language="en-IN",
                smart_format=True,
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                interim_results=True
            )

            conn = None
            try:
                conn = dg.listen.live.v("1")
            except Exception:
                try:
                    conn = dg.listen.live(options)
                except Exception:
                    conn = None

            if conn is None:
                log("Deepgram listen() not available; retrying...")
                await asyncio.sleep(1); continue

            def on_message(self, result, **kwargs):
                try:
                    transcript = result.channel.alternatives[0].transcript
                except Exception:
                    transcript = None
                if not transcript:
                    return

                # ---------- INTERIM ----------
                if not result.is_final:
                    transcript = transcript.strip().lower()

    # ONLY interrupt if interim contains the wakeword "jarvis"
                    if is_speaking and VOCAL_NAME in transcript:
                        log("Wakeword heard during TTS → interrupting.")
                        stop_speaking.set()
                        broadcast("LISTENING", "Yes Sir?", "Wakeword Interrupt")
                    return


                # final transcript
                clean_raw = transcript.strip().lower()
                clean = re.sub(r"[^\w\s]", "", clean_raw)
                now = time.time()
                if now < self_trigger_disabled_until:
                    log("Ignoring final transcript (self-trigger suppression)."); return

                log("User (final):", clean)
                process_final_stt(clean)

            # attach events
            try:
                conn.on(LiveTranscriptionEvents.Transcript, on_message)
            except Exception:
                try:
                    conn.on("Transcript", on_message)
                except Exception:
                    log("Deepgram event attach failed; reconnecting."); await asyncio.sleep(1); continue

            try:
                started = conn.start(options)
            except Exception:
                started = False

            if not started:
                log("Deepgram start returned False; retrying..."); await asyncio.sleep(1); continue

            log("✔ Deepgram Connected (stable)")

            if sd is None:
                log("sounddevice missing; cannot capture mic."); return

            stream = sd.InputStream(channels=1, samplerate=16000, callback=mic_callback, dtype="int16", blocksize=2048)
            stream.start()
            last_alive = time.time()
            KEEPALIVE_SEC = 8.0

            while not shutdown_event.is_set():
                await asyncio.sleep(0.01)
                try:
                    if not audio_queue.empty():
                        chunk = audio_queue.get()
                        try:
                            conn.send(chunk)
                            last_alive = time.time()
                        except Exception:
                            raise ConnectionError("send failed")
                    elif time.time() - last_alive > KEEPALIVE_SEC:
                        try:
                            conn.send(json.dumps({"type": "KeepAlive"}))
                            last_alive = time.time()
                        except Exception:
                            raise ConnectionError("keepalive failed")
                except ConnectionError as e:
                    log("⚠ STT connection error:", e); break
                except Exception as e:
                    log("Deepgram streaming exception:", e); break

            try: stream.stop()
            except: pass
            try: conn.finish()
            except: pass

            await asyncio.sleep(1)
            continue

        except Exception as e:
            log("⚠ STT loop failure:", e)
            await asyncio.sleep(1)
            continue

def start_stt_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(deepgram_loop())

# Reminders (kept minimal)
def set_reminder(text, when):
    try:
        ts = int(when.timestamp())
        conn = sqlite3.connect("jarvis_mem.db")
        conn.execute("INSERT INTO reminders (time, text) VALUES (?, ?)", (ts, text))
        conn.commit(); conn.close(); return True
    except Exception as e:
        log("Reminder save error:", e); return False

def parse_reminder_command(cmd):
    cmd = cmd.lower()
    if "remind me" not in cmd:
        return None, None
    m = re.search(r"remind me .*? to (.+)", cmd)
    if not m:
        return None, None
    reminder_text = m.group(1).strip()
    t = cmd.replace("remind me", "").replace("to " + reminder_text, "").strip()
    parsed_time = dateparser.parse(t)
    return reminder_text, parsed_time

def list_reminders():
    conn = sqlite3.connect("jarvis_mem.db")
    rows = conn.execute("SELECT id, time, text, triggered FROM reminders ORDER BY time").fetchall()
    conn.close(); return rows

def clear_reminders():
    conn = sqlite3.connect("jarvis_mem.db"); conn.execute("DELETE FROM reminders"); conn.commit(); conn.close()

def reminder_loop():
    while not shutdown_event.is_set():
        try:
            now = int(time.time())
            conn = sqlite3.connect("jarvis_mem.db")
            rows = conn.execute("SELECT id, text FROM reminders WHERE time <= ? AND triggered = 0", (now,)).fetchall()
            for rid, text in rows:
                conn.execute("UPDATE reminders SET triggered = 1 WHERE id = ?", (rid,)); conn.commit()
                speak(f"Sir, reminder: {text}", interruptible=True, minimal=False)
                broadcast("SHOW_CONTENT", text, "REMINDER")
            conn.close()
        except Exception as e:
            log("Reminder loop error:", e)
        time.sleep(1)

# System startup & Entrypoint
def system_startup():
    global murf_client, groq_client, system_ready_flag
    log("Waiting for GUI connection...")
    gui_ready.wait(timeout=30)
    broadcast("BOOT", "", "Establish Uplink...", 10); time.sleep(0.2)
    broadcast("BOOT", "", "Mounting Memory...", 30)
    init_db()
    broadcast("BOOT", "", "Initializing Murf...", 50)
    try:
        if Murf is not None:
            if MurfRegion is not None:
                murf_client = Murf(api_key=MURF_KEY, region=MurfRegion.GLOBAL)
            else:
                murf_client = Murf(api_key=MURF_KEY)
    except Exception as e:
        log("Murf init failed:", e); murf_client = None
    broadcast("BOOT", "", "Connecting to Groq Core...", 70)
    try:
        groq_client = Groq(api_key=GROQ_API_KEY) if Groq else None
        if groq_client:
            try:
                groq_client.chat.completions.create(model=AI_MODEL, messages=[{"role":"user","content":"ping"}], max_tokens=1)
                log("✔ Connected to Groq"); broadcast("BOOT","", "Neural Link Established",80)
            except Exception:
                broadcast("BOOT","", "NEURAL LINK FAILED",80)
        else:
            broadcast("BOOT","", "NEURAL LINK FAILED",80)
    except Exception as e:
        log("Groq init failed:", e); groq_client = None; broadcast("BOOT","", "NEURAL LINK FAILED",80)

    broadcast("BOOT", "", "Calibrating Audio...", 90)
    t_stt = threading.Thread(target=start_stt_task, daemon=True); t_stt.start()
    t_rem = threading.Thread(target=reminder_loop, daemon=True); t_rem.start()
    time.sleep(0.5)
    system_ready_flag = True
    broadcast("READY", "", "System Online", 100)
    speak("Systems online. At your service, Sir.", interruptible=False, minimal=True)

if __name__ == "__main__":
    if websockets:
        t_ws = threading.Thread(target=start_ws_server, daemon=True); t_ws.start()
    else:
        log("websockets missing; HUD disabled.")
    t_init = threading.Thread(target=system_startup, daemon=True); t_init.start()
    log("✔ Launching Interface...")

    try:
        if webview:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            html_path_candidates = ["Jarvis_ui.html","jarvis_ui.html","Jarvis_UI.html","HUD.html"]
            html_path = None
            for n in html_path_candidates:
                p = os.path.join(base_dir, n)
                if os.path.exists(p):
                    html_path = p; break
            if not html_path:
                log(f"[ERROR] HUD not found at: {os.path.join(base_dir, 'Jarvis_ui.html')}")
            else:
                log(f"[HUD] Loaded: {html_path}")
                window = webview.create_window(DISPLAY_NAME, url=f"file:///{html_path}", width=1000, height=700, background_color='#000000', frameless=False)
                webview.start()
        else:
            log("webview missing; run headless.")
            while not shutdown_event.is_set():
                time.sleep(0.5)
    except Exception as e:
        log("GUI Error:", e, traceback.format_exc())
    finally:
        shutdown_event.set()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

