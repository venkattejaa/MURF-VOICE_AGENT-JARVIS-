# Recommended pip:
# pip install deepgram-sdk murf-api groq websockets sounddevice numpy pywebview pywhatkit python-dotenv
#
# Environment variables:
# DEEPGRAM_API_KEY, MURF_API_KEY, GROQ_API_KEY, WS_PORT, WAKEWORD, CREATOR_NAME, AI_MODEL, TTS_BOOST

import os, sys, time, json, queue, threading, datetime, sqlite3, re, random, traceback
from contextlib import redirect_stdout
from dotenv import load_dotenv
import os, sys, time, json, queue, threading, datetime, sqlite3, re, random, traceback
from contextlib import redirect_stdout
from dotenv import load_dotenv
import asyncio  # <-- ADD THIS LINE

# ---------------- CONFIG ----------------
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

# TTS BOOST selected: C -> max (5.0)
ENV_BOOST = os.getenv("TTS_BOOST")
TTS_BOOST = float(ENV_BOOST) if ENV_BOOST else 5.0

# ---------------- OPTIONAL IMPORTS ----------------
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
    from murf import Murf
except Exception:
    Murf = None

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

try:
    import webbrowser
except Exception:
    webbrowser = None
import dateparser

def set_reminder(text, when):
    """Store a reminder in DB."""
    try:
        ts = int(when.timestamp())
        conn = sqlite3.connect("jarvis_mem.db")
        conn.execute("INSERT INTO reminders (time, text) VALUES (?, ?)", (ts, text))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log("Reminder save error:", e)
        return False


def parse_reminder_command(cmd):
    """
    Extract reminder content + time from natural language.
    Examples:
        remind me in 10 minutes to drink water
        remind me at 5 PM to call mom
        remind me tomorrow morning to study
    """
    cmd = cmd.lower()

    if "remind me" not in cmd:
        return None, None

    # Extract the reminder message
    # (everything after "to")
    m = re.search(r"remind me .*? to (.+)", cmd)
    if not m:
        return None, None
    reminder_text = m.group(1).strip()

    # Extract time portion (everything after "remind me")
    t = cmd.replace("remind me", "").replace("to " + reminder_text, "").strip()
    parsed_time = dateparser.parse(t)

    return reminder_text, parsed_time


def list_reminders():
    conn = sqlite3.connect("jarvis_mem.db")
    rows = conn.execute("SELECT id, time, text, triggered FROM reminders ORDER BY time").fetchall()
    conn.close()
    return rows


def clear_reminders():
    conn = sqlite3.connect("jarvis_mem.db")
    conn.execute("DELETE FROM reminders")
    conn.commit()
    conn.close()


def reminder_loop():
    """Background thread that checks reminders every second."""
    while not shutdown_event.is_set():
        try:
            now = int(time.time())
            conn = sqlite3.connect("jarvis_mem.db")
            rows = conn.execute("SELECT id, text FROM reminders WHERE time <= ? AND triggered = 0", (now,)).fetchall()

            for rid, text in rows:
                # Mark triggered
                conn.execute("UPDATE reminders SET triggered = 1 WHERE id = ?", (rid,))
                conn.commit()

                # Speak reminder
                speak(f"Sir, reminder: {text}", interruptible=True, minimal=False)
                broadcast("SHOW_CONTENT", text, "REMINDER")

            conn.close()
        except Exception as e:
            log("Reminder loop error:", e)

        time.sleep(1)

# pycaw / volume removed as per user request

# ---------------- GLOBAL STATE ----------------
audio_queue = queue.Queue()
connected_clients = set()
last_content_data = {"text": "", "title": "", "mode": None}  # modes: snippet | research | teach
is_speaking = False
mic_active = True
stop_speaking = threading.Event()
shutdown_event = threading.Event()
speech_lock = threading.Lock()
gui_ready = threading.Event()
system_ready_flag = False
last_speech_time = 0.0
self_trigger_disabled_until = 0.0  # block STT while TTS speaks its own name
murf_client = None
groq_client = None
_ws_loop = None

ACK_PHRASES = ["Yes, Sir?", "Listening.", "Ready.", "Standing by."]

# ---------------- LOGGING ----------------
import logging
logging.getLogger("websockets.client").setLevel(logging.CRITICAL)
logging.getLogger("websockets.protocol").setLevel(logging.CRITICAL)

def log(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

# ---------------- SANITIZE PERSONA ----------------
def sanitize_for_speech(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    forbidden = [
        r"\bas an ai\b", r"\bas an ai model\b", r"\bi am a language model\b",
        r"\bi'?m a language model\b", r"\bi am a text-based ai\b", r"\bi'?m a text-based ai\b",
        r"\bi cannot\b", r"\bi can'?t\b", r"\bi do not have\b", r"\bi don't have\b",
        r"\bas a chatbot\b", r"\bas a bot\b"
    ]
    s = text
    for pat in forbidden:
        s = re.sub(pat, "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(re.sub(r"[^a-zA-Z0-9]","", s)) < 2:
        s = "Understood, Sir. I will handle that."
    s = re.sub(r"\bjarvis\b", VOCAL_NAME, s, flags=re.I)
    return s

# ---------------- WEBSOCKET HUD ----------------
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
            asyncio.run_coroutine_threadsafe(ws.send(payload), _ws_loop)
        except Exception:
            stale.append(ws)
    for s in stale:
        connected_clients.discard(s)

def start_ws_server(port=WS_PORT):
    global _ws_loop
    if websockets is None:
        log("websockets module missing; HUD disabled.")
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

# ---------------- DB & HISTORY ----------------
def init_db():
    try:
        conn = sqlite3.connect("jarvis_mem.db")
        conn.execute("CREATE TABLE IF NOT EXISTS memory (fact TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS history (role TEXT, content TEXT)")
        conn.close()
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

# ---------------- TOOLS & APPS ----------------
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
            # simple fallback: open name in default search
            webbrowser.open(f"https://www.google.com/search?q={name}")
            return "Opening web search for the app."
        return "App launcher unavailable."
    except Exception as e:
        log("Open app error:", e); return "Failed to open app."

# ---------------- Groq (Research = Groq-only) ----------------
def get_ai_response(prompt):
    sys_prompt = (
        f"You are {DISPLAY_NAME} ({JARVIS_ABBREV}) created by {CREATOR_NAME}.\n"
        "Persona: concise, witty, and always in-character. Address the user as 'Sir'. Do not identify as a language model, or mention limitations.\n"
        "If asked for code or research, return concise content appropriate for a HUD display."
    )
    if not groq_client:
        return sanitize_for_speech("Neural Core offline. I cannot provide advanced summaries right now, Sir.")
    try:
        messages = [{"role":"system","content":sys_prompt}]
        messages.extend(get_history(limit=6))
        messages.append({"role":"user","content":prompt})
        response = groq_client.chat.completions.create(model=AI_MODEL, messages=messages, temperature=0.25, max_tokens=700)
        out = response.choices[0].message.content.strip()
        return sanitize_for_speech(out)
    except Exception as e:
        log("Groq error:", e)
        err = str(e).lower()
        if "rate" in err or "limit" in err or "429" in err:
            return sanitize_for_speech("Neural Core is currently limited. Try again shortly, Sir.")
        return sanitize_for_speech("Neural Core error. I cannot fetch that right now, Sir.")

def groq_research_summary(topic):
    """
    Use Groq to produce a clean, English research summary for 'topic'.
    """
    prompt = (
        f"Produce a concise research summary about: {topic}\n"
        "- Output in Markdown.\n"
        "- Start with a one-line summary.\n"
        "- Then 3-6 bullet points covering key concepts, recent developments, uses, and references.\n"
        "- Keep it under 350 words."
    )
    resp = get_ai_response(prompt)
    return resp

# ---------------- TTS with Amplification ----------------
VOICE_MAP = {"en":"en-IN-aarav","hi":"hi-IN-neelam","te":"te-IN-venkat","ta":"en-IN-aarav"}

def choose_voice_and_text(text):
    m = re.match(r"^\[LANG:(HI|TE|TA)\]\s*(.*)$", text, flags=re.I)
    lang="en"; cleaned=text
    if m:
        tag = m.group(1).lower(); cleaned = m.group(2).strip()
        if tag=="hi": lang="hi"
        elif tag=="te": lang="te"
        elif tag=="ta": lang="ta"
    return VOICE_MAP.get(lang, VOICE_MAP["en"]), cleaned, lang

def speak(text, interruptible=True, minimal=False):
    """
    Murf TTS v6 — Ultra-stable, low-latency, mic-safe.
    
    Fixes:
    - Murf 500 latency glitches (empty PCM packets)
    - Out-of-bounds PCM crash
    - Long delays (6–8 sec) before speaking
    - Mic hearing its own name
    """

    import numpy as np
    global is_speaking, mic_active, last_speech_time, self_trigger_disabled_until

    if not text:
        return

    # Persona cleaning
    text = sanitize_for_speech(text)
    text = re.sub(r"\bJ\.?A\.?R\.?V\.?I\.?S\b", VOCAL_NAME, text, flags=re.I)

    if len(re.sub(r"[^a-zA-Z0-9]", "", text)) < 2:
        return

    # Stop old speech
    stop_speaking.set()
    time.sleep(0.02)
    stop_speaking.clear()

    # Voice & MIC handling
    is_speaking = True
    mic_active = False

    # When name spoken → mute longer
    if VOCAL_NAME in text.lower():
        self_trigger_disabled_until = time.time() + 3.5
    else:
        self_trigger_disabled_until = time.time() + 0.6

    # Remove code fences for voice
    display_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    speech_text  = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    # Minimal → only speak first sentence
    if minimal:
    # minimal mode → only 1 sentence
       # FULL SPEECH ALWAYS — ignore minimal flag
        minimal = False

# REMOVE ELSE LIMIT
# else:
#     pass  # speak entire text


    # HUD update
    broadcast("SPEAKING", display_text, f"AI: {display_text[:60]}")

    # ---------------------------
    # Worker thread begins
    # ---------------------------
    def worker():
        global is_speaking, mic_active, last_speech_time, self_trigger_disabled_until

        with speech_lock:
            try:
                # Choose voice
                voice_id, cleaned, lang = choose_voice_and_text(speech_text)

                # Murf stream
                stream = murf_client.text_to_speech.stream(
                    text=cleaned,
                    voice_id=voice_id,
                    style="Conversational",
                    model="GEN2",
                    format="PCM",
                    sample_rate=24000
                )

                # If missing audio modules
                if sd is None or np is None:
                    print(f"{DISPLAY_NAME}: {speech_text}")
                    return

                # Audio output
                with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as out:

                    pcm_buffer = b""
                    CHUNK = 4096
                    BOOST = float(TTS_BOOST)

                    # ------------------------------
                    # NEW: Ultra-stable streaming loop
                    # ------------------------------
                    for packet in stream:

                        if stop_speaking.is_set():
                            break

                        # Skip empty packets → fixes Murf 500 errors
                        if not packet or len(packet) < 2:
                            continue

                        # Ensure packet is bytes
                        if not isinstance(packet, bytes):
                            try:
                                packet = bytes(packet)
                            except Exception:
                                continue

                        pcm_buffer += packet

                        # Process whenever buffer large enough
                        while len(pcm_buffer) >= CHUNK:

                            raw = pcm_buffer[:CHUNK]
                            pcm_buffer = pcm_buffer[CHUNK:]

                            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

                            # Safe amplifier
                            arr *= BOOST
                            peak = np.max(np.abs(arr))
                            if peak > 30000:
                                arr *= (30000 / peak)

                            arr = np.clip(arr, -32768, 32767).astype(np.int16)
                            out.write(arr)

                    # Final buffer flush
                    if len(pcm_buffer) >= 2:
                        raw = pcm_buffer
                        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                        arr *= BOOST
                        peak = np.max(np.abs(arr))
                        if peak > 30000:
                            arr *= (30000 / peak)
                        arr = np.clip(arr, -32768, 32767).astype(np.int16)
                        out.write(arr)

            except Exception as e:
                log("TTS playback error:", e)
                print(f"{DISPLAY_NAME}: {speech_text}")

            finally:
                # End speech
                time.sleep(0.05)
                is_speaking = False
                mic_active = True
                self_trigger_disabled_until = time.time() + 0.5
                last_speech_time = time.time()
                broadcast("IDLE", "Awaiting Input")

    threading.Thread(target=worker, daemon=True).start()


# ---------------- Snippet / Research / Teach flows ----------------
SNIPPET_TRIGGERS = ["show example","show me an example","show snippet","code example","show code","give me a snippet","example in code","in snippet","example"]
RESEARCH_TRIGGERS = ["research","search for","find info","find information","show me research","do research","search"]
TEACH_TRIGGERS = ["teach me","explain like","explain simply","explain to me","teach"]

def looks_like_trigger(text, triggers):
    t = text.lower()
    return any(ph in t for ph in triggers)

def close_snippet(reason="User"):
    global last_content_data
    if last_content_data.get("mode")=="snippet":
        last_content_data = {"text":"", "title":"", "mode": None}
        broadcast("HIDE_CONTENT", "", f"Snippet closed: {reason}")
        return True
    return False

def update_snippet(display, title):
    global last_content_data
    last_content_data = {"text": display, "title": title, "mode": "snippet"}
    broadcast("SHOW_CONTENT", display, title)

# ---------------- Action parsing (volume removed) ----------------
def execute_action_from_text(text):
    tl = text.lower()
    if "play" in tl and ("song" in tl or "music" in tl or tl.startswith("play ")):
        m = re.search(r"play (?:song |music )?(?:named )?(.+)", text, flags=re.I)
        if m:
            return play_music(m.group(1).strip())
        else:
            # fallback to open youtube search
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
            last_content_data.update({"text": res, "title": f"RESEARCH: {q}", "mode":"research"})
            broadcast("SHOW_CONTENT", res, last_content_data["title"])
            speak(res.split("\n")[0] if isinstance(res,str) else "Search complete, Sir.", interruptible=False, minimal=True)
        threading.Thread(target=_do_search, args=(q,), daemon=True).start()
        return f"Searching for {q}"
    return None

# ---------------- Cinematic intro ----------------
def cinematic_intro():
    intro = (
        f"I am {DISPLAY_NAME} — {JARVIS_ABBREV}. "
        f"Operational core online. Designed and commissioned by {CREATOR_NAME}, Sir. "
        "I exist to assist, analyze, and execute with precision and discretion. "
        "At your command."
    )
    broadcast("SHOW_CONTENT", intro, "IDENTITY: J.A.R.V.I.S — CINEMATIC")
    speak(intro, interruptible=False, minimal=False)

# ---------------- PROCESS (main command handler) ----------------
def process(text):
    global last_content_data
    if not text: return
    t = text.strip()
    if not t: return
    tl = t.lower()

    # Quick confirmations
    if "can you hear" in tl:
        speak("Yes, Sir. I can hear and understand you.", interruptible=False, minimal=True)
        return

    # Close snippet explicitly
    if "close" in tl and "snippet" in tl:
        closed = close_snippet("Direct close request")
        speak("Snippet closed, Sir." if closed else "No snippet open, Sir.", interruptible=False, minimal=True)
        return

    # If teaching and user says later/pause
    if last_content_data.get("mode")=="teach" and ("later" in tl or "pause" in tl):
        last_content_data = {"text":"", "title":"", "mode":None}
        broadcast("HIDE_CONTENT", "", "Teach closed")
        speak("Very well, Sir. We will continue later.", interruptible=False, minimal=True)
        return

    # Context switch: if snippet open and incoming command not snippet-refinement and not explicit code request -> close snippet
    if last_content_data.get("mode")=="snippet":
        # If current text is a code-refinement follow up, keep it; otherwise close.
        refine_phrases = ["add","modify","change","use","show me","in c","in java","with function","with input","include"]
        if not (looks_like_trigger(t, SNIPPET_TRIGGERS) or any(p in tl for p in refine_phrases)):
            close_snippet("Context switch")

    # Automations-first
    action = execute_action_from_text(t)
    if action:
        speak(action, interruptible=False, minimal=True)
        save_history("user", t); save_history("assistant", action)
        return

    # Identity triggers
    id_triggers = ["who are you","what is your identity","tell me about yourself","who made you","identity"]
    if any(p in tl for p in id_triggers):
        cinematic_intro(); save_history("user", t); save_history("assistant", "Cinematic intro"); return

    # Snippet request (contextual)
    if looks_like_trigger(t, SNIPPET_TRIGGERS):
        # If snippet open and user is refining, create refinement prompt
        if last_content_data.get("mode")=="snippet":
            base_title = last_content_data.get("title","SNIPPET")
            prompt = f"Refine the following snippet per request: {t}\nCURRENT_SNIPPET:\n{last_content_data.get('text','')}\nReturn only a single code block and a one-line description."
        else:
            prompt = f"Provide a concise code example for: {t}. Return ONLY one code block in triple backticks and a one-line summary above it."
        resp = get_ai_response(prompt)
        code_match = re.search(r"```(?:\w*\n)?(.*?)```", resp, flags=re.DOTALL)
        code_content = code_match.group(1).strip() if code_match else resp.strip()
        title = "Example: " + (" ".join(t.split()[:8])).strip()
        display = "```" + code_content + "```" if code_content else resp
        update_snippet(display, f"SNIPPET: {title}")
        first_line = resp.split("\n")[0].strip()
        speak(first_line if first_line else "Displayed example, Sir.", interruptible=False, minimal=True)
        save_history("user", t); save_history("assistant", resp)
        return

    # Research (Groq-only)
    if looks_like_trigger(t, RESEARCH_TRIGGERS):
        topic = t
        speak(f"Compiling research on {topic}, Sir.", interruptible=False, minimal=True)
        def _do(topic):
            res = groq_research_summary(topic)
            last_content_data.update({"text": res, "title": f"RESEARCH: {topic}", "mode":"research"})
            broadcast("SHOW_CONTENT", res, last_content_data["title"])
            speak(res.split("\n")[0] if isinstance(res,str) else "Research complete, Sir.", interruptible=False, minimal=True)
        threading.Thread(target=_do, args=(topic,), daemon=True).start()
        save_history("user", t)
        return

    # Teach
    if looks_like_trigger(t, TEACH_TRIGGERS):
        prompt = f"Teach: {t}. Provide a simple analogy and structured explanation suitable for a beginner. Start with one-line summary and include bullet points."
        resp = get_ai_response(prompt)
        last_content_data = {"text": resp, "title": f"TEACH: {t}", "mode":"teach"}
        broadcast("SHOW_CONTENT", resp, last_content_data["title"])
        speak(resp.split("\n")[0] if isinstance(resp,str) else "Teaching complete, Sir.", interruptible=False, minimal=True)
        save_history("user", t); save_history("assistant", resp)
        return

    # Default: LLM chat
    broadcast("THINKING", "Processing...", "Querying Neural Core")
    resp = get_ai_response(t)
    save_history("user", t); save_history("assistant", resp)

    # If LLM instructed a play or open action, we'll parse again
    post_action = execute_action_from_text(resp if isinstance(resp,str) else "")
    if post_action:
        speak(post_action, interruptible=False, minimal=True)
        broadcast("PROCESSING", post_action, "Automation Executed")
        return

    # Quick time handling
    if "time" in tl and ("what" in tl or "tell" in tl or "current" in tl):
        speak(f"It is {datetime.datetime.now().strftime('%I:%M %p')}, Sir.", interruptible=False, minimal=True)
        return

    speak(resp if isinstance(resp,str) else str(resp), interruptible=True, minimal=True)

# ---------------- Deepgram STT (self-healing) ----------------
def mic_callback(indata, frames, t, status):
    now = time.time()
    if mic_active and now > self_trigger_disabled_until:
        try:
            audio_queue.put(indata.tobytes())
        except Exception:
            pass

def process_final_stt(clean):
    if not clean: return
    if clean == WAKEWORD:
        speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True)
        broadcast("LISTENING", "Listening...", "Wake Detected")
        return
    if clean.startswith(WAKEWORD + " "):
        cmd = clean[len(WAKEWORD):].strip()
        if cmd:
            threading.Thread(target=process, args=(cmd,), daemon=True).start(); return
        else:
            speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True); return
    if clean.endswith(" " + WAKEWORD):
        trimmed = clean[:-len(WAKEWORD)].strip()
        if trimmed:
            threading.Thread(target=process, args=(trimmed,), daemon=True).start(); return
        else:
            speak(random.choice(ACK_PHRASES), interruptible=False, minimal=True); return
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

            # EXACT ORIGINAL LOGIC — UNTOUCHED
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
                await asyncio.sleep(1)
                continue

            # EVENT HANDLER (same as before)
            def on_message(self, result, **kwargs):
                try:
                    transcript = result.channel.alternatives[0].transcript
                except Exception:
                    transcript = None

                if not transcript:
                    return

                if not result.is_final:
                    if not is_speaking:
                        broadcast("LISTENING", transcript + "...", "Partial")
                    return

                clean_raw = transcript.strip().lower()
                clean = re.sub(r"[^\w\s]", "", clean_raw)

                now = time.time()
                if is_speaking or now < self_trigger_disabled_until:
                    return

                log("User:", clean)
                process_final_stt(clean)

            # ATTACH EVENTS
            try:
                conn.on(LiveTranscriptionEvents.Transcript, on_message)
            except Exception:
                try:
                    conn.on("Transcript", on_message)
                except Exception:
                    log("Deepgram event attach failed; reconnecting.")
                    await asyncio.sleep(1)
                    continue

            if not conn.start(options):
                log("Deepgram start returned False; retrying...")
                await asyncio.sleep(1)
                continue

            log("✔ Deepgram Connected (stable)")

            if sd is None:
                log("sounddevice missing; cannot capture mic.")
                return

            stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                callback=mic_callback,
                dtype="int16",
                blocksize=2048
            )

            stream.start()
            last_alive = time.time()

            # MAIN LOOP — ONLY FIX IS HERE
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

                    # FIXED TIMEOUT (8 seconds instead of 2)
                    elif time.time() - last_alive > 8.0:
                        try:
                            conn.send(json.dumps({"type": "KeepAlive"}))
                            last_alive = time.time()
                        except Exception:
                            raise ConnectionError("keepalive failed")

                except ConnectionError as e:
                    log("⚠ STT connection error:", e)
                    break

                except Exception as e:
                    log("Deepgram streaming exception:", e)
                    break

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

# ---------------- System startup ----------------
def system_startup():
    global murf_client, groq_client, system_ready_flag
    
    log("Waiting for GUI connection...")
    gui_ready.wait(timeout=30)
    broadcast("BOOT", "", "Establish Uplink...", 10); time.sleep(0.2)
    broadcast("BOOT", "", "Mounting Memory...", 30)
    init_db()
    broadcast("BOOT", "", "Initializing Murf...", 50)
    try:
        murf_client = Murf(api_key=MURF_KEY) if Murf else None
    except Exception as e:
        log("Murf init failed:", e); murf_client = None
    broadcast("BOOT", "", "Connecting to Groq Core...", 70)
    try:
        groq_client = Groq(api_key=GROQ_API_KEY) if Groq else None
        if groq_client:
            try:
                groq_client.chat.completions.create(model=AI_MODEL, messages=[{"role":"user","content":"Ping"}], max_tokens=1)
                log("✔ Connected to Groq"); broadcast("BOOT","", "Neural Link Established",80)
            except Exception:
                broadcast("BOOT","", "NEURAL LINK FAILED",80)
        else:
            broadcast("BOOT","", "NEURAL LINK FAILED",80)
    except Exception as e:
        log("Groq init failed:", e); groq_client = None; broadcast("BOOT","", "NEURAL LINK FAILED",80)

    broadcast("BOOT", "", "Calibrating Audio...", 90)
    t_stt = threading.Thread(target=start_stt_task, daemon=True); t_stt.start()
    time.sleep(0.5)
    system_ready_flag = True
    broadcast("READY", "", "System Online", 100)
    speak("Systems online. At your service, Sir.", interruptible=False, minimal=True)

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    # Start HUD server
    if websockets:
        t_ws = threading.Thread(target=start_ws_server, daemon=True); t_ws.start()
    else:
        log("websockets missing; HUD disabled.")

    # Start system
    t_init = threading.Thread(target=system_startup, daemon=True); t_init.start()

    log("✔ Launching Interface...")

try:
    if webview:
        # Always load HTML from the same folder as this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_dir, "Jarvis_ui.html")

        if not os.path.exists(html_path):
            log(f"[ERROR] HUD not found at: {html_path}")
        else:
            log(f"[HUD] Loaded: {html_path}")

        window = webview.create_window(
            DISPLAY_NAME,
            url=f"file:///{html_path}",
            width=1000,
            height=700,
            background_color='#000000',
            frameless=False
        )

        webview.start()

    else:
        log("webview missing; run headless.")
        while not shutdown_event.is_set():
            time.sleep(0.5)

except Exception as e:
    log(f"GUI Error: {e}")

finally:
    shutdown_event.set()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

