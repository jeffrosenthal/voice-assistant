import os
import re
import json
import time
import threading
import subprocess
import webbrowser
import numpy as np
import requests
import sounddevice as sd
from scipy.signal import resample
from faster_whisper import WhisperModel
from openwakeword.model import Model


# -----------------------------
# CONFIG
# -----------------------------

MIC_DEVICE = 6
SAMPLE_RATE = 48000
WAKE_THRESHOLD = 0.25
MIC_GAIN = 20
COMMAND_SECONDS = 4

VOICE_MODEL = "/home/mint/voices/en_US-amy-medium.onnx"


# -----------------------------
# MODELS
# -----------------------------

print("Loading Whisper...")
whisper = WhisperModel("base", compute_type="int8")

print("Loading wake word...")
wake_model = Model()

print("Warming up Gemma2...")
requests.post("http://localhost:11434/api/generate", json={
    "model": "gemma2:2b", "prompt": "hi", "stream": False,
    "options": {"num_predict": 1}
}, timeout=60)


# -----------------------------
# GLOBAL STATE
# -----------------------------

wake_triggered = False
last_wake_time = 0
stream = None
timer_count = 0
active_timers = {}  # {number: {"name": str, "name_file": str}}


# -----------------------------
# SPEECH OUTPUT
# -----------------------------

def speak(text):

    cmd = f'echo "{text}" | piper --model {VOICE_MODEL} --output-raw | aplay -r 22050 -f S16_LE -t raw -D plughw:0,0'

    subprocess.run(cmd, shell=True)


# -----------------------------
# RECORD COMMAND
# -----------------------------

def record_command():

    print("Listening for command...")

    chunks = []
    silent_chunks = 0
    heard_speech = False
    max_silent = 8  # ~0.64s of silence after speech to stop
    max_chunks = 80  # ~6.4s max recording

    with sd.InputStream(device=MIC_DEVICE, samplerate=SAMPLE_RATE, blocksize=3840, channels=1) as mic:
        # discard first few noisy chunks from stream opening
        for _ in range(5):
            mic.read(3840)

        for _ in range(max_chunks):
            data, _ = mic.read(3840)
            chunks.append(data[:, 0].copy())

            level = np.mean(np.abs(data))
            if level > 0.002:
                heard_speech = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            if heard_speech and silent_chunks >= max_silent:
                break

    if not chunks or not heard_speech:
        return ""

    import scipy.io.wavfile as wav
    audio = np.concatenate(chunks)
    audio_int16 = np.clip(audio * MIC_GAIN * 32767, -32768, 32767).astype(np.int16)
    wav.write("input.wav", SAMPLE_RATE, audio_int16)

    segments, _ = whisper.transcribe("input.wav")

    text = ""

    for s in segments:
        text += s.text

    text = text.strip().lower()

    print("You said:", text)

    return text


# -----------------------------
# COMMAND HANDLER
# -----------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

CLASSIFY_SYSTEM = '''Parse the voice command into JSON. Reply with ONLY valid JSON, no other text.
Do NOT calculate seconds yourself. Put the raw number and unit.
Available actions:

Set NEW timer: {\"action\":\"timer\",\"amount\":30,\"unit\":\"minute\",\"name\":\"Take cat out\"}
Cancel timer: {\"action\":\"cancel_timer\",\"number\":1}
Add time to EXISTING timer: {\"action\":\"add_time\",\"number\":1,\"amount\":5,\"unit\":\"minute\"}
Rename timer: {\"action\":\"rename_timer\",\"number\":1,\"name\":\"New name\"}
Open website: {\"action\":\"browse\",\"url\":\"youtube.com\"}
Search google: {\"action\":\"search\",\"query\":\"best pizza near me\"}
Get time: {\"action\":\"time\"}
Get date: {\"action\":\"date\"}
Get weather: {\"action\":\"weather\"}
General question: {\"action\":\"question\"}

RULES:
- "number" for add_time/cancel/rename = the timer NUMBER from the active list, NOT the amount.
- "set/start a timer" or "remind me to X in Y" = NEW timer action.
- "add X to [timer name]" or "X more to [timer name]" = add_time to existing timer.
- Only use add_time if the user specifically references an existing timer by name or number.
- For rename: match timer by name or number. "name" field is the NEW name.'''


def ollama_classify(text):

    # inject active timer info so AI knows what's running
    system = CLASSIFY_SYSTEM
    if active_timers:
        timer_info = "\n\nCurrently active timers:"
        for num, info in active_timers.items():
            timer_info += f"\n  Timer {num}: \"{info['name']}\""
        system += timer_info

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": "gemma2:2b",
            "prompt": text,
            "system": system,
            "stream": False,
            "format": "json",
            "context": [],
            "options": {"temperature": 0.1, "num_predict": 60}
        }, timeout=60)
        result = r.json().get("response", "").strip()
        print(f"  AI parsed: {result}")
        return json.loads(result)
    except Exception as e:
        print(f"  AI parse error: {e}")
        return None


def ollama_answer(question):

    today = time.strftime("%A, %B %d, %Y")
    now = time.strftime("%I:%M %p")

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": "gemma2:2b",
            "prompt": question,
            "system": f"Today is {today}, the current time is {now}. Answer in one or two short sentences.",
            "stream": False,
            "context": [],
            "options": {"temperature": 0.3, "num_predict": 100}
        }, timeout=30)
        answer = r.json().get("response", "").strip()
    except Exception:
        answer = ""

    answer = re.sub(r'[*#`]', '', answer)
    return answer if answer else "I could not come up with an answer"


def parse_time_from_text(text):
    """Extract total seconds from text, handling compound times like 'three minutes and twenty two seconds'."""
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "forty five": 45, "fifty": 50, "sixty": 60,
        "twenty five": 25, "twenty two": 22, "twenty three": 23,
        "twenty four": 24, "twenty six": 26, "twenty seven": 27,
        "twenty eight": 28, "twenty nine": 29, "thirty five": 35,
    }
    multiplier = {"second": 1, "minute": 60, "hour": 3600}

    def find_number_before(text, unit_pos):
        """Find the number (digit or word) right before a unit."""
        before = text[:unit_pos].strip()
        # check for digit
        digit_match = re.search(r'(\d+)\s*$', before)
        if digit_match:
            return int(digit_match.group(1))
        # check for word number (longest match first)
        for word, num in sorted(word_to_num.items(), key=lambda x: -len(x[0])):
            if before.endswith(word):
                return num
        return None

    total = 0
    found = False

    # find all unit mentions and the number before each
    for unit_match in re.finditer(r'(seconds?|minutes?|hours?)', text):
        num = find_number_before(text, unit_match.start())
        if num is not None:
            unit = unit_match.group(1).rstrip("s")
            total += num * multiplier.get(unit, 1)
            found = True

    if found:
        return total

    return None


def amount_to_seconds(amount, unit):
    """Convert amount and unit to seconds."""
    multiplier = {"second": 1, "minute": 60, "hour": 3600}
    unit = str(unit).rstrip("s").lower()
    return int(amount) * multiplier.get(unit, 1)


def start_timer(seconds, name=None):

    global timer_count
    timer_count += 1
    num = timer_count
    if not name:
        name = f"Timer {num}"

    control_dir = f"/tmp/timer_{num}"
    os.makedirs(control_dir, exist_ok=True)
    with open(os.path.join(control_dir, "name"), "w") as f:
        f.write(name)

    active_timers[num] = {"name": name, "control_dir": control_dir}

    def _timer():
        proc = subprocess.Popen(["python3", "/home/mint/timer_app.py", str(seconds), name, control_dir])
        proc.wait()
        if num in active_timers:
            current_name = active_timers[num]["name"]
            # only announce if not cancelled
            cancel_file = os.path.join(control_dir, "cancel")
            if not os.path.exists(cancel_file):
                speak(f"{current_name} done")
            active_timers.pop(num, None)
        import shutil
        shutil.rmtree(control_dir, ignore_errors=True)

    t = threading.Thread(target=_timer, daemon=True)
    t.start()
    speak(f"{name} set")


def rename_timer(num, new_name):

    if num not in active_timers:
        speak(f"Timer {num} not found")
        return

    active_timers[num]["name"] = new_name
    with open(os.path.join(active_timers[num]["control_dir"], "name"), "w") as f:
        f.write(new_name)

    speak(f"Timer {num} renamed to {new_name}")


def cancel_timer(num):

    if num not in active_timers:
        speak(f"Timer {num} not found")
        return

    name = active_timers[num]["name"]
    with open(os.path.join(active_timers[num]["control_dir"], "cancel"), "w") as f:
        f.write("1")

    active_timers.pop(num, None)
    speak(f"{name} cancelled")


def add_time_to_timer(num, extra_seconds):

    if num not in active_timers:
        speak(f"Timer {num} not found")
        return

    name = active_timers[num]["name"]
    with open(os.path.join(active_timers[num]["control_dir"], "add"), "w") as f:
        f.write(str(extra_seconds))

    mins = extra_seconds // 60
    secs = extra_seconds % 60
    if mins and secs:
        speak(f"Added {mins} minutes and {secs} seconds to {name}")
    elif mins:
        speak(f"Added {mins} minutes to {name}")
    else:
        speak(f"Added {secs} seconds to {name}")


def handle_command(text):

    if text == "":
        speak("I did not hear anything")
        return

    print(f"Processing: {text}")
    cmd = ollama_classify(text)

    if not cmd or "action" not in cmd:
        speak("I did not understand")
        return

    action = cmd["action"]

    # post-classification fix: if AI said "timer" but text references an existing timer name,
    # it's probably "add time" not "new timer"
    if action == "timer" and active_timers:
        cmd_name = (cmd.get("name") or "").lower()
        for num, info in active_timers.items():
            timer_name = info["name"].lower()
            if (cmd_name and (cmd_name in timer_name or timer_name in cmd_name)) or \
               (timer_name != f"timer {num}" and timer_name in text.lower()):
                action = "add_time"
                cmd["action"] = "add_time"
                cmd["number"] = num
                break

    if action == "timer":
        name = cmd.get("name")
        # always use our own time parsing - AI is bad at math
        secs = parse_time_from_text(text)
        if not secs:
            amount = cmd.get("amount")
            unit = cmd.get("unit", "second")
            if amount and unit:
                secs = amount_to_seconds(amount, unit)
            else:
                secs = 60
        start_timer(secs, name)

    elif action == "cancel_timer":
        num = cmd.get("number")
        if num:
            cancel_timer(int(num))
        elif len(active_timers) == 1:
            cancel_timer(next(iter(active_timers)))
        else:
            speak("Which timer?")

    elif action == "add_time":
        num = cmd.get("number")
        secs = parse_time_from_text(text)
        if not secs:
            amount = cmd.get("amount")
            unit = cmd.get("unit", "second")
            secs = amount_to_seconds(amount, unit) if amount else None
        if not num and len(active_timers) == 1:
            num = next(iter(active_timers))
        if num and secs:
            add_time_to_timer(int(num), secs)
        else:
            speak("Which timer and how much time?")

    elif action == "rename_timer":
        num = cmd.get("number")
        name = cmd.get("name")
        if num and name:
            rename_timer(int(num), name)
        else:
            speak("Which timer and what name?")

    elif action == "browse":
        url = cmd.get("url", "")
        if url:
            if not url.startswith("http"):
                url = "https://" + url
            webbrowser.open(url)
            speak(f"Opening {cmd.get('url')}")
        else:
            speak("What website?")

    elif action == "search":
        query = cmd.get("query", "")
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
            speak(f"Searching for {query}")
        else:
            speak("What should I search for?")

    elif action == "time":
        speak(time.strftime("The time is %I %M %p"))

    elif action == "date":
        speak(time.strftime("Today is %A, %B %d, %Y"))

    elif action == "weather":
        try:
            result = subprocess.run(
                ["curl", "-s", "wttr.in/?format=%C,+%t,+%h+humidity,+wind+%w"],
                capture_output=True, text=True, timeout=5
            )
            weather = result.stdout.strip()
            weather = weather.replace("+", "").replace("°C", " degrees")
            weather = re.sub(r'[↑↓←→↙↘↗↖]', '', weather)
            speak(f"Currently {weather}")
        except Exception:
            speak("I could not get the weather")

    elif action == "question":
        speak("Let me think")
        answer = ollama_answer(text)
        print("Answer:", answer)
        speak(answer)

    else:
        speak("Let me think")
        answer = ollama_answer(text)
        print("Answer:", answer)
        speak(answer)


# -----------------------------
# WAKE WORD CALLBACK
# -----------------------------

def callback(indata, frames, time_info, status):

    global wake_triggered, last_wake_time

    audio = indata[:, 0]

    # boost quiet mic and convert 48k -> 16k for wake model
    audio = audio * MIC_GAIN
    audio = resample(audio, int(len(audio) / 3))

    # convert to int16 range (openwakeword expects int16)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    prediction = wake_model.predict(audio_int16)

    score = prediction.get("hey_jarvis", 0)

    if score > WAKE_THRESHOLD and time.time() - last_wake_time > 3:

        print("Wake word detected")

        last_wake_time = time.time()

        wake_triggered = True


# -----------------------------
# START STREAM
# -----------------------------

def start_stream():

    global stream

    stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        blocksize=3840,
        channels=1,
        callback=callback
    )

    stream.start()


# -----------------------------
# MAIN
# -----------------------------

start_stream()

# play ready tone
subprocess.run(
    "python3 -c '"
    "import numpy as np,sys;"
    "t=np.linspace(0,0.15,int(22050*0.15));"
    "tone=np.concatenate([np.sin(2*np.pi*880*t)*0.3,np.sin(2*np.pi*1320*t)*0.3]);"
    "sys.stdout.buffer.write((tone*32767).astype(np.int16).tobytes())"
    "' | aplay -r 22050 -f S16_LE -t raw -D plughw:0,0",
    shell=True
)

print("Assistant ready. Say 'Hey Jarvis'")

while True:

    if wake_triggered:

        wake_triggered = False

        stream.stop()
        stream.close()

        wake_model.reset()

        time.sleep(0.3)

        speak("Yes")

        text = record_command()

        handle_command(text)

        wake_model.reset()
        last_wake_time = time.time()
        start_stream()

    time.sleep(0.1)
