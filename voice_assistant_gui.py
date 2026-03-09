#!/usr/bin/env python3
import os
import re
import json
import time
import threading
import subprocess
import webbrowser
import tkinter as tk
import numpy as np
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
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

VOICE_MODEL = "/home/mint/voices/en_US-amy-medium.onnx"
OLLAMA_URL = "http://localhost:11434/api/generate"

TV_IP = "192.168.1.150:7345"
TV_AUTH = "Z8lq39gk31"


# -----------------------------
# GUI
# -----------------------------

class AssistantGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Jarvis")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("800x700")
        self.root.attributes("-topmost", True)

        self.state = "loading"  # loading, idle, listening, processing, speaking
        self.prompt_count = 0
        self.on_timer_done = None  # callback when timer finishes
        self.on_tv_toggle = None   # callback for TV power button
        self.on_tv_vol = None      # callback for volume buttons
        self.on_mic_toggle = None  # callback for mic mute button

        self._build_ui()
        self._update_clock()

    def _build_ui(self):

        # top bar: clock
        top = tk.Frame(self.root, bg="#1a1a2e")
        top.pack(fill="x", padx=20, pady=(15, 5))

        self.clock_label = tk.Label(
            top, text="", font=("Helvetica", 28, "bold"),
            fg="#e0e0e0", bg="#1a1a2e"
        )
        self.clock_label.pack(side="left")

        self.date_label = tk.Label(
            top, text="", font=("Helvetica", 14),
            fg="#888888", bg="#1a1a2e"
        )
        self.date_label.pack(side="left", padx=(15, 0), pady=(8, 0))

        self.status_text = tk.Label(
            top, text="LOADING", font=("Helvetica", 14, "bold"),
            fg="#666666", bg="#1a1a2e"
        )
        self.status_text.pack(side="right")

        self.mic_muted = False
        self.mic_btn = tk.Button(
            top, text="MIC ON", font=("Helvetica", 10, "bold"),
            bg="#1a4a1a", fg="#44cc44", activebackground="#2a6a2a",
            relief="flat", padx=8, pady=4,
            command=self._toggle_mic
        )
        self.mic_btn.pack(side="right", padx=(0, 10))

        # middle area
        mid = tk.Frame(self.root, bg="#1a1a2e")
        mid.pack(fill="both", expand=True, padx=20, pady=5)

        # left: status circle
        left = tk.Frame(mid, bg="#1a1a2e", width=200)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        self.canvas = tk.Canvas(left, width=160, height=160, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(pady=(20, 10))
        self.circle = self.canvas.create_oval(10, 10, 150, 150, fill="#333355", outline="#444477", width=3)
        self.audio_level = 0.0  # updated by engine
        self._animate_circle()

        self.state_label = tk.Label(
            left, text="Loading...", font=("Helvetica", 12),
            fg="#888888", bg="#1a1a2e"
        )
        self.state_label.pack()

        # weather display below circle
        weather_frame = tk.Frame(left, bg="#1a1a2e")
        weather_frame.pack(pady=(15, 0), fill="x")

        self.weather_canvas = tk.Canvas(weather_frame, width=70, height=70, bg="#1a1a2e", highlightthickness=0)
        self.weather_canvas.pack()

        self.weather_temp = tk.Label(
            weather_frame, text="", font=("Helvetica", 16, "bold"),
            fg="#e0e0e0", bg="#1a1a2e"
        )
        self.weather_temp.pack()

        self.weather_desc = tk.Label(
            weather_frame, text="", font=("Helvetica", 10),
            fg="#888888", bg="#1a1a2e"
        )
        self.weather_desc.pack()

        self.weather_sun = tk.Label(
            weather_frame, text="", font=("Helvetica", 9),
            fg="#cc8844", bg="#1a1a2e"
        )
        self.weather_sun.pack()

        self.weather_updated = tk.Label(
            weather_frame, text="", font=("Helvetica", 8),
            fg="#555555", bg="#1a1a2e"
        )
        self.weather_updated.pack(pady=(3, 0))

        # right: timers
        right = tk.Frame(mid, bg="#1a1a2e")
        right.pack(side="left", fill="both", expand=True)

        timer_header = tk.Label(
            right, text="TIMERS", font=("Helvetica", 12, "bold"),
            fg="#666666", bg="#1a1a2e", anchor="w"
        )
        timer_header.pack(fill="x")

        self.timer_frame = tk.Frame(right, bg="#1a1a2e")
        self.timer_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.no_timers_label = tk.Label(
            self.timer_frame, text="No active timers",
            font=("Helvetica", 11), fg="#555555", bg="#1a1a2e", anchor="w"
        )
        self.no_timers_label.pack(anchor="w")

        self.timer_widgets = {}

        # TV controls row
        tv_row = tk.Frame(right, bg="#1a1a2e")
        tv_row.pack(fill="x", pady=(10, 0))

        tk.Label(tv_row, text="TV", font=("Helvetica", 12, "bold"),
                 fg="#666666", bg="#1a1a2e").pack(side="left", padx=(0, 10))

        self.tv_btn = tk.Canvas(tv_row, width=50, height=30, bg="#1a1a2e", highlightthickness=0, cursor="hand2")
        self.tv_btn.pack(side="left")
        self.tv_power_on = False
        self._draw_tv_btn()
        self.tv_btn.bind("<Button-1>", self._on_tv_btn_click)

        self.tv_status = tk.Label(tv_row, text="OFF", font=("Helvetica", 9),
                                  fg="#555555", bg="#1a1a2e")
        self.tv_status.pack(side="left", padx=(5, 15))

        tk.Button(tv_row, text="VOL-", font=("Helvetica", 9, "bold"),
                  bg="#222244", fg="#aaaacc", activebackground="#333366",
                  relief="flat", padx=6, pady=3,
                  command=lambda: self.on_tv_vol and self.on_tv_vol("down")
                  ).pack(side="left", padx=2)

        tk.Button(tv_row, text="VOL+", font=("Helvetica", 9, "bold"),
                  bg="#222244", fg="#aaaacc", activebackground="#333366",
                  relief="flat", padx=6, pady=3,
                  command=lambda: self.on_tv_vol and self.on_tv_vol("up")
                  ).pack(side="left", padx=2)

        self.tv_vol_label = tk.Label(tv_row, text="", font=("Helvetica", 9),
                                     fg="#aaaacc", bg="#1a1a2e")
        self.tv_vol_label.pack(side="left", padx=(8, 0))

        # bottom: transcript log
        bot = tk.Frame(self.root, bg="#1a1a2e")
        bot.pack(fill="x", padx=20, pady=(5, 15))

        log_header = tk.Label(
            bot, text="TRANSCRIPT", font=("Helvetica", 10, "bold"),
            fg="#666666", bg="#1a1a2e", anchor="w"
        )
        log_header.pack(fill="x")

        log_container = tk.Frame(bot, bg="#0d0d1a", highlightbackground="#333355", highlightthickness=1)
        log_container.pack(fill="x", pady=(3, 0))

        self.log_text = tk.Text(
            log_container, height=10, bg="#0d0d1a", fg="#cccccc",
            font=("Courier", 10), wrap="word", state="disabled",
            borderwidth=0, padx=8, pady=5, insertbackground="#cccccc",
            selectbackground="#333366"
        )
        self.log_text.pack(fill="x")

        self.log_text.tag_configure("prompt_num", foreground="#00ccff")
        self.log_text.tag_configure("you", foreground="#44cc44")
        self.log_text.tag_configure("ai", foreground="#ffaa44")
        self.log_text.tag_configure("jarvis", foreground="#cc88ff")
        self.log_text.tag_configure("error", foreground="#ff4444")

    def _toggle_mic(self):
        self.mic_muted = not self.mic_muted
        if self.mic_muted:
            self.mic_btn.config(text="MIC OFF", bg="#4a1a1a", fg="#cc4444", activebackground="#6a2a2a")
        else:
            self.mic_btn.config(text="MIC ON", bg="#1a4a1a", fg="#44cc44", activebackground="#2a6a2a")
        if self.on_mic_toggle:
            self.on_mic_toggle(self.mic_muted)

    def _draw_tv_btn(self):
        c = self.tv_btn
        c.delete("all")
        if self.tv_power_on:
            c.create_rectangle(2, 2, 48, 28, fill="#00cc44", outline="#00ff66", width=2)
            c.create_text(25, 15, text="ON", fill="white", font=("Helvetica", 10, "bold"))
        else:
            c.create_rectangle(2, 2, 48, 28, fill="#331111", outline="#444444", width=2)
            c.create_text(25, 15, text="OFF", fill="#555555", font=("Helvetica", 10, "bold"))

    def _on_tv_btn_click(self, event):
        if self.on_tv_toggle:
            self.on_tv_toggle()

    def set_tv_volume(self, level):
        self.tv_vol_label.config(text=f"Vol {level}")

    def set_tv_state(self, is_on):
        self.tv_power_on = is_on
        self._draw_tv_btn()
        self.tv_status.config(
            text="ON" if is_on else "OFF",
            fg="#00cc44" if is_on else "#555555"
        )

    def _get_weather_type(self, condition):
        condition = condition.lower().strip()
        for key in ["sunny", "clear"]:
            if key in condition:
                return "sun"
        if "partly" in condition:
            return "partly"
        for key in ["thunder", "storm"]:
            if key in condition:
                return "thunder"
        for key in ["rain", "drizzle", "shower"]:
            if key in condition:
                return "rain"
        for key in ["snow", "blizzard", "sleet"]:
            if key in condition:
                return "snow"
        for key in ["fog", "mist", "haze"]:
            if key in condition:
                return "fog"
        for key in ["cloud", "overcast"]:
            if key in condition:
                return "cloud"
        return "cloud"

    def _draw_weather_icon(self, wtype):
        c = self.weather_canvas
        c.delete("all")
        if wtype == "sun":
            # yellow sun circle with rays
            c.create_oval(20, 20, 50, 50, fill="#ffcc00", outline="#ffaa00", width=2)
            for angle in range(0, 360, 45):
                import math
                rad = math.radians(angle)
                x1 = 35 + 20 * math.cos(rad)
                y1 = 35 + 20 * math.sin(rad)
                x2 = 35 + 28 * math.cos(rad)
                y2 = 35 + 28 * math.sin(rad)
                c.create_line(x1, y1, x2, y2, fill="#ffcc00", width=2)
        elif wtype == "partly":
            # small sun behind cloud
            c.create_oval(10, 10, 35, 35, fill="#ffcc00", outline="#ffaa00", width=1)
            c.create_oval(20, 25, 45, 50, fill="#cccccc", outline="#aaaaaa", width=1)
            c.create_oval(30, 20, 60, 50, fill="#cccccc", outline="#aaaaaa", width=1)
            c.create_oval(15, 30, 50, 55, fill="#cccccc", outline="#aaaaaa", width=1)
        elif wtype == "cloud":
            c.create_oval(10, 20, 35, 45, fill="#888888", outline="#777777", width=1)
            c.create_oval(25, 10, 55, 45, fill="#999999", outline="#888888", width=1)
            c.create_oval(15, 30, 50, 55, fill="#888888", outline="#777777", width=1)
        elif wtype == "rain":
            # cloud with rain drops
            c.create_oval(10, 10, 35, 35, fill="#666688", outline="#555577", width=1)
            c.create_oval(25, 5, 55, 35, fill="#777799", outline="#666688", width=1)
            c.create_oval(15, 15, 50, 40, fill="#666688", outline="#555577", width=1)
            for x in [20, 30, 40]:
                c.create_line(x, 42, x - 3, 55, fill="#4488ff", width=2)
                c.create_line(x + 7, 45, x + 4, 58, fill="#4488ff", width=2)
        elif wtype == "thunder":
            c.create_oval(10, 5, 35, 30, fill="#555577", outline="#444466", width=1)
            c.create_oval(25, 0, 55, 30, fill="#666688", outline="#555577", width=1)
            c.create_oval(15, 10, 50, 35, fill="#555577", outline="#444466", width=1)
            # lightning bolt
            c.create_polygon(32, 30, 28, 45, 33, 45, 29, 62, 40, 40, 35, 40, 38, 30,
                             fill="#ffdd00", outline="#ffaa00")
        elif wtype == "snow":
            c.create_oval(10, 10, 35, 35, fill="#8888aa", outline="#7777aa", width=1)
            c.create_oval(25, 5, 55, 35, fill="#9999bb", outline="#8888aa", width=1)
            c.create_oval(15, 15, 50, 40, fill="#8888aa", outline="#7777aa", width=1)
            for x, y in [(20, 48), (30, 52), (40, 46), (25, 58), (35, 60)]:
                c.create_text(x, y, text="*", fill="white", font=("Helvetica", 10, "bold"))
        elif wtype == "fog":
            for i, y in enumerate([15, 25, 35, 45, 55]):
                w = 50 - i * 2
                c.create_line(10, y, 10 + w, y, fill="#999999", width=3, dash=(6, 4))

    def update_weather(self, wtype, temp, desc, sunrise="", sunset=""):
        self._draw_weather_icon(wtype)
        self.weather_temp.config(text=temp)
        self.weather_desc.config(text=desc)
        if sunrise and sunset:
            # strip seconds from times like "06:30:00" -> "6:30 AM"
            try:
                from datetime import datetime
                sr = datetime.strptime(sunrise.strip(), "%H:%M:%S").strftime("%-I:%M %p")
                ss = datetime.strptime(sunset.strip(), "%H:%M:%S").strftime("%-I:%M %p")
            except Exception:
                sr, ss = sunrise.strip(), sunset.strip()
            self.weather_sun.config(text=f"^ {sr}  v {ss}")
        self.weather_updated.config(text=f"Updated {time.strftime('%I:%M %p')}")

    def _update_clock(self):
        self.clock_label.config(text=time.strftime("%I:%M:%S %p"))
        self.date_label.config(text=time.strftime("%A, %B %d, %Y"))
        self.root.after(1000, self._update_clock)

    def _animate_circle(self):
        # pulse size based on audio level (idle/listening only)
        if self.state in ("idle", "listening", "wake"):
            level = min(self.audio_level, 1.0)
            # base radius 65, max expand 20px
            expand = int(level * 20)
            cx, cy = 80, 80
            r = 65 + expand
            self.canvas.coords(self.circle, cx - r, cy - r, cx + r, cy + r)
        else:
            self.canvas.coords(self.circle, 10, 10, 150, 150)
        self.root.after(50, self._animate_circle)

    def set_state(self, state):
        self.state = state
        colors = {
            "loading":    ("#2a2a3a", "#555555", "#666666", "Loading..."),
            "idle":       ("#1a4a1a", "#22aa22", "#44aa44", "Say 'Hey Jarvis'"),
            "wake":       ("#4a4a1a", "#aacc00", "#ccee22", "Wake detected"),
            "listening":  ("#1a2a5a", "#2288ff", "#44aaff", "Listening..."),
            "processing": ("#3a1a5a", "#aa44ff", "#cc66ff", "Processing..."),
            "thinking":   ("#4a3a1a", "#dd8800", "#ffaa22", "Thinking..."),
            "speaking":   ("#5a1a2a", "#ff4466", "#ff6688", "Speaking..."),
        }
        fill, outline, text_color, label = colors.get(state, ("#2a2a3a", "#555555", "#666666", state))
        self.canvas.itemconfig(self.circle, fill=fill, outline=outline, width=4)
        self.state_label.config(text=label, fg=text_color)
        self.status_text.config(text=state.upper(), fg=text_color)

    def log(self, text, tag=""):
        self.log_text.config(state="normal")
        if tag:
            self.log_text.insert("end", text + "\n", tag)
        else:
            self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def log_prompt(self, you_said, ai_parsed="", jarvis_said=""):
        self.prompt_count += 1
        self.log(f"--- Prompt {self.prompt_count} ---", "prompt_num")
        if you_said:
            self.log(f"  You: {you_said}", "you")
        if ai_parsed:
            self.log(f"  AI:  {ai_parsed}", "ai")
        if jarvis_said:
            self.log(f"  Jarvis: {jarvis_said}", "jarvis")

    def log_jarvis(self, text):
        self.log(f"  Jarvis: {text}", "jarvis")

    def add_timer_widget(self, num, name, total_seconds):
        frame = tk.Frame(self.timer_frame, bg="#222244", padx=10, pady=5)
        frame.pack(fill="x", pady=2)

        name_label = tk.Label(
            frame, text=f"#{num} {name}", font=("Helvetica", 12, "bold"),
            fg="#e0e0e0", bg="#222244", anchor="w"
        )
        name_label.pack(fill="x")

        time_label = tk.Label(
            frame, text="", font=("Helvetica", 20, "bold"),
            fg="#00ccff", bg="#222244", anchor="w"
        )
        time_label.pack(fill="x")

        self.timer_widgets[num] = {
            "frame": frame,
            "name_label": name_label,
            "time_label": time_label,
            "remaining": total_seconds,
        }

        self.no_timers_label.pack_forget()
        self._tick_timer(num)

    def _tick_timer(self, num):
        if num not in self.timer_widgets:
            return
        w = self.timer_widgets[num]
        r = w["remaining"]

        if r <= 0:
            w["time_label"].config(text="DONE!", fg="#00ff00")
            self.root.after(3000, lambda: self.remove_timer_widget(num))
            if self.on_timer_done:
                threading.Thread(target=self.on_timer_done, args=(num,), daemon=True).start()
            return

        mins, secs = divmod(r, 60)
        hrs, mins = divmod(mins, 60)
        if hrs:
            w["time_label"].config(text=f"{hrs}:{mins:02d}:{secs:02d}")
        else:
            w["time_label"].config(text=f"{mins}:{secs:02d}")

        w["remaining"] -= 1
        self.root.after(1000, lambda: self._tick_timer(num))

    def update_timer_name(self, num, new_name):
        if num in self.timer_widgets:
            self.timer_widgets[num]["name_label"].config(text=f"#{num} {new_name}")

    def add_time_to_widget(self, num, extra):
        if num in self.timer_widgets:
            self.timer_widgets[num]["remaining"] += extra

    def remove_timer_widget(self, num):
        if num in self.timer_widgets:
            self.timer_widgets[num]["frame"].destroy()
            del self.timer_widgets[num]
        if not self.timer_widgets:
            self.no_timers_label.pack(anchor="w")


# -----------------------------
# ASSISTANT ENGINE
# -----------------------------

class AssistantEngine:

    def __init__(self, gui):
        self.gui = gui
        self.gui.on_timer_done = self._on_timer_done
        self.gui.on_tv_toggle = self._tv_toggle
        self.gui.on_tv_vol = self._tv_vol
        self.gui.on_mic_toggle = self._on_mic_toggle
        self.wake_triggered = False
        self.last_wake_time = 0
        self.stream = None
        self.timer_count = 0
        self.active_timers = {}
        self.busy = False
        self.prompt_num = 0
        self.mic_muted = False
        self.tv_on = False

    def _on_timer_done(self, num):
        if num in self.active_timers:
            name = self.active_timers.pop(num)["name"]
            self.speak(f"{name} done")

    def load_models(self):
        self._gui(self.gui.log, "Loading Whisper...", "ai")
        self.whisper = WhisperModel("base", compute_type="int8")

        self._gui(self.gui.log, "Loading wake word model...", "ai")
        self.wake_model = Model()

        self._gui(self.gui.log, "Warming up Gemma2...", "ai")
        try:
            requests.post(OLLAMA_URL, json={
                "model": "gemma2:2b", "prompt": "hi", "stream": False,
                "options": {"num_predict": 1}
            }, timeout=120)
        except Exception:
            self._gui(self.gui.log, "Warning: Gemma2 warmup failed", "error")

        self._gui(self.gui.log, "Ready!", "jarvis")
        self.play_tone()
        self._gui(self.gui.set_state, "idle")
        self._gui(self._weather_loop)
        # sync TV button with actual state
        tv_state = self._tv_get_state()
        self.tv_on = tv_state
        self._gui(self.gui.set_tv_state, tv_state)
        self._update_tv_volume()

    def _fetch_weather(self):
        try:
            result = subprocess.run(
                ["curl", "-s", "wttr.in/Ocala+FL?format=%C|%t|%S|%s&u"],
                capture_output=True, text=True, timeout=10
            )
            parts = result.stdout.strip().split("|")
            if len(parts) >= 2:
                condition = parts[0].strip()
                temp = parts[1].strip().replace("+", "")
                sunrise = parts[2].strip() if len(parts) > 2 else ""
                sunset = parts[3].strip() if len(parts) > 3 else ""
                wtype = self.gui._get_weather_type(condition)
                self._gui(self.gui.update_weather, wtype, temp, condition, sunrise, sunset)
        except Exception:
            pass

    def _weather_loop(self):
        threading.Thread(target=self._fetch_weather, daemon=True).start()
        self.gui.root.after(20 * 60 * 1000, self._weather_loop)

    def play_tone(self):
        subprocess.run(
            "python3 -c '"
            "import numpy as np,sys;"
            "t=np.linspace(0,0.15,int(22050*0.15));"
            "tone=np.concatenate([np.sin(2*np.pi*880*t)*0.3,np.sin(2*np.pi*1320*t)*0.3]);"
            "sys.stdout.buffer.write((tone*32767).astype(np.int16).tobytes())"
            "' | aplay -r 22050 -f S16_LE -t raw -D plughw:0,0 2>/dev/null",
            shell=True
        )

    def _gui(self, func, *args):
        """Thread-safe GUI call."""
        self.gui.root.after(0, lambda: func(*args))

    def speak(self, text):
        self._gui(self.gui.set_state, "speaking")
        piper = subprocess.Popen(
            ["piper", "--model", VOICE_MODEL, "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        aplay = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-D", "plughw:0,0"],
            stdin=piper.stdout, stderr=subprocess.DEVNULL
        )
        piper.stdin.write(text.encode())
        piper.stdin.close()
        piper.stdout.close()
        aplay.wait()

    def record_command(self):
        self._gui(self.gui.set_state, "listening")

        chunks = []
        silent_chunks = 0
        heard_speech = False
        max_silent = 8
        max_chunks = 80

        with sd.InputStream(device=MIC_DEVICE, samplerate=SAMPLE_RATE, blocksize=3840, channels=1) as mic:
            for i in range(max_chunks):
                data, _ = mic.read(3840)
                chunks.append(data[:, 0].copy())

                # skip silence detection for first few chunks (noisy startup)
                if i < 3:
                    continue

                level = np.mean(np.abs(data))
                self.gui.audio_level = min(float(level * MIC_GAIN * 5), 1.0)
                if level > 0.002:
                    heard_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if heard_speech and silent_chunks >= max_silent:
                    break

        if not chunks or not heard_speech:
            return ""

        audio = np.concatenate(chunks)
        audio_int16 = np.clip(audio * MIC_GAIN * 32767, -32768, 32767).astype(np.int16)
        wav.write("/tmp/jarvis_input.wav", SAMPLE_RATE, audio_int16)

        self._gui(self.gui.set_state, "processing")
        segments, _ = self.whisper.transcribe("/tmp/jarvis_input.wav")

        text = ""
        for s in segments:
            text += s.text

        return text.strip().lower()

    # --- AI ---

    CLASSIFY_SYSTEM = '''Parse the voice command into JSON. Reply with ONLY valid JSON, no other text.
Do NOT calculate seconds yourself. Put the raw number and unit.
Available actions:

Set NEW timer: {"action":"timer","amount":30,"unit":"minute","name":"Take cat out"}
Cancel timer: {"action":"cancel_timer","number":1}
Add time to EXISTING timer: {"action":"add_time","number":1,"amount":5,"unit":"minute"}
Rename timer: {"action":"rename_timer","number":1,"name":"New name"}
Open website: {"action":"browse","url":"youtube.com"}
Search google: {"action":"search","query":"best pizza near me"}
Get time: {"action":"time"}
Get date: {"action":"date"}
Get weather: {"action":"weather"}
TV on: {"action":"tv","command":"on"}
TV off: {"action":"tv","command":"off"}
TV volume up: {"action":"tv","command":"volume_up","amount":5}
TV volume down: {"action":"tv","command":"volume_down","amount":5}
Set TV volume to specific level: {"action":"tv","command":"volume_set","amount":50}
TV mute: {"action":"tv","command":"mute"}
TV unmute: {"action":"tv","command":"unmute"}
General question: {"action":"question"}

RULES:
- "number" for add_time/cancel/rename = the timer NUMBER from the active list, NOT the amount.
- "set/start a timer" or "remind me to X in Y" = NEW timer action.
- "add X to [timer name]" or "X more to [timer name]" = add_time to existing timer.
- Only use add_time if the user specifically references an existing timer by name or number.
- For rename: match timer by name or number. "name" field is the NEW name.'''

    def ollama_classify(self, text):
        system = self.CLASSIFY_SYSTEM
        if self.active_timers:
            system += "\n\nCurrently active timers:"
            for num, info in self.active_timers.items():
                system += f'\n  Timer {num}: "{info["name"]}"'

        try:
            r = requests.post(OLLAMA_URL, json={
                "model": "gemma2:2b", "prompt": text,
                "system": system, "stream": False,
                "format": "json", "context": [],
                "options": {"temperature": 0.1, "num_predict": 60},
                "keep_alive": 0
            }, timeout=60)
            result = r.json().get("response", "").strip()
            return json.loads(result), result
        except Exception as e:
            return None, str(e)

    def _date_context(self):
        from datetime import date, timedelta
        today = date.today()
        year = today.year

        holidays = {
            "christmas": date(year, 12, 25),
            "new years": date(year + 1, 1, 1),
            "valentines day": date(year, 2, 14),
            "halloween": date(year, 10, 31),
            "independence day": date(year, 7, 4),
        }

        lines = [f"Today is {today.strftime('%A, %B %d, %Y')}, current time is {time.strftime('%I:%M %p')}."]
        for name, d in holidays.items():
            delta = (d - today).days
            if delta < 0:
                # already passed this year, use next year
                d = d.replace(year=year + 1)
                delta = (d - today).days
            lines.append(f"Days until {name}: {delta}")

        return " ".join(lines)

    def ollama_answer(self, question):
        context = self._date_context()
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": "gemma2:2b", "prompt": question,
                "system": f"{context} Answer in one or two short sentences. Use the facts above for any date/time questions.",
                "stream": False, "context": [],
                "options": {"temperature": 0.3, "num_predict": 100}
            }, timeout=30)
            answer = r.json().get("response", "").strip()
        except Exception:
            answer = ""
        answer = re.sub(r'[*#`]', '', answer)
        return answer if answer else "I could not come up with an answer"

    # --- time parsing ---

    def parse_time_from_text(self, text):
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
            before = text[:unit_pos].strip()
            digit_match = re.search(r'(\d+)\s*$', before)
            if digit_match:
                return int(digit_match.group(1))
            for word, num in sorted(word_to_num.items(), key=lambda x: -len(x[0])):
                if before.endswith(word):
                    return num
            return None

        total = 0
        found = False
        for unit_match in re.finditer(r'(seconds?|minutes?|hours?)', text):
            num = find_number_before(text, unit_match.start())
            if num is not None:
                unit = unit_match.group(1).rstrip("s")
                base = multiplier.get(unit, 1)
                total += num * base
                found = True
                # check for "and a half" after the unit
                after = text[unit_match.end():].strip()
                if after.startswith("and a half") or after.startswith("and half"):
                    total += base // 2
            else:
                # check for "half" before unit: "half a minute"
                before = text[:unit_match.start()].strip()
                if before.endswith("half a") or before.endswith("half"):
                    unit = unit_match.group(1).rstrip("s")
                    total += multiplier.get(unit, 1) // 2
                    found = True

        # handle "X and a half minutes" where "and a half" comes before unit
        half_match = re.search(r'(\d+|' + '|'.join(sorted(word_to_num.keys(), key=lambda x: -len(x))) + r')\s+and\s+a\s+half\s+(seconds?|minutes?|hours?)', text)
        if half_match and not found:
            num_str = half_match.group(1)
            num = int(num_str) if num_str.isdigit() else word_to_num.get(num_str)
            unit = half_match.group(2).rstrip("s")
            if num is not None:
                base = multiplier.get(unit, 1)
                total = num * base + base // 2
                found = True

        return total if found else None

    def amount_to_seconds(self, amount, unit):
        multiplier = {"second": 1, "minute": 60, "hour": 3600}
        unit = str(unit).rstrip("s").lower()
        return int(amount) * multiplier.get(unit, 1)

    # --- timer management ---

    def start_timer(self, seconds, name=None):
        self.timer_count += 1
        num = self.timer_count
        if not name:
            name = f"Timer {num}"

        self.active_timers[num] = {"name": name, "seconds": seconds}
        self._gui(self.gui.add_timer_widget, num, name, seconds)
        self.speak(f"{name} set")

    def rename_timer(self, num, new_name):
        if num not in self.active_timers:
            self.speak(f"Timer {num} not found")
            return
        self.active_timers[num]["name"] = new_name
        self.gui.root.after(0, lambda: self.gui.update_timer_name(num, new_name))
        self.speak(f"Timer {num} renamed to {new_name}")

    def cancel_timer(self, num):
        if num not in self.active_timers:
            self.speak(f"Timer {num} not found")
            return
        name = self.active_timers[num]["name"]
        self.active_timers.pop(num, None)
        self.gui.root.after(0, lambda: self.gui.remove_timer_widget(num))
        self.speak(f"{name} cancelled")

    def add_time_to_timer(self, num, extra_seconds):
        if num not in self.active_timers:
            self.speak(f"Timer {num} not found")
            return
        name = self.active_timers[num]["name"]
        self.gui.root.after(0, lambda: self.gui.add_time_to_widget(num, extra_seconds))
        mins = extra_seconds // 60
        secs = extra_seconds % 60
        if mins and secs:
            self.speak(f"Added {mins} minutes and {secs} seconds to {name}")
        elif mins:
            self.speak(f"Added {mins} minutes to {name}")
        else:
            self.speak(f"Added {secs} seconds to {name}")

    # --- command handler ---

    def handle_command(self, text):
        if text == "":
            self.speak("I did not hear anything")
            return

        # show what you said immediately
        self.prompt_num += 1
        pc = self.prompt_num
        self._gui(self.gui.log, f"--- Prompt {pc} ---", "prompt_num")
        self._gui(self.gui.log, f"  You: {text}", "you")

        self._gui(self.gui.set_state, "processing")
        cmd, raw = self.ollama_classify(text)

        self._gui(self.gui.log, f"  AI:  {raw}", "ai")

        if not cmd or "action" not in cmd:
            self._gui(self.gui.log_jarvis, "I did not understand")
            self.speak("I did not understand")
            return

        action = cmd["action"]

        # post-classification fix: if AI said "timer" but text references an existing timer name,
        # it's probably "add time" not "new timer"
        if action == "timer" and self.active_timers:
            cmd_name = (cmd.get("name") or "").lower()
            for num, info in self.active_timers.items():
                timer_name = info["name"].lower()
                # check if AI-returned name matches an active timer, or timer name appears in spoken text
                if (cmd_name and (cmd_name in timer_name or timer_name in cmd_name)) or \
                   (timer_name != f"timer {num}" and timer_name in text.lower()):
                    action = "add_time"
                    cmd["action"] = "add_time"
                    cmd["number"] = num
                    break

        if action == "timer":
            name = cmd.get("name")
            secs = self.parse_time_from_text(text)
            if not secs:
                amount = cmd.get("amount")
                unit = cmd.get("unit", "second")
                if amount and unit:
                    secs = self.amount_to_seconds(amount, unit)
                else:
                    secs = 60
            self.start_timer(secs, name)
            self._gui(self.gui.log_jarvis, f"Timer set: {name or 'Timer'} for {secs}s")

        elif action == "cancel_timer":
            num = cmd.get("number")
            if num:
                self.cancel_timer(int(num))
            elif len(self.active_timers) == 1:
                self.cancel_timer(next(iter(self.active_timers)))
            else:
                self.speak("Which timer?")

        elif action == "add_time":
            num = cmd.get("number")
            secs = self.parse_time_from_text(text)
            if not secs:
                amount = cmd.get("amount")
                unit = cmd.get("unit", "second")
                secs = self.amount_to_seconds(amount, unit) if amount else None
            if not num and len(self.active_timers) == 1:
                num = next(iter(self.active_timers))
            if num and secs:
                self.add_time_to_timer(int(num), secs)
            else:
                self.speak("Which timer and how much time?")

        elif action == "rename_timer":
            num = cmd.get("number")
            name = cmd.get("name")
            if num and name:
                self.rename_timer(int(num), name)
            else:
                self.speak("Which timer and what name?")

        elif action == "browse":
            url = cmd.get("url", "")
            if url:
                if not url.startswith("http"):
                    url = "https://" + url
                webbrowser.open(url)
                self.speak(f"Opening {cmd.get('url')}")
                self._gui(self.gui.log_jarvis, f"Opening {url}")
            else:
                self.speak("What website?")

        elif action == "search":
            query = cmd.get("query", "")
            if query:
                webbrowser.open(f"https://www.google.com/search?q={query}")
                self.speak(f"Searching for {query}")
                self._gui(self.gui.log_jarvis, f"Searching: {query}")
            else:
                self.speak("What should I search for?")

        elif action == "time":
            t = time.strftime("The time is %I %M %p")
            self.speak(t)
            self._gui(self.gui.log_jarvis, t)

        elif action == "date":
            d = time.strftime("Today is %A, %B %d, %Y")
            self.speak(d)
            self._gui(self.gui.log_jarvis, d)

        elif action == "weather":
            try:
                result = subprocess.run(
                    ["curl", "-s", "wttr.in/Ocala+FL?format=%C,+%t,+%h+humidity,+wind+%w&u"],
                    capture_output=True, text=True, timeout=5
                )
                weather = result.stdout.strip()
                weather = weather.replace("+", "").replace("\u00b0C", " degrees")
                weather = re.sub(r'[\u2191\u2193\u2190\u2192\u2199\u2198\u2197\u2196]', '', weather)
                self.speak(f"Currently {weather}")
                self._gui(self.gui.log_jarvis, f"Weather: {weather}")
            except Exception:
                self.speak("I could not get the weather")

        elif action == "tv":
            self.handle_tv(cmd)

        elif action == "question":
            self._gui(self.gui.set_state, "thinking")
            answer = self.ollama_answer(text)
            self.speak(answer)
            self._gui(self.gui.log_jarvis, answer)

        else:
            self._gui(self.gui.set_state, "thinking")
            answer = self.ollama_answer(text)
            self.speak(answer)
            self._gui(self.gui.log_jarvis, answer)

    # --- mic mute ---

    def _on_mic_toggle(self, muted):
        self.mic_muted = muted

    # --- TV control ---

    def _tv_toggle(self):
        threading.Thread(target=self._tv_toggle_worker, daemon=True).start()

    def _get_tv_volume(self):
        try:
            result = subprocess.run(
                ["pyvizio", f"--ip={TV_IP}", f"--auth={TV_AUTH}", "--device_type=tv", "get-volume-level"],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            import re
            m = re.search(r'current volume[:\s]+(\d+)', output, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception as e:
            print(f"get_tv_volume error: {e}")
        return None

    def _update_tv_volume(self):
        try:
            level = self._get_tv_volume()
            if level is not None:
                self._gui(self.gui.set_tv_volume, str(level))
        except Exception:
            pass

    def _tv_get_state(self):
        """Query actual TV power state."""
        try:
            result = subprocess.run(
                ["pyvizio", f"--ip={TV_IP}", f"--auth={TV_AUTH}", "--device_type=tv", "get-power-state"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout + result.stderr
            return "is on" in output.lower()
        except Exception:
            return self.tv_on

    def _tv_toggle_worker(self):
        current = self._tv_get_state()
        command = "off" if current else "on"
        self.handle_tv({"command": command}, silent=True)
        self._update_tv_volume()

    def _tv_vol(self, direction):
        threading.Thread(target=lambda: self.handle_tv({"command": f"volume_{direction}", "amount": 3}, silent=True), daemon=True).start()

    def handle_tv(self, cmd, silent=False):
        command = cmd.get("command", "")
        amount = int(cmd.get("amount", 5))

        def run(*args):
            return subprocess.run(
                ["pyvizio", f"--ip={TV_IP}", f"--auth={TV_AUTH}", "--device_type=tv"] + list(args),
                capture_output=True, text=True
            )

        try:
            if command == "on":
                run("power", "on")
                self.tv_on = True
                self._gui(self.gui.set_tv_state, True)
                if not silent: self.speak("Turning TV on")
            elif command == "off":
                run("power", "off")
                self.tv_on = False
                self._gui(self.gui.set_tv_state, False)
                if not silent: self.speak("Turning TV off")
            elif command == "volume_set":
                current = self._get_tv_volume()
                if current is None:
                    if not silent: self.speak("Could not read TV volume")
                else:
                    diff = amount - current
                    print(f"  Volume set: current={current} target={amount} diff={diff}")
                    if diff > 0:
                        run("volume", "up", str(diff))
                    elif diff < 0:
                        run("volume", "down", str(abs(diff)))
                    if not silent: self.speak(f"Volume set to {amount}")
                    self._update_tv_volume()
            elif command == "volume_up":
                run("volume", "up", str(amount))
                if not silent: self.speak("Volume up")
                self._update_tv_volume()
            elif command == "volume_down":
                run("volume", "down", str(amount))
                if not silent: self.speak("Volume down")
                self._update_tv_volume()
            elif command == "mute":
                run("mute", "on")
                if not silent: self.speak("TV muted")
            elif command == "unmute":
                run("mute", "off")
                if not silent: self.speak("TV unmuted")
            else:
                if not silent: self.speak("I don't know that TV command")
            self._gui(self.gui.log_jarvis, f"TV: {command}")
        except Exception as e:
            self.speak("Could not reach the TV")
            self._gui(self.gui.log, f"TV error: {e}", "error")

    # --- wake word ---

    def wake_callback(self, indata, frames, time_info, status):
        if self.mic_muted:
            return
        audio = indata[:, 0]

        # update audio level for circle animation (normalized 0-1)
        level = float(np.mean(np.abs(audio)) * MIC_GAIN * 5)
        self.gui.audio_level = min(level, 1.0)

        audio = audio * MIC_GAIN
        audio = resample(audio, int(len(audio) / 3))
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        prediction = self.wake_model.predict(audio_int16)
        score = prediction.get("hey_jarvis", 0)

        if score > WAKE_THRESHOLD and time.time() - self.last_wake_time > 3:
            self.last_wake_time = time.time()
            self.wake_triggered = True

    def start_stream(self):
        self.stream = sd.InputStream(
            device=MIC_DEVICE, samplerate=SAMPLE_RATE,
            blocksize=3840, channels=1, callback=self.wake_callback
        )
        self.stream.start()

    def _handle_wake(self):
        self.stream.stop()
        self.stream.close()
        self.wake_model.reset()
        self.gui.audio_level = 0.0

        self._gui(self.gui.set_state, "wake")
        time.sleep(0.3)

        self.speak("Yes")

        text = self.record_command()
        self.handle_command(text)

        self.wake_model.reset()
        self.last_wake_time = time.time()
        self._gui(self.gui.set_state, "idle")
        self.start_stream()
        self.busy = False

    def run_loop(self):
        if self.wake_triggered and not self.busy:
            self.wake_triggered = False
            self.busy = True
            threading.Thread(target=self._handle_wake, daemon=True).start()

        self.gui.root.after(100, self.run_loop)


# -----------------------------
# MAIN
# -----------------------------

def main():
    root = tk.Tk()
    gui = AssistantGUI(root)
    engine = AssistantEngine(gui)

    def init():
        engine.load_models()
        engine.start_stream()
        # schedule run_loop on main thread
        root.after(0, engine.run_loop)

    threading.Thread(target=init, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
