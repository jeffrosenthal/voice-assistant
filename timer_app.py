#!/usr/bin/env python3
import os
import sys
import tkinter as tk


seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
label = sys.argv[2] if len(sys.argv) > 2 else "Timer"
control_dir = sys.argv[3] if len(sys.argv) > 3 else None
remaining = seconds


def check_control():
    global remaining, label

    if not control_dir:
        root.after(500, check_control)
        return

    # check for name change
    name_file = os.path.join(control_dir, "name")
    try:
        with open(name_file, "r") as f:
            new_name = f.read().strip()
        if new_name and new_name != label:
            label = new_name
            title_label.config(text=label)
            root.title(label)
    except FileNotFoundError:
        pass

    # check for cancel
    cancel_file = os.path.join(control_dir, "cancel")
    if os.path.exists(cancel_file):
        root.destroy()
        return

    # check for added time
    add_file = os.path.join(control_dir, "add")
    try:
        with open(add_file, "r") as f:
            extra = int(f.read().strip())
        remaining += extra
        os.remove(add_file)
    except (FileNotFoundError, ValueError):
        pass

    root.after(500, check_control)


def update():
    global remaining

    if remaining <= 0:
        time_label.config(text="DONE!", fg="#00ff00")
        title_label.config(fg="#00ff00")
        root.after(5000, root.destroy)
        return

    mins, secs = divmod(remaining, 60)
    hrs, mins = divmod(mins, 60)

    if hrs:
        time_label.config(text=f"{hrs}:{mins:02d}:{secs:02d}")
    else:
        time_label.config(text=f"{mins}:{secs:02d}")

    remaining -= 1
    root.after(1000, update)


root = tk.Tk()
root.title(label)
root.configure(bg="black")
root.attributes("-topmost", True)
root.geometry("400x200")

title_label = tk.Label(root, text=label, font=("Helvetica", 24), fg="white", bg="black")
title_label.pack(pady=(20, 0))

time_label = tk.Label(root, text="", font=("Helvetica", 80, "bold"), fg="#00ccff", bg="black")
time_label.pack(expand=True)

update()
check_control()
root.mainloop()
