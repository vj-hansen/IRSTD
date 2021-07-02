

import tkinter as tk
import tkinter.ttk
import subprocess
from tkinter import *
import threading
import time


def run_cmd_1():
    try:
        t1 = threading.Thread(target=subprocess.run(["python3", "data_driven_method/test_model.py"],stdout=subprocess.PIPE)).start()
    except Exception as e:
        raise
    else:
        pass
    finally:
        pass


def run_cmd_2():
    try:
        res = subprocess.run(["python3", "model_driven_method/main.py"], stdout=subprocess.PIPE)
        txtBox = tk.Text(app, relief=RIDGE, borderwidth=2)
        txtBox.grid()
        txtBox.insert('end', res.stdout)
    except Exception as e:
        raise
    else:
        pass
    finally:
        pass

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IRSTD')
        self.createButtons()
      #  self.createText()

    #def createText(self, event=None):
       # txtBox = tk.Text(self, height=2, width=30, relief=RIDGE, borderwidth=2)
       # txtBox.grid()

    def createButtons(self, event=None):
        headingLabel = tk.Label(self, text="Select one:")
        headingLabel.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="w")
        tkinter.ttk.Separator(self, orient="horizontal").grid(row=1, column=0, columnspan=5, sticky='ew')

        Time = tk.Frame(self)
        Time.grid(row=5, column=0, columnspan=3)

        tk.Button(Time,
                text="Data Driven",
                fg="blue",
                command = run_cmd_1).grid(row=4, column=1)
        tk.Button(Time,
                text="Model Driven",
                fg="blue",
                command = run_cmd_2).grid(row=4, column=2)
        tk.Button(Time,
                text="Quit",
                fg="Red",
                command = self.destroy).grid(row=4, column=3)


app = Application()
app.mainloop()