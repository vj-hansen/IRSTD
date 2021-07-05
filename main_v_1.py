

import tkinter as tk
import tkinter.ttk
import subprocess
import threading
import time
from tkinter.filedialog import askdirectory

# create function for killing processes
# run: ps a
# run: kill -9 <all PID related to python3 model_driven..>

def run_cmd_1(loc):
    try:
        print(app.get_loc())
        #t1 = threading.Thread(
        ##    target=subprocess.run(["python3", "data_driven_method/test_model.py", "--path", loc],
         #   stdout=subprocess.PIPE)).start()
    except Exception as e:
        raise
    else:
        pass
    finally:
        pass


def run_cmd_2(loc):
    loc = app.get_loc()
    try:
        #res = subprocess.run(["python3", "model_driven_method/main.py", "--path", loc],
        #    stdout=subprocess.PIPE)
        subprocess.run("cal")
        print(loc)
        #res = subprocess.run(["python3", "model_driven_method/main.py", "--path", loc])
       # txtBox = tk.Text(app, relief=tk.RIDGE, borderwidth=2)
       # txtBox.grid()
       # txtBox.insert('end', res.stdout)
    except Exception as e:
        raise
    else:
        pass
    finally:
        pass

WINDOW_W = 1024
WINDOW_H = 768

class Application(tk.Tk):
    def __init__(self, location=""):
        super().__init__()
        self._location = location
        self.geometry(str(WINDOW_W)+'x'+str(WINDOW_H))
        self.title('IRSTD')
        self.createButtons()

    def get_loc(self):
        return self._location

    def set_loc(self, x):
        self._location = x

    def select_v_path(self):
        self.location = askdirectory()
        self.set_loc(self.location)
        if self.video_path.get() != "":
            self.video_path.delete(0, tk.END)
            self.video_path.insert(tk.END, self.location)
        else:
            self.video_path.insert(tk.END, self.location)

    def createButtons(self, event=None):
        headingLabel = tk.Label(self, text="Select one:")
        headingLabel.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="w")
        tkinter.ttk.Separator(self, orient="horizontal").grid(row=1, column=0, columnspan=5, sticky='ew')

        frame1 = tk.LabelFrame(
                self, 
                text="model_driven_method",
                width=WINDOW_W/2, 
                height=WINDOW_H/2)
        frame1.place(relx=0.2, rely=0.2)

        self.v_path = tk.Label(frame1, text="Select Path")
        self.v_path.place(relx=0.05, rely=0.45)


        self.video_path = tk.Entry(
                frame1, width=6, relief=tk.SUNKEN)
        self.video_path.place(relx=0.1, rely=0.55)

        self.file = tk.Button(
                frame1, 
                text="Browse",
                command=self.select_v_path)
        self.file.place(relx=0.6, rely=0.55)


        Time = tk.LabelFrame(
                self, 
                text="Frame2",
                width=WINDOW_W/2, 
                height=WINDOW_H/2)
        Time.place(relx=0.6, rely=0.2)
        
        loc = self.get_loc()

        self.dd = tk.Button(
                Time, 
                text="Data\nDriven",
                fg="blue", 
                command=lambda : run_cmd_1(loc))

        self.dd.place(relx=0.1, rely=0.5)

        self.md = tk.Button(
                Time, 
                text="Model\nDriven",
                fg="blue", 
                command=lambda : run_cmd_2(loc))
        self.md.place(relx=0.1, rely=0.05)

        self.qt = tk.Button(
                Time,
                text="Quit",
                fg="Red",
                command=self.destroy)
        self.qt.place(relx=0.5, rely=0.1)

app = Application()
app.mainloop()