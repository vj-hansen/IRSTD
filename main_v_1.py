

import tkinter as tk
import tkinter.ttk
import subprocess


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IRSTD')
        self.createButtons()

    def createButtons(self, event=None):
        headingLabel = tk.Label(self, text="Select one:")
        headingLabel.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="w")
        tkinter.ttk.Separator(self, orient="horizontal").grid(row=1, column=0, columnspan=5, sticky='ew')

        Time = tk.Frame(self)
        Time.grid(row=5, column=0, columnspan=3)
        tk.Button(Time, text="Data Driven" ,fg="blue", command = self.run_cmd_1).grid(row=4, column=1)
        tk.Button(Time, text="Model Driven" ,fg="blue", command = self.run_cmd_2).grid(row=4, column=2)

        tk.Button(Time, text="Quit" ,fg="Red", command = self.destroy).grid(row=4, column=3)

    def run_cmd_1(self):
        try:
            subprocess.run(["python3", "data_driven_method/test_model.py"])
        except Exception as e:
            raise
        else:
            pass
        finally:
            pass
        

    def run_cmd_2(self):
        try:
            subprocess.run(["python3", "model_driven_method/main.py"])
        except Exception as e:
            raise
        else:
            pass
        finally:
            pass
        

app = Application()
app.mainloop()