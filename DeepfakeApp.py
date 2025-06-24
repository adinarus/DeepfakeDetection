import tkinter as tk
from tkinter import filedialog, messagebox
import os
from test import test_image, test_video
from classifiers import Meso4, MesoInception4, XceptionNet, F3NetClassifier

class DeepfakeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Tool")
        self.geometry("1000x700")

        # Set app grid to expand
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.model_path = None
        self.model_class = None
        self.model_mode = None

        self.main_menu = MainMenu(self)
        self.test_page = TestPage(self)

        self.show_page(self.main_menu)

    def show_page(self, page):
        page.tkraise()
        if hasattr(page, "refresh"):
            page.refresh()

class MainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=0, sticky="nsew")

        self.available_models = [
            {"name": "Meso4", "path": "test/Meso4.h5", "class": Meso4, "mode": "original"},
            {"name": "MesoInception4", "path": "test/MesoInception4.h5", "class": MesoInception4, "mode": "original"},
            {"name": "Xception", "path": "test/Xception.h5", "class": XceptionNet, "mode": "original"},
            {"name": "F3NetFAD", "path": "test/F3NetFAD.h5", "class": F3NetClassifier, "mode": "FAD"},
            {"name": "F3NetLFS", "path": "test/F3NetLFS.h5", "class": F3NetClassifier, "mode": "LFS"},
            {"name": "F3NetMix", "path": "test/F3NetMix.h5", "class": F3NetClassifier, "mode": "Mix"}
        ]

        self.selected_model_index = tk.IntVar(value=0)

        self.create_widgets()

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        center_frame = tk.Frame(self)
        center_frame.grid(row=0, column=0)
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(center_frame)
        inner.grid(row=0, column=0)

        tk.Label(inner, text="Select a model to use:", font=("Arial", 14)).pack(pady=10)

        for i, model in enumerate(self.available_models):
            tk.Radiobutton(inner, text=model["name"], variable=self.selected_model_index, value=i).pack(anchor="w")

        tk.Button(inner, text="Confirm Selection", width=25, height=2, command=self.set_model).pack(pady=20)
        tk.Button(inner, text="Test on Image / Video", width=30, height=2,
                  command=lambda: self.master.show_page(self.master.test_page)).pack(pady=10)

    def set_model(self):
        index = self.selected_model_index.get()
        model_info = self.available_models[index]
        self.master.model_path = model_info["path"]
        self.master.model_class = model_info["class"]
        self.master.model_mode = model_info["mode"]
        messagebox.showinfo("Model Selected", f"Model: {model_info['name']} selected.")

class TestPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=0, sticky="nsew")
        self.selected_path = None

        self.create_widgets()

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        back_button = tk.Button(self, text="Back", command=lambda: self.master.show_page(self.master.main_menu))
        back_button.place(x=10, y=10)

        center_frame = tk.Frame(self)
        center_frame.grid(row=0, column=0)
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(center_frame)
        inner.grid(row=0, column=0)

        self.file_label = tk.Label(inner, text="No file selected")
        self.file_label.pack(pady=5)

        tk.Button(inner, text="Select Image / Video", command=self.select_file).pack(pady=5)
        tk.Button(inner, text="Predict", command=self.predict_file).pack(pady=10)

        self.result_label = tk.Label(inner, text="", font=("Courier", 12), justify="left")
        self.result_label.pack(pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image/Video", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if path:
            self.selected_path = path
            self.file_label.config(text=os.path.basename(path))

    def predict_file(self):
        if not self.master.model_path:
            messagebox.showwarning("Model Missing", "Please select a model on the main page.")
            return
        if not self.selected_path:
            messagebox.showwarning("File Missing", "Please select an image or video file first.")
            return

        ext = os.path.splitext(self.selected_path)[1].lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            label, confidence = test_image(self.master.model_path, self.selected_path,
                                           model_class=self.master.model_class, mode=self.master.model_mode)
        elif ext in [".mp4", ".avi"]:
            label, confidence = test_video(self.master.model_path, self.selected_path,
                                           model_class=self.master.model_class, mode=self.master.model_mode)
        else:
            messagebox.showerror("Unsupported Format", "File type not supported.")
            return

        result_text = f"Prediction: {label}\nConfidence: {confidence*100:.2f}%"
        self.result_label.config(text=result_text)

if __name__ == '__main__':
    app = DeepfakeApp()
    app.mainloop()
