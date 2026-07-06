import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


# Чтение файлов
def read_image(path):
    img_array = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return img


class ImageCalibratorApp:
    # Инициализация приложения: настройка главного окна и объявление рабочих переменны
    def __init__(self, root):
        self.root = root
        self.root.title("Постобработка изображений")
        self.root.geometry("1100x750")

        self.dark_files = []
        self.flat_files = []
        self.target_file = ""

        # Данные калибровки
        self.master_dark = None
        self.flat_minus_dark = None
        self.mean_flat = None

        # Массивы для динамического изменения размера
        self.img_orig_array = None
        self.img_corr_array = None
        self.final_result = None

        # Переменная для таймера задержки
        self.resize_job = None

        self.setup_ui()

    # Интерфейс
    def setup_ui(self):
        # Панель управления
        control_frame = tk.Frame(self.root, width=280, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # --- Настройка ---
        tk.Label(control_frame, text="1. Калибровка", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(
            pady=(10, 5))

        self.btn_darks = tk.Button(control_frame, text="Загрузить Темновые кадры", command=self.load_darks)
        self.btn_darks.pack(fill=tk.X, pady=5)
        self.lbl_darks = tk.Label(control_frame, text="Загружено: 0", bg="#f0f0f0")
        self.lbl_darks.pack()

        self.btn_flats = tk.Button(control_frame, text="Загрузить Белые кадры", command=self.load_flats)
        self.btn_flats.pack(fill=tk.X, pady=5)
        self.lbl_flats = tk.Label(control_frame, text="Загружено: 0", bg="#f0f0f0")
        self.lbl_flats.pack()

        self.btn_calibrate = tk.Button(control_frame, text="Запомнить дефекты", command=self.calibrate, bg="#ff9800",
                                       fg="white", font=("Arial", 10, "bold"))
        self.btn_calibrate.pack(fill=tk.X, pady=10)

        tk.Frame(control_frame, height=2, bg="#cccccc").pack(fill=tk.X, pady=15)

        # --- Очистка ---
        tk.Label(control_frame, text="2. Очистка фото", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(5, 5))

        self.btn_target = tk.Button(control_frame, text="Загрузить Тестовый кадр", command=self.load_target,
                                    state=tk.DISABLED)
        self.btn_target.pack(fill=tk.X, pady=5)
        self.lbl_target = tk.Label(control_frame, text="Файл не выбран", bg="#f0f0f0", wraplength=200)
        self.lbl_target.pack()

        self.btn_process = tk.Button(control_frame, text="Удалить деффекты", command=self.process_image, bg="#4CAF50",
                                     fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_process.pack(fill=tk.X, pady=10)

        self.btn_save = tk.Button(control_frame, text="Сохранить результат", command=self.save_result,
                                  state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=5)

        # --- СИСТЕМА СТАТУСА ---
        tk.Frame(control_frame, height=2, bg="#cccccc").pack(fill=tk.X, pady=15)
        self.lbl_status = tk.Label(control_frame, text="Статус: Ожидание действий...", bg="#f0f0f0",
                                   font=("Arial", 10, "italic"), wraplength=250)
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

        # Панель изображений
        self.image_frame = tk.Frame(self.root, bg="#2c2c2c")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_orig_title = tk.Label(self.image_frame, text="Оригинал", bg="#2c2c2c", fg="white", font=("Arial", 12))
        self.lbl_orig_title.place(relx=0.25, rely=0.02, anchor=tk.N)

        self.lbl_corr_title = tk.Label(self.image_frame, text="Обработанное изображение", bg="#2c2c2c", fg="white",
                                       font=("Arial", 12))
        self.lbl_corr_title.place(relx=0.75, rely=0.02, anchor=tk.N)

        self.canvas_orig = tk.Label(self.image_frame, bg="black")
        self.canvas_orig.place(relx=0.02, rely=0.08, relwidth=0.46, relheight=0.88)

        self.canvas_corr = tk.Label(self.image_frame, bg="black")
        self.canvas_corr.place(relx=0.52, rely=0.08, relwidth=0.46, relheight=0.88)

        # Привязка изменения размера
        self.image_frame.bind("<Configure>", self.on_resize)

    # Функция обновления статуса
    def update_status(self, text, color="black"):
        self.lbl_status.config(text=f"Статус: {text}", fg=color)
        self.root.update_idletasks()

    # Защита от моргания
    def on_resize(self, event):
        if event.widget == self.image_frame:
            if self.resize_job:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(10, self.redraw_images)

    # Принудительная отрисовка картинок "Оригинал" и "Результат"
    def redraw_images(self):
        self.display_responsive(self.img_orig_array, self.canvas_orig)
        self.display_responsive(self.img_corr_array, self.canvas_corr)

    # Открытие окна выбора темновых кадров
    def load_darks(self):
        self.dark_files = filedialog.askopenfilenames(title="Выберите темновые кадры",
                                                      filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        count = len(self.dark_files)
        self.lbl_darks.config(text=f"Загружено: {count}")
        if count > 0:
            self.update_status(f"Загружено {count} Dark-кадров.", "blue")

    # Открытие окна выбора белых кадров
    def load_flats(self):
        self.flat_files = filedialog.askopenfilenames(title="Выберите белые кадры",
                                                      filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        count = len(self.flat_files)
        self.lbl_flats.config(text=f"Загружено: {count}")
        if count > 0:
            self.update_status(f"Загружено {count} Flat-кадров.", "blue")

    # Сложение и усреднение кадров
    def compute_master(self, file_paths):
        if not file_paths:
            return None
        first_img = read_image(file_paths[0]).astype(np.float32)
        master = np.zeros_like(first_img)

        for path in file_paths:
            img = read_image(path).astype(np.float32)
            master += img

        master /= len(file_paths)
        return master

    # Расчет эталонных дефектов
    def calibrate(self):
        if not self.dark_files or not self.flat_files:
            self.update_status("Отсутствуют кадры для калибровки!", "red")
            messagebox.showerror("Ошибка", "Сначала загрузите темновые и белые кадры!")
            return

        try:
            self.update_status("Создание мастер-кадров... Пожалуйста, подождите.", "#d2691e")  # Оранжевый цвет

            self.master_dark = self.compute_master(self.dark_files)
            master_flat = self.compute_master(self.flat_files)

            self.flat_minus_dark = master_flat - self.master_dark
            self.flat_minus_dark[self.flat_minus_dark <= 0] = 1.0
            self.mean_flat = np.mean(self.flat_minus_dark)

            self.btn_target.config(state=tk.NORMAL)
            self.update_status("Калибровка завершена. Ожидание тестового кадра.", "green")
            messagebox.showinfo("Готово", "Дефекты запомнены!\nТеперь вы можете загружать тестовые кадры.")

        except Exception as e:
            self.update_status("Ошибка калибровки!", "red")
            messagebox.showerror("Ошибка калибровки", str(e))

    # Загрузка изображения для очистки
    def load_target(self):
        self.target_file = filedialog.askopenfilename(title="Выберите тестовый кадр",
                                                      filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if self.target_file:
            self.lbl_target.config(text=os.path.basename(self.target_file))
            self.img_orig_array = read_image(self.target_file)
            self.redraw_images()
            self.btn_process.config(state=tk.NORMAL)
            self.update_status("Тестовый кадр загружен. Готов к очистке.", "blue")

    # Вычитание темнового шума и выравнивание освещенности
    def process_image(self):
        if not self.target_file or self.master_dark is None:
            return

        try:
            self.update_status("Удаление дефектов... Пожалуйста, подождите.", "#d2691e")

            target = self.img_orig_array.astype(np.float32)

            target_minus_dark = target - self.master_dark
            target_minus_dark[target_minus_dark < 0] = 0

            corrected = (target_minus_dark / self.flat_minus_dark) * self.mean_flat

            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            self.final_result = corrected
            self.img_corr_array = corrected

            self.redraw_images()
            self.btn_save.config(state=tk.NORMAL)

            self.update_status("Изображение успешно очищено.", "green")

        except Exception as e:
            self.update_status("Ошибка обработки!", "red")
            messagebox.showerror("Ошибка обработки", str(e))

    # Адаптивное изменение размера изображения под рамку окна
    def display_responsive(self, img_array, label_widget):
        if img_array is None:
            return

        w = label_widget.winfo_width()
        h = label_widget.winfo_height()

        if w < 10 or h < 10:
            return

        img_h, img_w = img_array.shape
        aspect = img_w / img_h

        if w / h > aspect:
            new_h = h
            new_w = int(aspect * h)
        else:
            new_w = w
            new_h = int(w / aspect)

        resized = cv2.resize(img_array, (new_w, new_h))

        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(resized_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    # Конвертация очищенного массива обратно в изображение и её сохранение на компьютер
    def save_result(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            try:
                final_rgb = cv2.cvtColor(self.final_result, cv2.COLOR_GRAY2RGB)
                img_pil = Image.fromarray(final_rgb)
                img_pil.save(save_path, quality=100)
                self.update_status(f"Файл сохранен: {os.path.basename(save_path)}", "green")
                messagebox.showinfo("Сохранено", f"Файл успешно сохранен в:\n{save_path}")
            except Exception as e:
                self.update_status("Ошибка сохранения!", "red")
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCalibratorApp(root)
    root.mainloop()