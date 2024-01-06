# Βιβλιοθήκες
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import customtkinter
import customtkinter as ctk
from PIL import Image, ImageTk
import pytesseract
import os
from updated_ocr import load_ocr
import numpy as np
import argparse
import imutils
import cv2
import time

ctk.set_appearance_mode("Dark")  # Επιλογή του χρώματος του παραθύρου
ctk.set_default_color_theme("blue")  # Επιλογή χρώματος των widgets


class Toplevel(customtkinter.CTk):
    '''Παράθυρο ειαγωγής ονόματος αρχείου'''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title('Εισαγωγή ονόματος αρχείου')
        self.geometry('300x100')

        self.labely = customtkinter.CTkLabel(self, text='Εισάγετε το όνομα του νέου αρχείου')
        self.labely.pack(pady=10)

        self.entry = customtkinter.CTkEntry(self, corner_radius=2, width=10)
        self.entry.pack(pady=5)

        self.okbtn = customtkinter.CTkButton(self, text='Oκ', command=self.on_ok)
        self.okbtn.pack(padx=4, side=ctk.RIGHT)

        self.dltbtn = customtkinter.CTkButton(self, text='Άκυρο', command=self.on_cancel)
        self.dltbtn.pack(padx=2, side=ctk.LEFT)

        self.file_name = None

    def on_ok(self):
        entered_name = self.entry.get().strip()
        if not entered_name or "/" in entered_name:
            messagebox.showerror("Error", "μη έγκυρο όνομα αρχείου")
        else:
            self.file_name = entered_name
            self.destroy()

    def on_cancel(self):
        self.destroy()


class Main_window(customtkinter.CTk):
    '''Κύριο παράθυρο/λειτουργία'''

    def __init__(self):  # Δημιουργία του πρώτου-κεντρικού παραθύρου και εισαγωγή των widgets
        super().__init__()
        self.title("Αναγνώριση Χειρόγραφων Χαρακτήρων")
        self.geometry("620x380")
        self.grid_columnconfigure((0, 1), weight=1)
        self.frame = customtkinter.CTkFrame(self, fg_color='transparent')
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.button = customtkinter.CTkButton(self, text="Οδηγίες", command=self.show_second_window)
        self.button.grid(row=10, column=0, padx=20, pady=20, sticky="ewe", columnspan=1)

        self.btnstart = customtkinter.CTkButton(self, text='Pytesseract', command=self.select_file)
        self.btnstart.grid(row=10, column=1, padx=20, pady=20, sticky="ewe", columnspan=1)

        self.newbtn = customtkinter.CTkButton(self, text="Tensorflow (Βήμα 1)", command=self.add_image)
        self.newbtn.grid(row=20, column=0, padx=20, pady=20, sticky="ewe", columnspan=1)

        self.process = customtkinter.CTkButton(self, text="Tensorflow (Βήμα 2)", command=self.start_ocr)
        self.process.grid(row=20, column=1, padx=20, pady=20, sticky="ewe", columnspan=1)

    def add_image(self):  # Συνάρτηση για την επιλογή αρχείου πατώντας το κουμπί tensorflow(βήμα 1)

        self.import_data = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    def start_ocr(self):  # OCR με την tensorflow
        answer = []
        preds, boxes, image = load_ocr(self.import_data)
        # Ορισμός της λίστας με τα label names
        labelNames = "0123456789"
        labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        labelNames = [l for l in labelNames]

        # loop over the predictions and bounding box locations together
        for (pred, (x, y, w, h)) in zip(preds, boxes):
            # Βρίσκει το index της label με την μεγαλύτερη πιθανότητα και μετά εξάγει την πιθανότητα και αυτή τη label
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]

            # Σχεδιάζει την πρόβλεψη της εικόνας
            print("[INFO] {} - {:.2f}%".format(label, prob * 100))
            answer.append(label)  # Προσθήκη της label στην λίστα answer
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),
                          2)  # Σχεδίαση πλαισίου γύρω από την ανιχνευόμενη εικόνα με την OpenCV
            cv2.putText(image, label, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Δείξε την εικόνα
            cv2.imshow("Image", image)
            cv2.waitKey(1000)
        result = ''.join(answer)
        file_name = simpledialog.askstring("", "Εισάγετε το όνομα του νέου αρχείου:")
        self.create_text_file(file_name, result)  # κλήση της συνάρτησης create_file_text

    def show_second_window(self):  # Συνάρτηση για την δημιουργία παραθύρου οδηγιών χρήσης της εφαρμογής
        self.second_frame = customtkinter.CTkFrame(self, width=400,
                                                   height=400)  # Δημιουργία frame στο κεντρικό παράθυρο
        self.second_frame.grid(row=0, column=0, padx=30, pady=10, sticky="nsew")

        instructions_label = customtkinter.CTkTextbox(self.second_frame)  # Εισαγωγή κειμένου μέσα στο frame
        instructions_label.pack(pady=10)
        instructions_label.insert("0.0", "Πώς να χρησιμοποιήσετε την εφαρμογή: \n\n"
                                          "Η εφαρμογή σας δίνει δύο \nτρόπους που μπορείτε να \nπραγματοποιήσετε OCR.\n\nΠατώντας το κουμπί \n'Pytesseract' επιλέγετε το \nεπιθυμητό αρχείο εικόνας \nκαι εκτελείται η αναγνώριση των χαρακτήρων με την \nPytesseract.\n\nΓια να πραγματοποιήσετε \nOCR με την Tensorflow \nπατήστε αρχικά του κουμπί \n'Tensorflow (Βήμα 1)' για να \nεπιλέξετε αρχείο εικόνας. \nΈπειτα πατήστε 'Tensorflow  (Βήμα 2)' για να πραγματοποι- ήσετε OCR με την \nTensorflow.\n\n Ευχαριστούμε! ")

    def select_file(self):  # Συνάρτηση για την επιλογή αρχείου εικόνας από τον χρήστη
        global_file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg")])  # Ο χρήστης πρέπει να επιλέξει αρχεία εικόνας αυτής της μορφής
        if global_file_path:
            file_name = simpledialog.askstring("",
                                               "Εισάγετε το όνομα του νέου αρχείου:")  # Εμφάνιση του παραθύρου για εισαγωγή ονόματος αρχείου
            if file_name is not None:
                if not file_name or "/" in file_name or "?" in file_name or "<" in file_name or ":" in file_name or ">" in file_name or "|" in file_name or "*" in file_name:  # Έλεγχος χρήσης αποδεκτού ονόματος αρχείου
                    messagebox.showerror("Κάτι πήγε λάθος",
                                         "Εισάγετε έγκυρο όνομα αρχείου.")  # Μήνυμα λάθους σε περίπτωση εισαγωγής μη έγκυρης μορφής ονόματος αρχείου.
                else:
                    ocr_result = self.ocr_image(
                        global_file_path)  # Kλήση της συνάρτησης ocr_image για την αναγνώριση του κειμένου μέσω της pytesseract
                    self.create_text_file(file_name, ocr_result)  # Κλήση της συνάρτησης create_text_file
            else:
                messagebox.showerror("Κάτι πήγε λάθος",
                                     "Εισάγετε έγκυρο όνομα αρχείου.")  # Μήνυμα λάθους στην περίπτωση που ο χρήστης δεν εισάγει όνομα αρχείου(δηλαδή πατήσει απλώς enter)
        else:
            messagebox.showerror("Κάτι πήγε λάθος",
                                 "Δεν διαλέξατε αρχείο εικόνας.")  # Μήνυμα λάθους στην περίπτωση που ο χρήστης δεν επιλέξει αρχείο εικόνας για επεξεργασία.

    def ocr_image(self, file_path):  # Συνάρτηση για την εκτέλεση ocr
        try:
            image = Image.open(file_path)  # χρήση της βιβλιοθήκης
            ocr_result = pytesseract.image_to_string(image)  # ocr με pytesseract
            messagebox.showinfo("Αποτέλεσμα του OCR:", f"Το αποτέλεσμα του OCR είναι:\n{ocr_result}")
            return ocr_result
        except Exception as e:  # Μήνυμα λάθους
            messagebox.showerror("Κάτι πήγε λάθος", f"Σφάλμα κατά την εκτέλεση OCR: {str(e)}")

    @staticmethod
    def create_text_file(file_name, content):  # Συνάρτηση για την δημιουργία του αρχείου με το αποτέλεσμα του ocr
        try:
            with open(file_name + ".txt", "w", encoding="utf-8") as file:  # Δημιουργία του αρχείου txt
                file.write(content)
            messagebox.showinfo(":)",
                                f"Το αρχείο '{file_name}.txt' δημιουργήθηκε με επιτυχία. Είναι αποθηκευμένο στο: {os.getcwd()}")  # Ενημέρωση του χρήστη για την επιτυχή δημιουργία του αρχείου την θέση αποθήκευσής του.
        except Exception as e:
            messagebox.showerror("Κάτι πήγε λάθος", f"Σφάλμα κατα την δημιουργία του αρχείου: {str(e)}")


app = Main_window()
app.mainloop()
