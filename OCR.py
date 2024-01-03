from tkinter import *
from tkinter import filedialog, messagebox,simpledialog
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

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green") #Επιλογή χρώματος των widgets

class Toplevel(customtkinter.CTk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title ('Εισαγωγή ονόματος αρχείου')
        self.geometry('300x100')

        self.labely=customtkinter.CTkLabel(self,text='Εισάγετε το όνομα του νέου αρχείου')
        self.labely.pack(pady=10)

        self.entry=customtkinter.CTkEntry(self,corner_radius=2,width=10)
        self.entry.pack(pady=5)

        self.okbtn= customtkinter.CTkButton(self,text='Oκ',command=self.on_ok)
        self.okbtn.pack(padx=4,side=ctk.RIGHT)

        self.dltbtn= customtkinter.CTkButton(self,text='Άκυρο',command=self.on_cancel)
        self.dltbtn.pack(padx=2,side=ctk.LEFT)

        self.file_name=None

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
    '''Κύριο παράθυρο'''
    def __init__(self):
        super().__init__()
        self.title("Αναγνώριση Χειρόγραφων Χαρακτήρων")
        self.geometry("520x380")
        self.grid_columnconfigure((0, 1), weight=1)
        self.frame = customtkinter.CTkFrame(self, fg_color='transparent')
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.button = customtkinter.CTkButton(self, text="Οδηγίες", command=self.show_second_window)
        self.button.grid(row=10, column=0, padx=20, pady=20, sticky="ew", columnspan=1)
        self.btnstart = customtkinter.CTkButton(self, text='Έναρξη', command=self.select_file)
        self.btnstart.grid(row=10, column=1, padx=20, pady=20, sticky="ewe", columnspan=1)
        self.newbtn=customtkinter.CTkButton(self, text="select image", command=self.add_image)
        self.newbtn.grid(row=20, column=0, padx=20, pady=20, sticky="ewe", columnspan=1)
        self.process=customtkinter.CTkButton(self, text="process image", command=self.start_ocr)
        self.process.grid(row=20, column=1, padx=20, pady=20, sticky="ewe", columnspan=1)
        
    def add_image(self):
        
         self.import_data = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        

        
        
    
        
    def start_ocr(self):
        answer = []
        preds, boxes, image = load_ocr(self.import_data)
        # define the list of label names
        labelNames = "0123456789"
        labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        labelNames = [l for l in labelNames]

        # loop over the predictions and bounding box locations together
        for (pred, (x, y, w, h)) in zip(preds, boxes):
            # find the index of the label with the largest corresponding
            # probability, then extract the probability and label
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]
        
            # draw the prediction on the image
            print("[INFO] {} - {:.2f}%".format(label, prob * 100))
            answer.append(label)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
            # show the image
            cv2.imshow("Image", image)
            cv2.waitKey(1000)
        result = ''.join(answer)
        file_name = simpledialog.askstring("", "Εισάγετε το όνομα του νέου αρχείου:")
        self.create_text_file(file_name, result)
        '''if file_name is not None:
            if not file_name or "/" in file_name or "?" in file_name or "<" in file_name or ":" in file_name or ">" in file_name or "|" in file_name or "*" in file_name:
                 messagebox.showerror("Κάτι πήγε λάθος", "Εισάγετε έγκυρο όνομα αρχείου.")
            else:
                ocr_result = self.ocr_image(global_file_path)
                self.create_text_file(file_name, ocr_result)
        else:
            messagebox.showerror("Κάτι πήγε λάθος", "Εισάγετε έγκυρο όνομα αρχείου.")'''
            
        
        
        
    
    def show_second_window(self):
        self.second_frame = customtkinter.CTkFrame(self, width=400, height=400)
        self.second_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        instructions_label = customtkinter.CTkTextbox(self.second_frame)
        instructions_label.pack(pady=10)
        instructions_label.insert("0.0", "Πώς να χρησιμοποιήσετε την  εφαρμογή: \n\n"
                                          "Πατώντας το κουμπί 'Έναρξη'   μπορείτε να επιλέξετε το αρ- χείο εικόνας στο οποίο θέλετε να πραγματοποιήσετε OCR.")

    def select_file(self):
        global_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if global_file_path:
            file_name = simpledialog.askstring("", "Εισάγετε το όνομα του νέου αρχείου:")
            if file_name is not None:
                if not file_name or "/" in file_name or "?" in file_name or "<" in file_name or ":" in file_name or ">" in file_name or "|" in file_name or "*" in file_name:
                    messagebox.showerror("Κάτι πήγε λάθος", "Εισάγετε έγκυρο όνομα αρχείου.")
                else:
                    ocr_result = self.ocr_image(global_file_path)
                    self.create_text_file(file_name, ocr_result)
            else:
                messagebox.showerror("Κάτι πήγε λάθος", "Εισάγετε έγκυρο όνομα αρχείου.")
        else:
            messagebox.showerror("Κάτι πήγε λάθος", "Δεν διαλέξατε αρχείο εικόνας.")


    def ocr_image(self, file_path):
        try:
            image = Image.open(file_path)
            ocr_result = pytesseract.image_to_string(image)
            messagebox.showinfo("Αποτέλεσμα του OCR:", f"Το αποτέλεσμα του OCR είναι:\n{ocr_result}")
            return ocr_result
        except Exception as e:
            messagebox.showerror("Κάτι πήγε λάθος", f"Σφάλμα κατά την εκτέλεση OCR: {str(e)}")
    
    @staticmethod
    def create_text_file(file_name, content):
        try:
            with open(file_name + ".txt", "w", encoding="utf-8") as file:
                file.write(content)
            messagebox.showinfo("Γιουπι!", f"Το αρχείο '{file_name}.txt' δημιουργήθηκε με επιτυχία. Είναι αποθηκευμένο στο: {os.getcwd()}")
        except Exception as e:
            messagebox.showerror("Κάτι πήγε λάθος", f"Σφάλμα κατα την δημιουργία του αρχείου: {str(e)}")
            



app = Main_window()
app.mainloop()
