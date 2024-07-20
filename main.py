import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner")
        
        self.cap = cv2.VideoCapture(0)
        
        self.frame_label = Label(root)
        self.frame_label.pack(pady=10)
        
        self.scan_button = Button(root, text="Scan Document", command=self.scan_document)
        self.scan_button.pack(pady=10)
        
        self.save_button = Button(root, text="Save as PDF", command=self.save_as_pdf, state=DISABLED)
        self.save_button.pack(pady=10)
        
        self.scanned_image = None

    def scan_document(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from the webcam.")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                document_contour = approx
                cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 2)

                pts = document_contour.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")

                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                (tl, tr, br, bl) = rect

                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

                self.scanned_image = warped
                self.display_image(warped)
                self.save_button.config(state=NORMAL)
        
    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.frame_label.config(image=image)
        self.frame_label.image = image
        
    def save_as_pdf(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if filename:
            pdf_filename = filename
            c = canvas.Canvas(pdf_filename, pagesize=letter)
            c.drawImage(self.scanned_image, 0, 0, letter[0], letter[1])
            c.save()
            messagebox.showinfo("Success", f"Saved {pdf_filename}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = Tk()
    app = DocumentScannerApp(root)
    app.run()