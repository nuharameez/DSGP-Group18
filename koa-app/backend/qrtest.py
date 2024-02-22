import cv2
import qrcode
import base64
from PIL import Image
from io import BytesIO

# Function to read and display the file path stored in a QR code
def read_and_display_qr_code(image_file):
    # Read the QR code image
    qr_image = cv2.imread(image_file)

    # Initialize the QRCode detector
    qr_detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    data, bbox, _ = qr_detector.detectAndDecode(qr_image)

    # If QR code detected, decode the file path
    if data:
        file_path = data.strip()  # Remove any leading or trailing whitespaces

        # Load and display the image corresponding to the file path
        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
                image = Image.open(BytesIO(image_data))
                image.show()
        except FileNotFoundError:
            print("File not found:", file_path)
    else:
        print("QR code not detected.")

# Example usage:
image_file = "site.png"
read_and_display_qr_code(image_file)