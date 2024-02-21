import cv2
import requests
from PIL import Image
from io import BytesIO

# Load the QR code image
qr_code_image = cv2.imread(r'C:\Users\chanu\DSGP-Group18\QR\qrcode_95674291_172e96abc541b82a063d2605c185e7b8.png')

# Convert the image to grayscale
gray = cv2.cvtColor(qr_code_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to binarize
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Initialize the QR code detector
detector = cv2.QRCodeDetector()

# Detect QR code in the image
data, vertices_array, binary_qrcode = detector.detectAndDecode(qr_code_image)

if data:
    print("QR Code detected! Data:", data)

    # Download the image from the URL
    response = requests.get(data)

    if response.status_code == 200:
        # Open the downloaded image
        image = Image.open(BytesIO(response.content))

        # Save the image to local device
        image.save('downloaded_image.jpg')
        print("Image downloaded successfully!")
    else:
        print("Failed to download image. Status code:", response.status_code)
else:
    print("No QR Code detected in the image.")
