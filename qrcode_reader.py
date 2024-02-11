"""""

import qrcode

qr = qrcode.QRCode(version=1, box_size=10, border=5)

#data = input("Enter the url you want as a QR code: ")
data = "Bone_Xrays/Test/a4888249bacb0ebacd0fa7be38fd38_big_gallery.jpg"
qr.add_data(data)

qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")

img.save("qrcode.png")

import base64
import qrcode
from qrcode.constants import ERROR_CORRECT_L


with open(r"C:chanu\DSGP-Group18\Bone_Xrays\Test\AP+Hand+Xray.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

# Determine the appropriate version based on the data length
data_length = len(encoded_string)
version = 1  # Default version

# Iterate through versions from 1 to 40 to find the smallest version that can accommodate the data
for i in range(1, 41):
    if data_length < qrcode.util.MODE_INDICATOR[i][3]:
        version = i
        break

qr = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L)
#qr = qrcode.QRCode(version={"correct version"}, error_correction=qrcode.constants.{"correct error correction type"})
qr.add_data(encoded_string)
qr.make()
# Create an image object
img = qr.make_image()

# Save the QR code image
img.save("qrcode2.png")



import segno

#qrcode = segno.make_qr("Hello, World")
#qrcode.save("basic_qrcode.png")

import base64
import qrcode

# Open your image file
with open("Bone_Xrays/Test/a4888249bacb0ebacd0fa7be38fd38_big_gallery.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

# Create a QR code
qr = qrcode.QRCode()
qr.add_data(encoded_string)
# Automatically determine the best version while ensuring it doesn't exceed a certain maximum version
max_version = 40  # Set the maximum version
while True:
    try:
        qr.make()
        break
    except ValueError:
        if qr.version < max_version:
            qr.version += 1
        else:
            raise

# Save the QR code or use it as needed
qr.make_image().save("image_qrcode.png")

import qrcode
import base64

# Function to encode an image file to base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Path to the image you want to embed in the QR code
image_path = "Bone_Xrays/Test/AP+Hand+Xray.png"

# Convert the image to base64 string
image_base64 = image_to_base64(image_path)

# Combine the image base64 string with additional data (e.g., URL)
data = "Your additional data here: " + image_base64

# Create the QR code instance
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)

# Add the combined data to the QR code
qr.add_data(data)

# Determine the appropriate version based on the data length
qr.make(fit=True)
version = qr.version

# Recreate the QR code instance with the determined version
qr = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L)
qr.add_data(data)

# Make the QR code
qr.make()

# Create an image object
img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code image
img.save("qrcode_with_image.png")

"""""
import cv2
import webbrowser

def read_qr_code(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Initialize the QR code detector
    qr_code_detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    retval, decoded_info, points, straight_qrcode = qr_code_detector.detectAndDecodeMulti(img)

    if retval:
        # Extract the link
        link = decoded_info[0]
        print(f"QR Code link: {link}")
        return link
    else:
        print("No QR code found in the image.")
        return None

# Provide the path to the QR code image
qr_image_path = "QR/WhatsApp Image 2024-02-08 at 22.21.32_85f3cb70.jpg"
extracted_link = read_qr_code(qr_image_path)

# Now you can open the extracted link (e.g., using a web browser)
if extracted_link:
    print(f"Opening link: {extracted_link}")
    webbrowser.open(extracted_link)
    # Add your code here to open the link programmatically
else:
    print("QR code extraction failed.")
