# Import the modules
import qrcode
import base64

# Open the image file and read the bytes
with open(r"C:\Users\multi\Desktop\All Folders\auto_test\0\9006407_2.png", "rb") as image_file:
    image_data = image_file.read()

# Encode the image data as a base64 string
encoded_string = base64.b64encode(image_data)

# Create a QR code object with the encoded string
qr = qrcode.QRCode()
qr.add_data(encoded_string)
qr.make()

# Save the QR code image
qr.make_image().save("qrcode.png")

