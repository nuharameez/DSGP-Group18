import pyqrcode

def generate_qr_code(image_path, output_path):
    qr = pyqrcode.create(image_path)
    qr.png(output_path, scale=8)


image_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\CW DATASETS\auto_test\2\9082631_2.png"  # Change this to your image file path
output_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\QRCodes\output_qr_code.png"  # Output QR code file path

generate_qr_code(image_path, output_path)
print("QR code generated successfully!")
