import qrcode

# Function to generate QR code for a given file path
def generate_qr_code(file_path, output_file="qr_code.png"):
    # Create a QR code object
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # Add the file path data to the QR code
    qr.add_data(file_path)
    qr.make(fit=True)

    # Create an image from the QR code
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Save the QR code image
    qr_img.save(output_file)

    print(f"QR code generated for file path: {file_path}")

# Example usage:
file_path = r"C:\Users\multi\Desktop\All Folders\auto_test\0\9006407_2.png"
generate_qr_code(file_path, "qrcode.png")
