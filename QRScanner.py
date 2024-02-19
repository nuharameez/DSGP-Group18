import cv2

def read_qr_code(image_path):
    image = cv2.imread(image_path)
    detector = cv2.QRCodeDetector()
    data, vertices_array, _ = detector.detectAndDecode(image)

    if vertices_array is not None:
        print("QR Code detected!")
        return data
    else:
        print("No QR Code detected.")

def display_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("Image from QR code", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = r"C:\Users\MSI\Downloads\IIT STUFF\CM 2603 DS\QRCodes\output_qr_code.png"  # Change this to the QR code image file path
image_data = read_qr_code(image_path)
if image_data:
    print("Image path extracted from QR code:", image_data)
    display_image(image_data)
else:
    print("Unable to extract image path from QR code.")
