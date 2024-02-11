import qrcode

# Example data (URL)
data = "Bone_Xrays/Test/AP+Hand+Xray.png"

# Output file name
filename = "qrtrial.png"

# Generate QR code
img = qrcode.make(data)

# Save the image to a file
img.save(filename)
