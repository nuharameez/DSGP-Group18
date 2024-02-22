import React, { useState } from 'react';
//import './UploadQRCodePage.css'; // You can style this page separately if needed

const UploadQRCodePage = ({ handleQRCodeUpload }) => {
  const [qrCodeFile, setQRCodeFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type if needed

    setQRCodeFile(file);
  };

  const handleUpload = () => {
    if (qrCodeFile) {
      handleQRCodeUpload(qrCodeFile);
    }
  };

  return (
    <div className="upload-qr-code-container">
      <h2>Upload QR Code</h2>
      <input
        type="file"
        accept=".png, .jpg, .jpeg"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default UploadQRCodePage;
