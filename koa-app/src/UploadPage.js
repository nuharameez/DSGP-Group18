import React, { useRef } from 'react';

const UploadPage = () => {
  const fileInputRef = useRef(null);

  const handleFileUpload = () => {
    fileInputRef.current.click();
  };

  const handleFileSelected = (event) => {
    const file = event.target.files[0];
    // Handle the selected file
    console.log('File selected:', file);
  };

  return (
    <div className="upload-page">
      <h2>Upload Image</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileSelected}
        ref={fileInputRef}
        style={{ display: 'none' }}
      />
      <button className="solid-button" onClick={handleFileUpload}>Upload Image</button>
    </div>
  );
};

export default UploadPage;
