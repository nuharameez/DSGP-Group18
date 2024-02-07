import React, { useState } from 'react';
import './App.css'; // Make sure to adjust the path based on your project structure
import axios from 'axios';

const App = () => {
  const [qrCode, setQrCode] = useState('');
  const [kneeBone, setKneeBone] = useState('');
  const [normalOrNot, setNormalOrNot] = useState('');
  const [imagePath, setImagePath] = useState('');
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [notKneeBone, setNotKneeBone] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous results and messages
    setQrCode('');
    setKneeBone('');
    setNormalOrNot('');
    setImagePath('');
    setUploading(false);
    setAnalyzing(false);
    setNotKneeBone(false);

    setUploading(true);

    // Simulate a 1-second delay before setting the QR code
    setTimeout(() => {
      setQrCode(file.name);
    }, 1000);

    // Simulate a 1-second delay before showing the uploaded image
    setTimeout(() => {
      setImagePath('');
      setUploading(false);
      setImagePath(URL.createObjectURL(file));
    }, 2000);

    // Simulate a 1-second delay before analyzing the image
    setTimeout(() => {
      setAnalyzing(true);
    }, 3000);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate a 1-second delay before sending the request
      setTimeout(async () => {
        const response = await axios.post('http://localhost:5000/analyze', formData);
        const analysisResult = response.data;

        if (analysisResult.knee_bone_result === 'Not a Knee Bone') {
          setNotKneeBone(true);
        } else {
          setKneeBone(analysisResult.knee_bone_result);
          setNormalOrNot(analysisResult.normal_result);
        }
      }, 4000);
    } catch (error) {
      console.error('Error analyzing image:', error);
    } finally {
      // Simulate a 1-second delay before completing the analysis
      setTimeout(() => {
        setAnalyzing(false);
      }, 5000);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Knee Osteoarthritis Detection System</h1>
      <div className="content-container">
        <div className="input-container">
          <h2>Insert QR code</h2>
          <input
            type="file"
            id="qrCodeInput"
            accept=".png, .jpg, .jpeg"
            onChange={handleFileUpload}
          />
          {qrCode && <p className="qr-code">QR Code: {qrCode}</p>}
        </div>
        <div className="result-container">
          {uploading && <p className="uploading-message">Uploading image...</p>}
          {imagePath && (
            <div className="uploaded-image-container">
              <h3>Uploaded Image</h3>
              {notKneeBone && (
                <p className="not-knee-bone-message">Image uploaded is not a knee bone</p>
              )}
              <div className="uploaded-image">
                <img src={imagePath} alt="Uploaded" />
              </div>
            </div>
          )}
          {analyzing && <p className="analyzing-message">Analyzing image...</p>}
          {!analyzing && kneeBone && !notKneeBone && (
            <div className="result-box">
              <h2 className="result-title">Results</h2>
              <div className="result-item">
                <p>Knee Bone or Not:</p>
                <p className={kneeBone === 'Knee Bone Verified' ? 'knee-bone-verified' : 'not-knee-bone'}>
                  {kneeBone}
                </p>
              </div>
              <div className="result-item">
                <p>Normal or Not:</p>
                <p className={normalOrNot === 'Abnormal' ? 'abnormal-result' : 'normal-result'}>
                  {normalOrNot}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
