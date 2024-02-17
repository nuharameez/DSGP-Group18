import React, { useState } from 'react';
import './App.css';
import UploadPage from './UploadPage'; // Import the UploadPage component
import axios from 'axios';

const App = () => {
  const [showModal, setShowModal] = useState(false);
  const [signedIn, setSignedIn] = useState(false); // Add state for tracking sign-in status
  const [qrCode, setQrCode] = useState('');
  const [kneeBone, setKneeBone] = useState('');
  const [normalOrNot, setNormalOrNot] = useState('');
  const [severity, setSeverity] = useState('');
  const [imagePath, setImagePath] = useState('');
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [notKneeBone, setNotKneeBone] = useState(false);

  const handleSignInClick = () => {
    setShowModal(true);
  };

  const handleSignInStartScanningClick = () => {
    setShowModal(true);
  };

  const handleModalClose = () => {
    setShowModal(false);
  };

  const handleSignInModalSubmit = () => {
    setSignedIn(true); // Update sign-in status to true
    setShowModal(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous results and messages
    setQrCode('');
    setKneeBone('');
    setNormalOrNot('');
    setSeverity('');
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
          if (analysisResult.normal_result === 'Abnormal') {
            setSeverity(analysisResult.severity); // Set severity if abnormal
          }
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
      <header className="header">
        <div className="nav">
          <div className="nav-left">
            <button className="button active">Home</button>
            <button className="button">Services</button>
            <button className="button">About</button>
          </div>
          <div className="nav-right">
            <button className="solid-button" onClick={handleSignInClick}>Sign In</button>
            <button className="outline-button">Sign Up</button>
          </div>
        </div>
      </header>
      <div className="content">
        {!signedIn && (
          <>
            <h1 className="main-heading">Knee Osteoarthritis Detection</h1>
            <p className="description">Welcome to our Knee Osteoarthritis Detection platform. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum nec mi bibendum, elementum magna ut, finibus dolor. Aenean eget arcu rhoncus, ultrices sapien nec, imperdiet urna. Donec malesuada ex quis elit convallis, at aliquam magna efficitur.</p>
          </>
        )}
        {signedIn && (
          <>
            <h1 className="main-heading">Knee Osteoarthritis Detection</h1>
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
                      {normalOrNot === 'Abnormal' && (
                        <div className="result-item">
                          <p>Severity:</p>
                          <p className="severity">{severity}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
      {showModal && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={handleModalClose}>&times;</span>
            <h2>Sign In</h2>
            <form onSubmit={handleSignInModalSubmit}>
              <div className="form-group">
                <label htmlFor="hospitalName">Hospital Name:</label>
                                <input type="text" id="hospitalName" name="hospitalName" />
              </div>
              <div className="form-group">
                <label htmlFor="hospitalCode">Hospital Code:</label>
                <input type="text" id="hospitalCode" name="hospitalCode" />
              </div>
              <button type="submit" className="solid-button">Sign In</button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;