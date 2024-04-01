import React, { useState } from 'react';
import './App.css'; // Import the main CSS file
//import './background.css';
import axios from 'axios';
//import qrIcon from './qricon.png'; // Import the qricon.png file

const LoginPage = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');

  const handleLogin = (event) => {
    event.preventDefault();
    // Perform login logic
    onLogin();
  };

  const handleSignUp = (event) => {
    event.preventDefault();
    // Perform sign-up logic
    console.log('Signed up with:', { username, email, password });
  };

  const toggleSignUp = () => {
    setIsSignUp(!isSignUp);
    console.log('isSignUp:', !isSignUp);
  };

  return (
  <body-login>
    <div className="sign-in-form" id="signInForm">
      <h2>{isSignUp ? 'Sign Up' : 'Sign In'}</h2>
      <form onSubmit={isSignUp ? handleSignUp : handleLogin}>
        {isSignUp && (
          <input
            type="email"
            id="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        )}
        <input
          type="text"
          id="username"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          id="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">{isSignUp ? 'Confirm' : 'Sign In'}</button>
      </form>
      <p>{isSignUp ? 'Already have an account?' : "Don't have an account?"}</p>
      <button onClick={toggleSignUp}>{isSignUp ? 'Sign In' : 'Sign Up'}</button>
    </div>
  </body-login>
  );
};

const App = () => {
  const [showLogin, setShowLogin] = useState(true);

  const handleLogin = () => {
    setShowLogin(false);
  };

  const [showModal, setShowModal] = useState(false);
  const [qrCode, setQrCode] = useState('');
  const [detectionResult, setDetectionResult] = useState('');
  const [normalResult, setNormalResult] = useState('');
  const [gradeResult, setGradeResult] = useState('');
  const [imagePath, setImagePath] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);
  const [processing, setProcessing] = useState(false); // Added state for processing status
  const [imageUploaded, setImageUploaded] = useState(false); // Added state to track image upload

  const handleModalClose = () => {
    setShowModal(false);
  };

  const handleFileUpload = async () => {
    if (!fileToUpload) return;

    // Clear previous results and messages
    setDetectionResult('');
    setNormalResult('');
    setGradeResult('');
    setProcessing(true); // Set processing status to true

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData);
      const analysisResult = response.data;

      setTimeout(() => {
        setDetectionResult(`${analysisResult.knee_bone_result}`);
      }, 700);

      setTimeout(() => {
        setNormalResult(`Knee Bone is ${analysisResult.normal_result}`);
      }, 1400);

      setTimeout(() => {
        setGradeResult(`${analysisResult.severity}`);
      }, 2100);


      // Update imagePath with the base64 encoded image data if available
      if (analysisResult.image_base64) {
        setImagePath(`data:image/jpeg;base64,${analysisResult.image_base64}`);
      }
        setProcessing(false); // Set processing status to false after all results are set
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  const handleInputChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileToUpload(file);
      setImageUploaded(true); // Set imageUploaded to true when a file is selected
      setImagePath(URL.createObjectURL(file)); // Display the selected image immediately
    }
  };

  return (
    <>
      {showLogin ? (
        <LoginPage onLogin={handleLogin} />
      ) : (
        <div className="app-container">
          <nav className="navigation">
            <a href="#" className="logo">OsteoSense</a>
            <ul className="nav-links">
              <li><a href="#">Home</a></li>
              <li><a href="#">About</a></li>
              <li><a href="#">Contact</a></li>
            </ul>
          </nav>
          <div className="content-container">
            <div className="input-container">
              <input type="file" id="imageInput" accept="image/*" style={{ display: 'none' }} onChange={handleInputChange} />
              <div id="imageDisplay" onClick={() => document.getElementById('imageInput').click()}>
                {imageUploaded && <img src={imagePath} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '100%' }} />} {/* Render uploaded image if imagePath is set and imageUploaded is true */}
              </div>
              <button id="processButton" onClick={handleFileUpload} disabled={!imageUploaded}>Process Image</button> {/* Disable button if no image is uploaded */}
            </div>
            <div className="result-container" id="resultContainer">
              {processing && <p>Analyzing...</p>} {/* Display "Analyzing..." only when processing */}
              {!processing && ( // Display results only if not processing
                <>
                  <div className="result-box">
                    <h3>Detection</h3>
                    <p id="detectionResult">{detectionResult}</p>
                  </div>
                  <div className="result-box">
                    <h3>Normal or Not</h3>
                    <p id="normalResult">{normalResult}</p>
                  </div>
                  <div className="result-box">
                    <h3>Severity</h3>
                    <p id="gradeResult">{gradeResult}</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default App;
