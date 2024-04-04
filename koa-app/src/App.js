import React, { useState } from 'react';
import './App.css'; // Import the main CSS file
import axios from 'axios';

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
  const [fileToUpload, setFileToUpload] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [imageUploaded, setImageUploaded] = useState(false);
  const [imagePath, setImagePath] = useState('');
  const [detectionResult, setDetectionResult] = useState('');
  const [normalResult, setNormalResult] = useState('');
  const [gradeResult, setGradeResult] = useState('');
  const [treatmentResults, setTreatmentResults] = useState([]);

  const handleLogin = () => {
    setShowLogin(false);
  };

  const handleFileUpload = async () => {
    if (!fileToUpload) return;

    setDetectionResult('');
    setNormalResult('');
    setGradeResult('');
    setTreatmentResults([]);
    setProcessing(true);

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
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

      setTimeout(() => {
        setTreatmentResults(analysisResult.treatments);
      }, 2800);

      setTimeout(() => {
        setImagePath(`data:image/jpeg;base64,${analysisResult.image_base64}`);
        setProcessing(false);
      }, 3500);
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  const handleInputChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileToUpload(file);
      setImageUploaded(true);
      setImagePath(URL.createObjectURL(file));
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
                {imageUploaded && <img src={imagePath} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '100%' }} />}
              </div>
              <button id="processButton" onClick={handleFileUpload} disabled={!imageUploaded}>Process Image</button>
            </div>
            <div className="result-container" id="resultContainer">
              {processing && <p>Analyzing...</p>}
              {!processing && (
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
                  <div className="result-box" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    <h3>Treatment</h3>
                    {treatmentResults.length > 0 ? (
                      <ul>
                        {treatmentResults.map((treatment, index) => (
                          <li key={index}>{treatment}</li>
                        ))}
                      </ul>
                    ) : (
                      <p></p>
                    )}
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
