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
  const [age, setAge] = useState(''); // Added state for age
  const [gender, setGender] = useState(''); // Added state for gender

  const handleLogin = () => {
    setShowLogin(false);
  };

  const [showModal, setShowModal] = useState(false);
  const [qrCode, setQrCode] = useState('');
  const [detectionResult, setDetectionResult] = useState('');
  const [normalResult, setNormalResult] = useState('');
  const [gradeResult, setGradeResult] = useState('');
  const [treatmentResult, setTreatmentResult] = useState('');
  const [imagePath, setImagePath] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [imageUploaded, setImageUploaded] = useState(false);

  const handleModalClose = () => {
    setShowModal(false);
  };

  const handleFileUpload = async () => {
    if (!fileToUpload || !age || !gender) return; // Check if age and gender are provided

    setDetectionResult('');
    setNormalResult('');
    setGradeResult('');
    setTreatmentResult('');
    setProcessing(true);

    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('age_category', age); // Add age to form data
    formData.append('gender', gender); // Add gender to form data

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

      let treatment = 'No treatment needed'; // Default treatment for normal knee
      if (analysisResult.normal_result !== 'Normal') {
        setTimeout(() => {
          treatment = `Treatment: ${analysisResult.treatments}`;
        }, 2800);
      }

      setTimeout(() => {
        setTreatmentResult(treatment);
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
              <label htmlFor="age" style={{ marginTop: '20px' }}>Age:</label>
              <input type="text" id="age" value={age} onChange={(e) => setAge(e.target.value)} style={{ marginBottom: '10px', padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }} />
              <label htmlFor="gender" style={{ marginBottom: '10px' }}>Gender:</label>
              <select id="gender" value={gender} onChange={(e) => setGender(e.target.value)} style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}>
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
              <button id="processButton" onClick={handleFileUpload} disabled={!imageUploaded} style={{ marginTop: '10px', padding: '10px 20px', backgroundColor: '#15d39a', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', transition: 'background-color 0.3s ease' }}>Process Image</button>
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
                  <div className="result-box">
                    <h3>Treatment</h3>
                    <p id="treatmentResult">{treatmentResult}</p>
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
