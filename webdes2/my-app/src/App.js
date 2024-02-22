// App.js
import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

const App = () => {
  const [showModal, setShowModal] = useState(false);
  const [qrCode, setQrCode] = useState('');
  const [detectionResult, setDetectionResult] = useState('');
  const [normalResult, setNormalResult] = useState('');
  const [gradeResult, setGradeResult] = useState('');
  const [imagePath, setImagePath] = useState('');

  const handleModalClose = () => {
    setShowModal(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous results and messages
    setQrCode('');
    setDetectionResult('');
    setNormalResult('');
    setGradeResult('');
    setImagePath('');

    setQrCode(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData);
      const analysisResult = response.data;

      setDetectionResult(`${analysisResult.knee_bone_result}`);
      setNormalResult(`Knee Bone is ${analysisResult.normal_result}`);
      setGradeResult(`${analysisResult.severity}`);
      setImagePath(URL.createObjectURL(file));
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  return (
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
          <label htmlFor="imageInput" className="custom-icon" id="customIcon">
            <img src="qricon.png" alt="Upload Image" />
          </label>
          <input type="file" id="imageInput" accept="image/*" onChange={handleFileUpload} />
          <div id="imageDisplay" onClick={() => document.getElementById('imageInput').click()}></div>
          <button id="processButton">Process Image</button>
        </div>
        <div className="result-container" id="resultContainer">
          <div className="result-box">
            <h3>Detection</h3>
            <p id="detectionResult">{detectionResult}</p>
          </div>
          <div className="result-box">
            <h3>Normal or Not</h3>
            <p id="normalResult">{normalResult}</p>
          </div>
          <div className="result-box">
            <h3>Grade</h3>
            <p id="gradeResult">{gradeResult}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
