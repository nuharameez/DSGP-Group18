import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const App = () => {
  const [showModal, setShowModal] = useState(false);
  const [qrCode, setQrCode] = useState('');
  const [detectionResult, setDetectionResult] = useState('');
  const [normalResult, setNormalResult] = useState('');
  const [gradeResult, setGradeResult] = useState('');
  const [allTreatments, setAllTreatments] = useState([]); // New state for all treatments
  const [displayedTreatments, setDisplayedTreatments] = useState([]); // New state for displayed treatments
  const [imagePath, setImagePath] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    if (allTreatments.length > 0) {
      // Initially display 3 random treatments
      const randomTreatments = getRandomTreatments(allTreatments, 3);
      setDisplayedTreatments(randomTreatments);
    }
  }, [allTreatments]);

  const getRandomTreatments = (treatments, num) => {
    const shuffled = treatments.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, num);
  };

  const handleModalClose = () => {
    setShowModal(false);
  };

  const handleFileUpload = async () => {
    if (!fileToUpload) return;

    setQrCode(fileToUpload.name);

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      setProcessing(true);

      const response = await axios.post('http://localhost:5000/analyze', formData);
      const analysisResult = response.data;

      setDetectionResult(`${analysisResult.knee_bone_result}`);
      setNormalResult(`Knee Bone is ${analysisResult.normal_result}`);
      setGradeResult(`${analysisResult.severity}`);

      if (analysisResult.knee_bone_result === 'Not a Knee Bone') {
        // If it's not a knee bone, clear treatments and return
        setAllTreatments([]);
        setDisplayedTreatments([]);
        setImagePath(URL.createObjectURL(fileToUpload));
        return;
      }

      setAllTreatments(analysisResult.treatments);
      setDisplayedTreatments(getRandomTreatments(analysisResult.treatments, 5));

      setImagePath(URL.createObjectURL(fileToUpload));
    } catch (error) {
      console.error('Error analyzing image:', error);
    } finally {
      setProcessing(false);
    }
  };

  const handleInputChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileToUpload(file);
    }
  };

  const handleRefresh = () => {
    // Show another set of 5 random treatments
    setDisplayedTreatments(getRandomTreatments(allTreatments, 3));
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
          <input type="file" id="imageInput" accept="image/*" style={{ display: 'none' }} onChange={handleInputChange} />
          <div id="imageDisplay" onClick={() => document.getElementById('imageInput').click()}></div>
          <button id="processButton" onClick={handleFileUpload}>Process Image</button>
        </div>
        <div className="result-container" id="resultContainer">
          {processing && <p>Analyzing...</p>}
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
          {/* Display 3 random treatments */}
          {displayedTreatments.length > 0 && (
            <div className="result-box">
              <h3>Treatment Recommendation</h3>
              <ul>
                {displayedTreatments.map((treatment, index) => (
                  <li key={index}>{treatment}</li>
                ))}
              </ul>
              <button onClick={handleRefresh}>Refresh</button> {/* Refresh button */}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
