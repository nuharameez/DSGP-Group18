import React, { useState } from 'react';
import './App.css'; // Make sure to adjust the path based on your project structure
import axios from 'axios';

const App = () => {
  const [qrCode, setQrCode] = useState('');
  const [kneeBone, setKneeBone] = useState('');
  const [diseaseKnee, setDiseaseKnee] = useState('');
  const [klGrading, setKlGrading] = useState('');
  const [treatmentRecommendation, setTreatmentRecommendation] = useState('');

  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];

    // Assuming the QR code is read from the file, you might need to implement a QR code reader
    // For simplicity, just setting the filename here
    setQrCode(file.name);

    // If a file is selected, directly analyze it
    if (file) {
      setSelectedFile(file);
      analyzeImage(file);
    }
  };

  const analyzeImage = (file) => {
    const formData = new FormData();
    formData.append('file', file);

    axios.post('http://localhost:5000/analyze', formData)
      .then(response => {
        const analysisResult = response.data.result;

        // Update state based on the analysis result
        setDiseaseKnee(analysisResult);

        // Optionally, you can set other state variables based on the result
        // For example, setKlGrading, setTreatmentRecommendation, etc.

        setResult(analysisResult);
      })
      .catch(error => {
        console.error('Error analyzing image:', error);
      });
  };

  return (
    <div className="app-container">
      <h1>Knee Osteoarthritis Detection System<br /><br /></h1>
      <div className="content-container">
        <div className="left-container">
          <h2>Insert QR code</h2>
          <input
            type="file"
            id="qrCodeInput"
            accept=".png, .jpg, .jpeg"
            onChange={handleFileUpload}
          />
          {qrCode && <p>QR Code: {qrCode}</p>}
        </div>
        <div className="right-container">
          <h2>Results</h2>
          <div className="result-item">
            <div className={`result-box ${kneeBone === 'Normal' ? 'normal-result' : 'abnormal-result'}`}>
              <p>Knee Bone or Not:</p>
              <p>{kneeBone}</p>
            </div>
          </div>
          <div className="result-item">
            <div className={`result-box ${diseaseKnee === 'Normal' ? 'normal-result' : 'abnormal-result'}`}>
              <p>Disease Knee or Not:</p>
              <p>{diseaseKnee}</p>
            </div>
          </div>
          <div className="result-item">
            <div className={`result-box ${klGrading === 'Normal' ? 'normal-result' : 'abnormal-result'}`}>
              <p>KL Grading:</p>
              <p>{klGrading}</p>
            </div>
          </div>
          <div className="result-item">
            <div className={`result-box ${treatmentRecommendation === 'Normal' ? 'normal-result' : 'abnormal-result'}`}>
              <p>Treatment Recommendation:</p>
              <p>{treatmentRecommendation}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
