// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

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
        setResult(response.data.result);
      })
      .catch(error => {
        console.error('Error analyzing image:', error);
      });
  };

  return (
    <div>
      <h1>Image Analyzer</h1>
      <input type="file" onChange={handleFileChange} />
      {result && <p>Result: {result}</p>}
    </div>
  );
}

export default App;
