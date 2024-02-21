/*
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data.result);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <div className="App">
      <h1>Knee Bone Identifier</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handlePredict}>Predict</button>
      {result && <p>{result}</p>}
    </div>
  );
}
export default App;




import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');
  const [qrCodeLink, setQrCodeLink] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data.result);
      setQrCodeLink(response.data.QR_code_link); // Set the QR code link
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <div className="App">
      <h1>Knee Bone Identifier</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handlePredict}>Predict</button>
      {result && <p>{result}</p>}
      {qrCodeLink && (
        <div>
          <p>QR Code Link: {qrCodeLink}</p>
          <a href={qrCodeLink} target="_blank" rel="noopener noreferrer">Visit QR Code Link</a>
        </div>
      )}
    </div>
  );
}
export default App;
*/

import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedQRCode, setSelectedQRCode] = useState(null);
  const [qrCodeLink, setQrCodeLink] = useState('');
  const [selectedXRayImage, setSelectedXRayImage] = useState(null);
  const [result, setResult] = useState('');

  const handleQRCodeChange = (event) => {
    setSelectedQRCode(event.target.files[0]);
  };

  const handleXRayImageChange = (event) => {
    setSelectedXRayImage(event.target.files[0]);
  };

  const decodeQRCode = async () => {
    const formData = new FormData();
    formData.append('image', selectedQRCode);

    try {
      const response = await axios.post('http://localhost:5000/decode_qr_code', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setQrCodeLink(response.data.QR_code_link);
    } catch (error) {
      console.error('Error decoding QR code:', error);
    }
  };

  const predict = async () => {
    // Implement the prediction logic here
    const formData = new FormData();
    formData.append('image', selectedXRayImage);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data.result);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <div className="App">
      <h1>Knee Bone Identifier</h1>
      <h2>Upload QR Code</h2>
      <input type="file" onChange={handleQRCodeChange} />
      <button onClick={decodeQRCode}>Decode QR Code</button>
      {qrCodeLink && (
        <div>
          <p>QR Code Link: <a href={qrCodeLink} target="_blank" rel="noopener noreferrer">{qrCodeLink}</a></p>
        </div>
      )}
      <h2>Upload X-Ray Image</h2>
      <input type="file" onChange={handleXRayImageChange} />
      <button onClick={predict}>Predict</button>
      {result && <p>{result}</p>}
    </div>
  );
}
export default App;
