#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from io import BytesIO
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="Number Prediction API", version="3.0")

# File paths
MODEL_PATH = "model.pth"
STATS_PATH = "model_stats.json"
CONFIG_PATH = "model_config.json"
MAPPINGS_PATH = "value_mappings.npy"

# Model Definition - MUST MATCH train.py
class NumberPredictor(nn.Module):
    def __init__(self, num_classes, num_positions, hidden_dim, num_heads, num_layers, dropout=0.2):
        super(NumberPredictor, self).__init__()
        
        self.num_classes = num_classes
        self.num_positions = num_positions
        
        # Embeddings
        self.value_embedding = nn.Embedding(num_classes, hidden_dim)
        self.position_embedding = nn.Embedding(100, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads - one per position
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_positions)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, num_positions)
        batch_size, seq_len, num_pos = x.shape
        
        # Embed each value
        x_flat = x.reshape(batch_size * seq_len * num_pos)
        embedded = self.value_embedding(x_flat)
        embedded = embedded.reshape(batch_size, seq_len, num_pos, -1)
        
        # Average across positions
        x = embedded.mean(dim=2)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0)
        x = x + pos_emb
        
        # Transformer
        x = self.transformer(x)
        
        # Use last timestep
        x = x[:, -1, :]
        
        # Predict each position
        outputs = [head(x) for head in self.output_heads]
        return torch.stack(outputs, dim=1)

# Load configuration and statistics
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    unique_values = np.load(MAPPINGS_PATH)
    value_to_idx = {int(val): idx for idx, val in enumerate(unique_values)}
    idx_to_value = {idx: int(val) for idx, val in enumerate(unique_values)}
    
    print("✓ Configuration loaded")
    print(f"  Sequence length: {config['sequence_length']}")
    print(f"  Input dimension: {stats['num_columns']}")
    print(f"  Number of classes: {stats['num_classes']}")
    
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}. Please run train.py first.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = NumberPredictor(
        num_classes=stats['num_classes'],
        num_positions=stats['num_columns'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please run train.py first.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

def analyze_data(dataset):
    """Analyze uploaded dataset"""
    diagnostics = {
        "total_rows": int(dataset.shape[0]),
        "total_columns": int(dataset.shape[1]),
        "value_range": f"[{int(dataset.min())}, {int(dataset.max())}]",
        "mean_value": float(dataset.mean()),
        "std_value": float(dataset.std()),
        "last_5_rows": dataset[-5:].tolist(),
        "unique_values_count": len(np.unique(dataset))
    }
    return diagnostics

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Number Sequence Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .model-info {
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .info-item {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .info-label { color: #666; font-size: 0.9em; }
        .info-value { color: #667eea; font-size: 1.3em; font-weight: bold; }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 968px) {
            .main-content { grid-template-columns: 1fr; }
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .card h2 { color: #667eea; margin-bottom: 20px; }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        
        .upload-area:hover { background: #e9ecef; }
        .upload-icon { font-size: 3em; margin-bottom: 15px; }
        input[type="file"] { display: none; }
        
        .file-info {
            margin-top: 15px;
            padding: 15px;
            background: #e7f3ff;
            border-radius: 8px;
            display: none;
        }
        
        .file-info.show { display: block; }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
            width: 100%;
            font-weight: bold;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .results { display: none; }
        .results.show { display: block; }
        
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }
        
        .prediction-values {
            font-size: 1.2em;
            font-weight: bold;
            line-height: 1.8;
        }
        
        .diagnostic-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .diagnostic-label { color: #666; font-size: 0.9em; }
        .diagnostic-value { color: #333; font-size: 1.1em; font-weight: 600; }
        
        .data-table {
            width: 100%;
            margin-top: 15px;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        .data-table th, .data-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }
        
        .data-table th { background: #667eea; color: white; }
        .data-table tr:nth-child(even) { background: #f8f9fa; }
        
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error, .success {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            display: none;
        }
        
        .error { background: #ff6b6b; color: white; }
        .success { background: #51cf66; color: white; }
        .error.show, .success.show { display: block; }
        
        .badge {
            display: inline-block;
            background: #51cf66;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 Number Sequence Prediction <span class="badge">v3.0</span></h1>
            <p>AI-powered pattern recognition for number sequences</p>
            
            <div class="model-info">
                <div class="info-item">
                    <div class="info-label">Sequence Length</div>
                    <div class="info-value">{{sequence_length}}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Input Columns</div>
                    <div class="info-value">{{num_columns}}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Value Range</div>
                    <div class="info-value">{{value_range}}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Model Accuracy</div>
                    <div class="info-value">{{accuracy}}%</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>📤 Upload Dataset</h2>
                
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <h3>Drop your Excel file here</h3>
                    <p>or click to browse</p>
                    <input type="file" id="fileInput" accept=".xlsx,.xls">
                </div>
                
                <div class="file-info" id="fileInfo">
                    <strong>Selected:</strong> <span id="fileName"></span>
                </div>
                
                <div class="error" id="errorMsg"></div>
                <div class="success" id="successMsg"></div>
                
                <button class="btn" id="predictBtn" disabled>Predict Next Row</button>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Making prediction...</p>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Results & Diagnostics</h2>
                
                <div class="results" id="results">
                    <div class="prediction-box">
                        <div style="margin-bottom: 10px;">Predicted Next Row:</div>
                        <div class="prediction-values" id="predictionValues"></div>
                    </div>
                    
                    <div>
                        <h3 style="color: #667eea; margin: 20px 0 15px;">Dataset Info</h3>
                        
                        <div class="diagnostic-item">
                            <div class="diagnostic-label">Total Rows</div>
                            <div class="diagnostic-value" id="totalRows">-</div>
                        </div>
                        
                        <div class="diagnostic-item">
                            <div class="diagnostic-label">Value Range</div>
                            <div class="diagnostic-value" id="dataRange">-</div>
                        </div>
                        
                        <div class="diagnostic-item">
                            <div class="diagnostic-label">Average Value</div>
                            <div class="diagnostic-value" id="avgValue">-</div>
                        </div>
                        
                        <h3 style="color: #667eea; margin: 20px 0 10px;">Last 5 Rows Used</h3>
                        <div id="lastRowsTable"></div>
                    </div>
                </div>
                
                <div style="text-align: center; color: #999; margin-top: 20px;" id="placeholder">
                    <p>Upload a file and click predict to see results</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        document.getElementById('uploadArea').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            if (!file) return;
            
            if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
                showError('Please upload an Excel file');
                return;
            }
            
            selectedFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileInfo').classList.add('show');
            document.getElementById('predictBtn').disabled = false;
            showSuccess('File loaded! Click Predict.');
            document.getElementById('results').classList.remove('show');
        }
        
        document.getElementById('predictBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('predictBtn').disabled = true;
            document.getElementById('errorMsg').classList.remove('show');
            document.getElementById('results').classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/predict-web/', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction failed');
                }
                
                displayResults(data);
                
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('predictBtn').disabled = false;
            }
        });
        
        function displayResults(data) {
            document.getElementById('predictionValues').textContent = data.prediction.join(', ');
            
            const diag = data.diagnostics;
            document.getElementById('totalRows').textContent = diag.total_rows;
            document.getElementById('dataRange').textContent = diag.value_range;
            document.getElementById('avgValue').textContent = diag.mean_value.toFixed(2);
            
            let tableHTML = '<table class="data-table"><thead><tr>';
            for (let i = 1; i <= diag.total_columns; i++) {
                tableHTML += `<th>C${i}</th>`;
            }
            tableHTML += '</tr></thead><tbody>';
            
            diag.last_5_rows.forEach(row => {
                tableHTML += '<tr>';
                row.forEach(val => {
                    tableHTML += `<td>${val}</td>`;
                });
                tableHTML += '</tr>';
            });
            
            tableHTML += '</tbody></table>';
            document.getElementById('lastRowsTable').innerHTML = tableHTML;
            
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('results').classList.add('show');
            showSuccess('Prediction completed!');
        }
        
        function showError(message) {
            document.getElementById('errorMsg').textContent = message;
            document.getElementById('errorMsg').classList.add('show');
            document.getElementById('successMsg').classList.remove('show');
        }
        
        function showSuccess(message) {
            document.getElementById('successMsg').textContent = message;
            document.getElementById('successMsg').classList.add('show');
            document.getElementById('errorMsg').classList.remove('show');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    html = HTML_TEMPLATE.replace('{{sequence_length}}', str(config['sequence_length']))
    html = html.replace('{{num_columns}}', str(stats['num_columns']))
    html = html.replace('{{value_range}}', f"[{stats['min']}, {stats['max']}]")
    accuracy = config.get('best_val_acc', 0)
    html = html.replace('{{accuracy}}', f"{accuracy:.1f}")
    return html

@app.post("/predict-web/")
async def predict_web(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Please upload an Excel file")
        
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents), header=None)
        dataset = df.values.astype(np.int64)
        
        if dataset.shape[1] != stats['num_columns']:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {stats['num_columns']} columns, got {dataset.shape[1]}"
            )
        
        sequence_length = config['sequence_length']
        if dataset.shape[0] < sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {sequence_length} rows, got {dataset.shape[0]}"
            )
        
        diagnostics = analyze_data(dataset)
        
        # Get last sequence and convert to indices
        last_sequence = dataset[-sequence_length:]
        last_sequence_indices = np.zeros_like(last_sequence, dtype=np.int64)
        for i in range(last_sequence.shape[0]):
            for j in range(last_sequence.shape[1]):
                last_sequence_indices[i, j] = value_to_idx[int(last_sequence[i, j])]
        
        # Predict
        sequence_tensor = torch.LongTensor(last_sequence_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(sequence_tensor)
            predictions_indices = torch.argmax(outputs, dim=2).cpu().numpy()[0]
        
        # Convert back to original values
        predictions = np.array([idx_to_value[int(idx)] for idx in predictions_indices])
        
        return {
            "success": True,
            "prediction": predictions.tolist(),
            "diagnostics": diagnostics,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "accuracy": config.get('best_val_acc', 0)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

