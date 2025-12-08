# api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import tempfile
import os
from main import extract_dental_quote
from typing import Dict, Any

app = FastAPI(
    title="Dental Quote Extractor API",
    description="API pour extraire les informations des devis dentaires PDF",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML intégré directement
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🦷 Dental Quote Extractor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .content {
            padding: 40px;
        }
        .upload-section {
            border: 3px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-section.dragover {
            border-color: #2ecc71;
            background: #f8fffe;
        }
        .upload-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            margin: 10px;
        }
        .upload-btn:hover { background: #2980b9; }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .info-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .treatments-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .treatments-table th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .treatments-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .treatments-table tr:hover {
            background: #f5f5f5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Extracteur de Devis Dentaires</h1>
            <p>Extrayez automatiquement les informations des devis dentaires PDF</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h3>📤 Télécharger un devis PDF</h3>
                <p style="margin-bottom: 20px; color: #7f8c8d;">
                    Glissez-déposez ou cliquez pour sélectionner un fichier PDF
                </p>
                <input type="file" id="fileInput" accept=".pdf" style="display: none;">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Choisir un fichier PDF
                </button>
                <div id="fileInfo" style="margin-top: 20px;"></div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Extraction des données en cours...</p>
            </div>
            
            <div id="errorMessage" style="display: none; background: #e74c3c; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;"></div>
            <div id="successMessage" style="display: none; background: #2ecc71; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;"></div>
            
            <div class="results" id="results"></div>
        </div>
    </div>
    
    <script>
        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        
        // Drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadSection.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadSection.addEventListener(eventName, () => {
                uploadSection.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadSection.addEventListener(eventName, () => {
                uploadSection.classList.remove('dragover');
            }, false);
        });
        
        uploadSection.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', handleFileSelect, false);
        
        function handleDrop(e) {
            const files = e.dataTransfer.files;
            handleFiles(files);
        }
        
        function handleFileSelect(e) {
            handleFiles(e.target.files);
        }
        
        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type === 'application/pdf') {
                    document.getElementById('fileInfo').innerHTML = 
                        '<strong>Fichier:</strong> ' + file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)';
                    uploadFile(file);
                } else {
                    showError('Veuillez sélectionner un fichier PDF valide.');
                }
            }
        }
        
        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            showLoading();
            hideError();
            hideSuccess();
            hideResults();
            
            try {
                const response = await fetch('/extract-quote', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Erreur lors de l\'extraction');
                }
                
                const data = await response.json();
                showSuccess('✅ Extraction réussie !');
                displayResults(data);
                
            } catch (error) {
                showError('❌ Erreur: ' + error.message);
            } finally {
                hideLoading();
            }
        }
        
        function displayResults(data) {
            let html = '';
            
            // Dentiste
            if (data.dentiste) {
                html += '<div class="section"><h4>👨‍⚕️ Dentiste</h4><div class="info-grid">';
                html += createInfoItem('Nom', data.dentiste.nom_praticien);
                html += createInfoItem('RPPS', data.dentiste.rpps);
                html += createInfoItem('Adresse', data.dentiste.adresse);
                html += '</div></div>';
            }
            
            // Patient
            if (data.patient) {
                html += '<div class="section"><h4>👤 Patient</h4><div class="info-grid">';
                html += createInfoItem('Nom', data.patient.nom);
                html += createInfoItem('Date naissance', data.patient.date_naissance);
                html += createInfoItem('N° Sécurité Sociale', data.patient.numero_securite_sociale);
                html += '</div></div>';
            }
            
            // Traitements
            if (data.traitements && data.traitements.length > 0) {
                html += '<div class="section"><h4>🦷 Traitements</h4><table class="treatments-table">';
                html += '<tr><th>Code</th><th>Dent</th><th>Description</th><th>Honoraires</th></tr>';
                data.traitements.forEach(t => {
                    html += `<tr>
                        <td>${t.code_acte || '-'}</td>
                        <td>${t.dent || '-'}</td>
                        <td>${t.description || '-'}</td>
                        <td>${formatCurrency(t.honoraires)}</td>
                    </tr>`;
                });
                html += '</table></div>';
            }
            
            // Résumé financier
            if (data.financier) {
                html += '<div class="section"><h4>💰 Résumé Financier</h4><div class="info-grid">';
                html += createInfoItem('Total Honoraires', formatCurrency(data.financier.honoraires_total));
                html += createInfoItem('Reste à Charge', formatCurrency(data.financier.reste_a_charge));
                html += '</div></div>';
            }
            
            results.innerHTML = html;
            showResults();
        }
        
        function createInfoItem(label, value) {
            return value ? `<div class="info-item"><strong>${label}:</strong><br>${value}</div>` : '';
        }
        
        function formatCurrency(amount) {
            return amount ? new Intl.NumberFormat('fr-FR', {style: 'currency', currency: 'EUR'}).format(amount) : '-';
        }
        
        function showLoading() { loading.style.display = 'block'; }
        function hideLoading() { loading.style.display = 'none'; }
        function showResults() { results.style.display = 'block'; }
        function hideResults() { results.style.display = 'none'; }
        function showError(msg) { errorMessage.textContent = msg; errorMessage.style.display = 'block'; }
        function hideError() { errorMessage.style.display = 'none'; }
        function showSuccess(msg) { successMessage.textContent = msg; successMessage.style.display = 'block'; }
        function hideSuccess() { successMessage.style.display = 'none'; }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec l'interface HTML intégrée"""
    return HTMLResponse(content=HTML_INTERFACE, status_code=200)

@app.post("/extract-quote")
async def extract_quote(file: UploadFile = File(...)):
    """Extrait les informations d'un devis dentaire PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format PDF")
    
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extraire les données
        result = extract_dental_quote(temp_path)
        
        # Nettoyer
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dental-quote-extractor"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)