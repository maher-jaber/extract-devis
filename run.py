#!/usr/bin/env python3
# run.py - Version corrigée et robuste
import uvicorn
import os
import sys

def setup_static_files():
    """Configurer les fichiers statiques de manière robuste"""
    # Créer le répertoire static
    os.makedirs("static", exist_ok=True)
    
    # Vérifier si index.html existe à la racine
    root_index = "index.html"
    static_index = "static/index.html"
    
    # Si index.html existe à la racine, le copier dans static/
    if os.path.exists(root_index):
        print(f"Copie de {root_index} vers {static_index}")
        try:
            with open(root_index, 'r', encoding='utf-8') as src:
                content = src.read()
            with open(static_index, 'w', encoding='utf-8') as dst:
                dst.write(content)
            print(f"✓ {root_index} copié avec succès dans static/")
        except Exception as e:
            print(f"✗ Erreur lors de la copie: {e}")
            return False
    # Si index.html existe déjà dans static/
    elif os.path.exists(static_index):
        print(f"✓ {static_index} existe déjà")
        return True
    # Si index.html n'existe nulle part, créer un fichier minimal
    else:
        print("⚠ Aucun fichier index.html trouvé. Création d'un fichier minimal...")
        html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dental Quote Extractor</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 50px auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 3px dashed #3498db; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; cursor: pointer; }
        .upload-area:hover { background: #f8f9fa; }
        .btn { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #2980b9; }
        .result { margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 5px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦷 Dental Quote Extractor</h1>
        <p>API pour extraire les informations des devis dentaires PDF</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <h3>📤 Télécharger un devis PDF</h3>
            <p>Cliquez ici ou glissez-déposez votre fichier PDF</p>
            <input type="file" id="fileInput" accept=".pdf" style="display: none;" onchange="uploadFile()">
        </div>
        
        <div id="loading" style="display: none; text-align: center;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
            <p>Extraction en cours...</p>
        </div>
        
        <div id="result" class="result"></div>
    </div>
    
    <script>
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            if (!fileInput.files[0]) return;
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            loadingDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/extract-quote', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    resultDiv.innerHTML = '<h3 style="color: #27ae60;">✅ Extraction réussie !</h3>' +
                                         '<pre style="background: white; padding: 15px; border-radius: 5px; overflow: auto;">' + 
                                         JSON.stringify(data, null, 2) + '</pre>';
                    resultDiv.style.display = 'block';
                } else {
                    const error = await response.json();
                    resultDiv.innerHTML = '<h3 style="color: #e74c3c;">❌ Erreur</h3>' +
                                         '<p>' + error.detail + '</p>';
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                resultDiv.innerHTML = '<h3 style="color: #e74c3c;">❌ Erreur réseau</h3>' +
                                     '<p>' + error.message + '</p>';
                resultDiv.style.display = 'block';
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
        
        // Gestion du drag and drop
        const uploadArea = document.querySelector('.upload-area');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.style.borderColor = '#2ecc71';
                uploadArea.style.background = '#f8fffe';
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.style.borderColor = '#3498db';
                uploadArea.style.background = '';
            }, false);
        });
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                document.getElementById('fileInput').files = files;
                uploadFile();
            }
        }
    </script>
</body>
</html>"""
        
        try:
            with open(static_index, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✓ Fichier {static_index} créé avec succès")
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la création: {e}")
            return False
    
    return True

def check_dependencies():
    """Vérifier les dépendances essentielles"""
    try:
        import fastapi
        import uvicorn
        import paddleocr
        import easyocr
        import pdfplumber
        import fitz
        print("✓ Toutes les dépendances sont disponibles")
        return True
    except ImportError as e:
        print(f"✗ Dépendance manquante: {e}")
        print("Veuillez exécuter: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Dental Quote Extractor - Démarrage")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        sys.exit(1)
    
    # Configurer les fichiers statiques
    if not setup_static_files():
        print("✗ Impossible de configurer les fichiers statiques")
        sys.exit(1)
    
    # Vérifier que index.html existe bien
    if not os.path.exists("static/index.html"):
        print("✗ ERREUR: static/index.html n'existe toujours pas!")
        sys.exit(1)
    
    print("✓ Configuration terminée")
    print("✓ Démarrage du serveur sur http://0.0.0.0:8000")
    print("=" * 50)
    
    # Lancer l'API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # False en production/docker
        log_level="info",
        access_log=True
    )