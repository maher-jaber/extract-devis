from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques (templates)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/templates/index.html")
async def frontend():
    return FileResponse("static/index.html")

@app.post("/extract-quote")
async def extract_quote(file: UploadFile = File(...)):
    """
    Extrait les informations d'un devis dentaire PDF
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Le fichier doit être au format PDF"
        )
    
    try:
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extraire les données
        result = extract_dental_quote(temp_path)
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'extraction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)