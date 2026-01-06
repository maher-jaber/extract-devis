# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import json
import uuid
import logging
from typing import Dict, Any
import traceback
from datetime import datetime

# Import de notre version ultra-performante
from main import DentalQuoteExtractor, logger

app = FastAPI(
    title="Dental Quote Extractor API Ultra",
    description="API ultra-performante pour extraire les informations des devis dentaires PDF (texte et scannés)",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
extractor = DentalQuoteExtractor()
processing_cache: Dict[str, Dict[str, Any]] = {}

# Fichiers statiques
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Page d'accueil"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {
        "message": "Dental Quote Extractor API Ultra",
        "version": "2.0.0",
        "endpoints": {
            "documentation": "/api/docs",
            "extract": "/extract-quote",
            "batch_extract": "/batch-extract",
            "health": "/health",
            "status": "/status/{job_id}"
        }
    }

@app.get("/api")
async def api_info():
    """Informations sur l'API"""
    return {
        "api": "Dental Quote Extractor Ultra",
        "version": "2.0.0",
        "description": "Extraction avancée de devis dentaires PDF (texte et scannés)",
        "features": [
            "PDF texte et scannés",
            "OCR multi-moteur (PaddleOCR, EasyOCR, Tesseract)",
            "Analyse profonde de structure",
            "Extraction de tables complexes",
            "Correction automatique des erreurs OCR"
        ],
        "endpoints": [
            {"method": "POST", "path": "/extract-quote", "desc": "Extraire un devis unique"},
            {"method": "POST", "path": "/batch-extract", "desc": "Extraire plusieurs devis"},
            {"method": "GET", "path": "/health", "desc": "Vérifier l'état de l'API"},
            {"method": "GET", "path": "/status/{job_id}", "desc": "Vérifier le statut d'un job"}
        ]
    }

@app.post("/extract-quote")
async def extract_quote(
    file: UploadFile = File(...),
    debug: bool = False,
    return_raw_text: bool = False
):
    """
    Extraire les informations d'un devis dentaire PDF
    
    Args:
        file: Fichier PDF à analyser
        debug: Activer le mode debug (sauvegarde les fichiers intermédiaires)
        return_raw_text: Inclure le texte brut extrait dans la réponse
    
    Returns:
        JSON structuré avec toutes les informations extraites
    """
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Le fichier doit être au format PDF (.pdf)"
        )
    
    job_id = str(uuid.uuid4())
    processing_cache[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "start_time": datetime.now().isoformat(),
        "message": "Début de l'extraction"
    }
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        processing_cache[job_id]["message"] = "Analyse de la structure du PDF..."
        
        # Extraction avec la version ultra-performante
        result = extractor.extract_from_pdf(temp_path, debug=debug)
        
        # Nettoyer les fichiers temporaires
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Ajouter des métadonnées API
        result["api_metadata"] = {
            "job_id": job_id,
            "filename": file.filename,
            "file_size": len(content),
            "processing_time": (datetime.now() - datetime.fromisoformat(processing_cache[job_id]["start_time"])).total_seconds(),
            "debug_mode": debug
        }
        
        # Optionnel: inclure le texte brut
        if return_raw_text and "full_text" in result.get("raw_data", {}):
            result["raw_text"] = result["raw_data"]["full_text"]
        
        # Mettre à jour le cache
        processing_cache[job_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result_summary": {
                "treatments_found": len(result.get("treatments", [])),
                "total_amount": result.get("financial_summary", {}).get("honoraires_total"),
                "patient_name": result.get("basic_info", {}).get("patient", {}).get("nom")
            }
        })
        
        # Si debug activé, inclure des liens vers les fichiers de debug
        if debug:
            debug_files = []
            for fname in os.listdir("."):
                if fname.startswith(f"DEBUG_{os.path.basename(temp_path).replace('.pdf', '')}"):
                    debug_files.append(fname)
            result["debug_files"] = debug_files
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        # Mettre à jour le cache avec l'erreur
        processing_cache[job_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        
        logger.error(f"Erreur lors de l'extraction de {file.filename}: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Erreur lors de l'extraction",
                "message": str(e),
                "job_id": job_id,
                "filename": file.filename
            }
        )
    finally:
        # Nettoyer le cache après 1 heure
        if job_id in processing_cache:
            # Planifier la suppression (simplifiée)
            pass

@app.post("/batch-extract")
async def batch_extract(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Extraire plusieurs devis dentaires PDF en lot
    
    Args:
        files: Liste de fichiers PDF à analyser
    
    Returns:
        Job ID pour suivre la progression
    """
    
    job_id = str(uuid.uuid4())
    
    # Vérifier tous les fichiers
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"Le fichier {file.filename} doit être au format PDF"
            )
    
    # Initialiser le job
    processing_cache[job_id] = {
        "status": "processing",
        "type": "batch",
        "total_files": len(files),
        "processed_files": 0,
        "results": [],
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "message": f"Début du traitement de {len(files)} fichiers"
    }
    
    # Traiter en arrière-plan
    if background_tasks:
        background_tasks.add_task(
            process_batch_extraction,
            job_id,
            files
        )
    else:
        # Traitement synchrone
        process_batch_extraction(job_id, files)
    
    return {
        "job_id": job_id,
        "message": f"Traitement de {len(files)} fichiers démarré",
        "status_endpoint": f"/status/{job_id}"
    }

async def process_batch_extraction(job_id: str, files: list[UploadFile]):
    """Traiter l'extraction par lot en arrière-plan"""
    try:
        results = []
        errors = []
        
        for i, file in enumerate(files):
            try:
                processing_cache[job_id]["message"] = f"Traitement du fichier {i+1}/{len(files)}: {file.filename}"
                processing_cache[job_id]["current_file"] = file.filename
                
                # Sauvegarder temporairement le fichier
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                # Extraction
                result = extractor.extract_from_pdf(temp_path, debug=False)
                
                # Nettoyer
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                # Ajouter au résultat
                file_result = {
                    "filename": file.filename,
                    "success": True,
                    "data": result
                }
                results.append(file_result)
                
                # Mettre à jour la progression
                processing_cache[job_id]["processed_files"] = i + 1
                processing_cache[job_id]["progress"] = f"{(i + 1) / len(files) * 100:.1f}%"
                
            except Exception as e:
                error_info = {
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                }
                errors.append(error_info)
                logger.error(f"Erreur avec {file.filename}: {e}")
        
        # Finaliser le job
        processing_cache[job_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results": results,
            "errors": errors,
            "message": f"Traitement terminé: {len(results)} succès, {len(errors)} échecs"
        })
        
    except Exception as e:
        processing_cache[job_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "message": "Échec du traitement par lot"
        })

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Vérifier le statut d'un job d'extraction
    
    Args:
        job_id: ID du job retourné par /extract-quote ou /batch-extract
    
    Returns:
        Statut du job et résultats si disponible
    """
    if job_id not in processing_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID {job_id} non trouvé"
        )
    
    job_info = processing_cache[job_id]
    
    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "message": job_info.get("message", ""),
        "start_time": job_info.get("start_time"),
        "end_time": job_info.get("end_time")
    }
    
    # Ajouter des informations selon le type de job
    if job_info.get("type") == "batch":
        response.update({
            "type": "batch",
            "progress": job_info.get("progress", "0%"),
            "total_files": job_info.get("total_files", 0),
            "processed_files": job_info.get("processed_files", 0),
            "completed_files": len(job_info.get("results", [])),
            "failed_files": len(job_info.get("errors", []))
        })
    
    # Si le job est terminé, inclure les résultats
    if job_info["status"] == "completed":
        if job_info.get("type") == "batch":
            response["results"] = job_info.get("results", [])
            response["errors"] = job_info.get("errors", [])
        else:
            response["result_summary"] = job_info.get("result_summary", {})
    
    # Si le job a échoué, inclure l'erreur
    if job_info["status"] == "failed":
        response["error"] = job_info.get("error", "Erreur inconnue")
    
    return response

@app.get("/health")
async def health_check():
    """
    Vérifier l'état de santé de l'API
    
    Returns:
        Statut de l'API et informations système
    """
    try:
        # Vérifier l'espace disque
        import shutil
        disk_usage = shutil.disk_usage("/")
        disk_free_gb = disk_usage.free / (1024**3)
        
        # Vérifier la mémoire
        import psutil
        memory = psutil.virtual_memory()
        
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "disk_free_gb": round(disk_free_gb, 2),
                "disk_used_percent": round((disk_usage.used / disk_usage.total) * 100, 2),
                "memory_used_percent": round(memory.percent, 2),
                "memory_available_gb": round(memory.available / (1024**3), 2)
            },
            "api": {
                "version": "2.0.0",
                "active_jobs": len([j for j in processing_cache.values() if j["status"] == "processing"]),
                "total_jobs": len(processing_cache)
            },
            "ocr_engines": {
                "paddleocr": True,  # À adapter selon votre initialisation
                "easyocr": True,
                "tesseract": True
            }
        }
        
        # Vérifier les prérequis
        if disk_free_gb < 1:
            health_info["status"] = "warning"
            health_info["warning"] = "Espace disque faible"
        elif memory.percent > 90:
            health_info["status"] = "warning"
            health_info["warning"] = "Moire élevée"
        
        return health_info
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/download-result/{job_id}")
async def download_result(job_id: str):
    """
    Télécharger le résultat d'un job au format JSON
    
    Args:
        job_id: ID du job
    
    Returns:
        Fichier JSON téléchargeable
    """
    if job_id not in processing_cache:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job_info = processing_cache[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Le job n'est pas encore terminé (statut: {job_info['status']})"
        )
    
    # Pour les jobs batch, créer un fichier de résultats
    if job_info.get("type") == "batch":
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "total_files": job_info.get("total_files"),
            "successful": len(job_info.get("results", [])),
            "failed": len(job_info.get("errors", [])),
            "results": job_info.get("results", []),
            "errors": job_info.get("errors", []),
            "processing_time": job_info.get("processing_time")
        }
        filename = f"batch_results_{job_id}.json"
    else:
        # Pour les jobs simples, on ne stocke pas les résultats complets dans le cache
        # Dans une vraie implémentation, on les sauvegarderait dans un fichier
        raise HTTPException(
            status_code=501, 
            detail="Téléchargement des résultats simples non implémenté. Utilisez directement la réponse de /extract-quote."
        )
    
    # Créer un fichier temporaire JSON
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_file:
        json.dump(result_data, temp_file, indent=2, ensure_ascii=False, default=str)
        temp_path = temp_file.name
    
    # Retourner le fichier
    return FileResponse(
        temp_path,
        media_type='application/json',
        filename=filename,
        background=lambda: os.unlink(temp_path)  # Nettoyer après envoi
    )

@app.get("/cleanup")
async def cleanup_old_jobs(hours: int = 24):
    """
    Nettoyer les anciens jobs du cache
    
    Args:
        hours: Nettoyer les jobs plus vieux que X heures (défaut: 24)
    
    Returns:
        Nombre de jobs nettoyés
    """
    try:
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        jobs_to_remove = []
        for job_id, job_info in processing_cache.items():
            end_time = job_info.get("end_time")
            if end_time:
                job_end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if job_end < cutoff_time:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del processing_cache[job_id]
        
        return {
            "cleaned_jobs": len(jobs_to_remove),
            "remaining_jobs": len(processing_cache),
            "cutoff_hours": hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Middleware pour le logging des requêtes
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    
    # Exécuter la requête
    response = await call_next(request)
    
    # Calculer le temps de réponse
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Logger
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Ajouter le temps de réponse dans les headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Dental Quote Extractor API Ultra - Version 2.0.0")
    print("="*60)
    print("Démarré sur http://localhost:8000")
    print("Documentation: http://localhost:8000/api/docs")
    print("="*60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )