#!/usr/bin/env python3
import uvicorn
import os

if __name__ == "__main__":
    # Créer le répertoire static s'il n'existe pas
    os.makedirs("static", exist_ok=True)
    
    # Vérifier que index.html existe dans static
    if not os.path.exists("static/index.html"):
        print("ERREUR: Le fichier static/index.html n'existe pas!")
        print("Veuillez déplacer votre fichier index.html dans le dossier static/")
        exit(1)
    
    # Lancer l'API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )