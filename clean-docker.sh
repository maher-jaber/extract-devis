Write-Host "Nettoyage complet de Docker..."

# Arrêter tous les conteneurs
docker stop (docker ps -aq) | Out-Null

# Supprimer tous les conteneurs
docker rm -f (docker ps -aq) | Out-Null

# Supprimer toutes les images
docker rmi -f (docker images -aq) | Out-Null

# Supprimer tous les volumes
docker volume rm -f (docker volume ls -q) | Out-Null

# Supprimer les réseaux non par défaut
docker network rm @(docker network ls -q | Where-Object {$_ -notin @("bridge","host","none")}) | Out-Null

# Supprimer les caches et builds intermédiaires
docker builder prune -af

Write-Host "Nettoyage terminé."
