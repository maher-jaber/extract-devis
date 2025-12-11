#!/usr/bin/env python3
# main2.py - Version RTX 2050 optimisée
# Extraction robuste de devis dentaires (PDF texte + PDF scannés)

import cv2
import re
import json
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pdfplumber
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import easyocr
import os
import torch
import paddle;
# -------------------------
# Configuration GPU RTX 2050
# -------------------------
device = 'cuda' if paddle.device.get_device() else 'cpu'
print(f"Device détecté: {device}")

# -------------------------
# Initialisation OCR optimisée pour RTX 2050 (4GB VRAM)
# -------------------------
try:
    # PaddleOCR configuré pour 4GB VRAM
    ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='fr'
    )

    print("✓ PaddleOCR initialisé (GPU)" if device == 'cuda' else "✓ PaddleOCR initialisé (CPU)")
except Exception as e:
    print(f"✗ Erreur PaddleOCR: {e}")
    ocr_engine = None

try:
    # EasyOCR plus léger
    easyocr_reader = easyocr.Reader(['fr'], gpu=True)
    print("✓ EasyOCR initialisé (GPU)" if ocr_engine != None else "✓ EasyOCR initialisé (CPU)")
except Exception as e:
    print(f"✗ Erreur EasyOCR: {e}")
    easyocr_reader = None

# -------------------------
# Utilitaires généraux - OPTIMISÉS
# -------------------------

def correct_ocr_codes(text: str) -> str:
    """Corrige les codes HBLD mal reconnus par l'OCR"""
    corrections = {
        r'HBLD0[5-9]0': 'HBLD350',  # HBLD050 → HBLD350
        r'HBLD\d{2}': lambda m: m.group(0),  # Garder les autres
        r'HB LD': 'HBLD',
        r'H BL D': 'HBLD',
        r'HB LD(\d+)': r'HBLD\1',
    }
    
    for pattern, replacement in corrections.items():
        if callable(replacement):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def safe_strip(s: Optional[str]) -> Optional[str]:
    return str(s).strip() if s is not None and str(s).strip() else None

def norm_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def to_float_amount(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        # Gérer les formats français: 1 234,56 ou 1234,56
        s = str(s).replace('\u00A0', '').replace(' ', '').replace(',', '.')
        # Nettoyer tout sauf chiffres et point
        s = re.sub(r'[^\d\.\-]', '', s)
        return float(s) if s else None
    except:
        return None

def find_first_match(patterns: List[str], text: str, flags=0) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return norm_whitespace(m.group(1)) if m.groups() else norm_whitespace(m.group(0))
    return None

# -------------------------
# PRÉ-TRAITEMENT AMÉLIORÉ pour RTX 2050
# -------------------------
def preprocess_image_for_ocr(img_path: str) -> Optional[np.ndarray]:
    """Pré-traitement optimisé pour documents scannés"""
    try:
        # Charger l'image en couleur
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Redimensionner pour RTX 2050 (max 1200px)
        h, w = img.shape[:2]
        if h > 1200 or w > 1200:
            scale = min(1200/h, 1200/w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Améliorer le contraste avec CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Réduction du bruit
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Seuillage adaptatif
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        return binary
        
    except Exception as e:
        print(f"Erreur pré-traitement: {e}")
        return None

# -------------------------
# OCR AMÉLIORÉ - Capture MAXIMALE de texte
# -------------------------
def run_ocr_complete(img_path: str) -> str:
    """Exécute PaddleOCR et EasyOCR pour capturer le maximum de texte"""
    all_text = []
    
    # Pré-traitement
    processed_img = preprocess_image_for_ocr(img_path)
    if processed_img is not None:
        temp_path = "temp_processed.png"
        cv2.imwrite(temp_path, processed_img)
        img_to_ocr = temp_path
    else:
        img_to_ocr = img_path
    
    # 1. PADDLEOCR (meilleur pour le français)
    if ocr_engine:
        try:
            result = ocr_engine.ocr(img_to_ocr)
            if result:
                page_texts = []
                for line in result[0]:
                    if len(line) >= 2:
                        text_info = line[1]
                        if text_info and len(text_info) >= 2:
                            text = str(text_info[0]).strip()
                            confidence = float(text_info[1])
                            # Seuil BAS pour capturer PLUS de texte
                            if confidence > 0.05 and text:  # Seuil à 5% seulement!
                                page_texts.append(text)
                
                if page_texts:
                    paddle_text = " ".join(page_texts)
                    all_text.append(f"[PADDLE] {paddle_text}")
                    print(f"    PaddleOCR: {len(paddle_text)} caractères")
        except Exception as e:
            print(f"    PaddleOCR erreur: {e}")
    
    # 2. EASYOCR (bon complément)
    if easyocr_reader:
        try:
            results = easyocr_reader.readtext(
                img_to_ocr,
                paragraph=False,
                width_ths=0.5,
                height_ths=0.5,
                min_size=5,
                text_threshold=0.1  # Seuil bas
            )
            
            easy_texts = []
            for bbox, text, conf in results:
                if conf > 0.1 and text.strip():  # Seuil à 10%
                    easy_texts.append(text.strip())
            
            if easy_texts:
                easy_text = " ".join(easy_texts)
                all_text.append(f"[EASY] {easy_text}")
                print(f"    EasyOCR: {len(easy_text)} caractères")
                
        except Exception as e:
            print(f"    EasyOCR erreur: {e}")
    
    # Nettoyage
    if 'temp_path' in locals() and os.path.exists(temp_path):
        os.remove(temp_path)
    
    return " ".join(all_text)

# -------------------------
# OCR PDF COMPLET - Amélioré
# -------------------------
def ocr_from_pdf_enhanced(pdf_path: str) -> str:
    """OCR amélioré pour capturer TOUT le texte"""
    doc = fitz.open(pdf_path)
    all_pages_text = []
    
    print(f"  OCR amélioré sur {len(doc)} pages...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_texts = []
        
        # Essayer TROIS niveaux de zoom différents
        for zoom in [0.6, 0.8, 1.0]:  # Plus de zooms pour plus de capture
            try:
                # Rendre la page
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Sauvegarder
                temp_img = f"temp_page_{page_num}_z{int(zoom*10)}.png"
                pix.save(temp_img)
                
                # OCR sur cette image
                ocr_text = run_ocr_complete(temp_img)
                
                if ocr_text and len(ocr_text) > 50:  # Au moins 50 caractères
                    page_texts.append(f"--- Zoom {zoom} ---\n{ocr_text}")
                    print(f"    Page {page_num+1} zoom {zoom}: {len(ocr_text)} caractères")
                
                # Nettoyer
                os.remove(temp_img)
                
            except Exception as e:
                print(f"    Page {page_num+1} zoom {zoom} erreur: {e}")
                continue
        
        # Garder le texte le plus long
        if page_texts:
            best_text = max(page_texts, key=len)
            all_pages_text.append(f"\n{'='*60}\nPAGE {page_num+1}\n{'='*60}\n{best_text}")
    
    doc.close()
    
    # Fusionner tout le texte
    full_text = "\n".join(all_pages_text)
    
    # POST-TRAITEMENT CRITIQUE pour améliorer la qualité
    if full_text:
        # 1. Corriger les erreurs OCR courantes
        corrections = {
            'H B L D': 'HBLD',
            'HBL D': 'HBLD',
            'H BLD': 'HBLD',
            'H B L': 'HBL',
            'couronn e': 'couronne',
            'dentair e': 'dentaire',
            'prothè se': 'prothèse',
            'zircon e': 'zircone',
            'pose d une': 'pose d\'une',
            'couronn€': 'couronne',
            'dentair€': 'dentaire',
            '€uro': 'euro',
            'l o': 'l\'o',
            'd une': 'd\'une',
            'qu une': 'qu\'une',
            'sans rest€': 'sans reste',
            'à charg€': 'à charge',
        }
        
        for wrong, correct in corrections.items():
            full_text = full_text.replace(wrong, correct)
        
        # 2. Normaliser les espaces et sauts de ligne
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
        
        # 3. Ajouter des séparateurs pour les tables
        lines = full_text.split('\n')
        enhanced_lines = []
        for line in lines:
            # Si la ligne contient un HBLD et des montants, c'est probablement une ligne de table
            if re.search(r'HBLD\d+', line, re.IGNORECASE) and re.search(r'\d+[.,]\d{2}', line):
                enhanced_lines.append(f"[TABLE_LINE] {line}")
            else:
                enhanced_lines.append(line)
        
        full_text = "\n".join(enhanced_lines)
    
    print(f"  OCR terminé: {len(full_text)} caractères au total")
    return full_text
def extract_basic_fields(full_text: str) -> Dict[str, Any]:
    # Créer une version normalisée pour la recherche
    full_text_norm = re.sub(r'\s+', ' ', full_text)
    
    # ---- Dentiste ----
    nom_praticien = find_first_match([
        r"Nom\s*Pr[ée]nom\s*[:\-]?\s*([A-Z][^:\n]+?(?=\s*(?:Identifiant|RPPS|ADELI|Raison)))",
        r"Nom[^:\n]*[:\-]\s*([A-Z][^:\n]+?(?=\s*(?:Prénom|Prenom|Identifiant)))",
        r"Chirurgien[^:\n]*[:\-]\s*([A-Z][^:\n]+)",
        r"Docteur\s+([A-Z][^:\n]+)"
    ], full_text_norm, flags=re.IGNORECASE)
    
    if nom_praticien:
        nom_praticien = re.split(r'Nom et prénom|Date de naissance|Identifiant|RPPS|ADELI', 
                               nom_praticien, flags=re.IGNORECASE)[0].strip()

    rpps = find_first_match([
        r"RPPS[^0-9]*([0-9]{8,})",
        r"identifiant du praticien rpps[^0-9]*([0-9]{8,})",
        r"R\s*P\s*P\s*S[^0-9]*([0-9]{8,})"
    ], full_text_norm, flags=re.IGNORECASE)
    
    adeli = find_first_match([
        r"ADELI[^0-9]*([0-9]{8,})",
        r"adeli[^0-9]*([0-9]{8,})",
        r"A\s*D\s*E\s*L\s*I[^0-9]*([0-9]{8,})"
    ], full_text_norm, flags=re.IGNORECASE)

    # Adresse dentiste
    adresse_dentiste = find_first_match([
        r"Raison[^:\n]*[:\-]\s*([^:\n]+?(?=\s*(?:Numéro|N°|Patient|DEVIS)))",
        r"Adresse[^:\n]*[:\-]\s*([^:\n]+)"
    ], full_text_norm, flags=re.IGNORECASE)

    # FINESS dentiste
    finess_match = re.search(r'(?:FINESS|N[°º]\s+[^:\n]*)[:\-]?\s*(\d+)', full_text_norm, re.IGNORECASE)
    finess_dentiste = finess_match.group(1) if finess_match else None
    
    # ---- Patient ----
    patient_nom = find_first_match([
        r"Patient[^:\n]*[:\-]\s*([A-Z][^:\n]+?(?=\s*(?:Date|Naissance|N°)))",
        r"Nom[^:\n]*[:\-]\s*([A-Z][^:\n]+?(?=\s*(?:Date|Naissance)))",
        r"Bénéficiaire[^:\n]*[:\-]\s*([A-Z][^:\n]+)"
    ], full_text_norm, flags=re.IGNORECASE)
    
    if not patient_nom:
        pm = re.search(r"([A-Z][A-Z\s\-]+)\s+(?:Date\s+de\s+naissance|Naissance)", full_text_norm, re.IGNORECASE)
        if pm:
            patient_nom = pm.group(1).strip()

    date_naissance = find_first_match([
        r"Date[^:\n]*naissance[^:\n]*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Naissance[^:\n]*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Né\(e\)[^:\n]*(\d{2}/\d{2}/\d{4})"
    ], full_text_norm, flags=re.IGNORECASE)

    num_secu = find_first_match([
        r"S[ée]curit[ée][^:\n]*[:\-]?\s*([0-9]{13,15})",
        r"N[°º][^:\n]*[:\-]?\s*([0-9]{13,15})",
        r"(?:NIR|Numéro)[^:\n]*[:\-]?\s*([0-9]{13,15})"
    ], full_text_norm, flags=re.IGNORECASE)

    adresse_patient = find_first_match([
        r"Adresse[^:\n]*patient[^:\n]*[:\-]\s*([^:\n]+)",
        r"Domicile[^:\n]*[:\-]\s*([^:\n]+)"
    ], full_text_norm, flags=re.IGNORECASE)

    # ---- Devis ----
    numero_devis = find_first_match([
        r"Devis[^:\n]*[:\-]\s*([0-9A-Za-z\-]+)",
        r"N[°º][^:\n]*devis[^:\n]*[:\-]\s*([0-9]+)",
        r"R[ée]f[ée]rence[^:\n]*[:\-]\s*([0-9A-Za-z\-]+)"
    ], full_text_norm, flags=re.IGNORECASE)

    date_devis = find_first_match([
        r"Date[^:\n]*devis[^:\n]*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Devis[^:\n]*[:\-].*?(\d{2}/\d{2}/\d{4})",
        r"Fait[^:\n]*[:\-].*?(\d{2}/\d{2}/\d{4})"
    ], full_text_norm, flags=re.IGNORECASE)

    validite = find_first_match([
        r"Valable[^:\n]*[:\-].*?(\d{2}/\d{2}/\d{4})",
        r"Validit[ée][^:\n]*[:\-].*?(\d{2}/\d{2}/\d{4})"
    ], full_text_norm, flags=re.IGNORECASE)
           
    return {
        "dentiste": {
            "nom_praticien": nom_praticien,
            "rpps": rpps,
            "adeli": adeli,
            "finess": finess_dentiste,
            "adresse": adresse_dentiste,
            "numero_etablissement": finess_dentiste
        },
        "patient": {
            "nom": patient_nom,
            "date_naissance": date_naissance,
            "numero_securite_sociale": num_secu,
            "adresse": adresse_patient,
            "numero_etablissement": None
        },
        "devis": {
            "numero": numero_devis,
            "date_devis": date_devis,
            "validite": validite
        }
    }

# -------------------------
# Extraction totaux financiers
# -------------------------
def extract_totals_from_tables(tables_data: List[Dict[str, Any]], full_text: str) -> Dict[str, Optional[float]]:
    totals = {
        "honoraires_total": None,
        "prix_dispositif_medical": None,
        "base_remboursement_amo": None,
        "montant_rembourse_amo": None,
        "reste_a_charge": None
    }
    
    # Chercher dans les tables d'abord
    for tinfo in tables_data:
        table = tinfo['table']
        for row in reversed(table):
            if not row or not any(row):
                continue
            row_text = ' '.join([str(c) for c in row if c])
            if re.search(r'\bTOTAL\b', row_text, re.IGNORECASE):
                found = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2}))', row_text)
                nums = [to_float_amount(x) for x in found]
                nums = [n for n in nums if n is not None]
                if len(nums) >= 1: totals["honoraires_total"] = nums[0]
                if len(nums) >= 2: totals["prix_dispositif_medical"] = nums[1]
                if len(nums) >= 3: totals["base_remboursement_amo"] = nums[2]
                if len(nums) >= 4: totals["reste_a_charge"] = nums[3]
                if totals["honoraires_total"] is not None:
                    return totals

    # Si pas trouvé dans les tables, chercher dans le texte
    for pattern in [r'TOTAL[^\n]*€[^\n]*([0-9\s.,]+)', r'Total[^€]*€[^\n]*([0-9\s.,]+)']:
        matches = re.finditer(pattern, full_text, flags=re.IGNORECASE)
        for m in matches:
            if m.group(1):
                found = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2}))', m.group(1))
                nums = [to_float_amount(x) for x in found if to_float_amount(x) is not None]
                
                if len(nums) >= 1: totals["honoraires_total"] = totals["honoraires_total"] or nums[0]
                if len(nums) >= 2: totals["prix_dispositif_medical"] = totals["prix_dispositif_medical"] or nums[1]
                if len(nums) >= 3: totals["base_remboursement_amo"] = totals["base_remboursement_amo"] or nums[2]
                if len(nums) >= 4: totals["reste_a_charge"] = totals["reste_a_charge"] or nums[3]
    
    return totals

# -------------------------
# Infos complémentaires
# -------------------------
def extract_consent_and_misc(full_text: str) -> Dict[str, Any]:
    text_norm = re.sub(r'\s+', ' ', full_text)

    # Consentement éclairé
    consent_present = bool(re.search(
        r'(consentement|accord).*?(éclair|informé)',
        text_norm, re.IGNORECASE
    ))

    # Organisme complémentaire
    mutuelle = find_first_match([
        r"Mutuelle[^:\n]*[:\-]?\s*([A-Z0-9][A-Za-z0-9 \-]+)",
        r"Complémentaire[^:\n]*[:\-]?\s*([A-Z0-9][A-Za-z0-9 \-]+)"
    ], text_norm, flags=re.IGNORECASE)

    # Base AMO – brute si non trouvée ailleurs
    base_amo = find_first_match([
        r"Base\s+AMO[^0-9]*([0-9\s.,]+)",
        r"Remboursement\s+AMO[^0-9]*([0-9\s.,]+)"
    ], text_norm, flags=re.IGNORECASE)
    base_amo = to_float_amount(base_amo)

    # Acompte (si payé ou demandé)
    acompte = find_first_match([
        r"Acompte[^0-9]*([0-9\s.,]+)",
        r"Versement\s+initial[^0-9]*([0-9\s.,]+)"
    ], text_norm, flags=re.IGNORECASE)
    acompte = to_float_amount(acompte)

    # Mentions obligatoires légales
    mentions_legales = bool(re.search(
        r"(information\s+précontractuelle|notice\s+HAS|code\s+de\s+la\s+santé)",
        text_norm, re.IGNORECASE
    ))

    return {
        "consentement_eclaire": consent_present,
        "mutuelle": mutuelle,
        "base_amo_detectee": base_amo,
        "acompte": acompte,
        "mentions_legales_presentes": mentions_legales
    }

# -------------------------
# EXTRACTION PDF - Optimisée
# -------------------------
def extract_pdf_text_and_tables_optimized(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Version optimisée de l'extraction PDF"""
    
    # Essayer pdfplumber d'abord
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            tables = []
            
            for i, page in enumerate(pdf.pages):
                # Texte
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text_content += page_text + "\n\n"
                
                # Tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if any(any(cell for cell in row if cell) for row in table):
                        tables.append({"page": i+1, "table": table, "source": "pdfplumber"})
            
            # Vérifier si c'est un vrai PDF texte
            is_text_pdf = (
                len(text_content.strip()) > 1000 or  # Beaucoup de texte
                ('DEVIS' in text_content and 'dentaire' in text_content.lower()) or
                len(re.findall(r'HBLD\d+', text_content, re.IGNORECASE)) >= 2
            )
            
            if is_text_pdf:
                print(f"✓ PDF texte détecté: {len(text_content)} caractères, {len(tables)} tables")
                return text_content, tables
            else:
                print(f"✗ PDF probablement scanné (peu de texte: {len(text_content)} caractères)")
                
    except Exception as e:
        print(f"pdfplumber échoué: {e}")
    
    # Si pdfplumber échoue ou peu de texte -> OCR
    print("→ Passage à l'OCR amélioré...")
    ocr_text = ocr_from_pdf_enhanced(pdf_path)
    
    # CORRIGER LES CODES OCR
    ocr_text = correct_ocr_codes(ocr_text)
    
    # Pour l'OCR, on n'a pas de vraies tables, on les simule
    tables_from_ocr = []
    lines = ocr_text.split('\n')
    
    for i, line in enumerate(lines):
        if '[TABLE_LINE]' in line:
            # Extraire les parties de la ligne
            line_clean = line.replace('[TABLE_LINE] ', '')
            # Essayer de diviser en colonnes
            parts = re.split(r'\s{2,}', line_clean)  # Au moins 2 espaces = séparateur
            if len(parts) >= 4:
                tables_from_ocr.append({
                    "page": 1,
                    "table": [parts],  # Une ligne = une table
                    "source": "ocr_simulated"
                })
    
    return ocr_text, tables_from_ocr
# -------------------------
# EXTRACTION TRAITEMENTS - TRÈS AMÉLIORÉE
# -------------------------
def extract_treatments_aggressive(full_text: str, tables_data: List[Dict]) -> List[Dict[str, Any]]:
    """Extraction ULTRA FLEXIBLE des traitements"""
    treatments = []
    
    print("  Extraction FLEXIBLE des traitements...")
    
    # 1. D'abord, chercher TOUTES les occurrences de HBLD dans le texte
    hbld_pattern = r'HBLD\d{3,}'
    all_hbld_matches = list(re.finditer(hbld_pattern, full_text, re.IGNORECASE))
    
    print(f"  → {len(all_hbld_matches)} codes HBLD trouvés dans le texte")
    
    # 2. Pour chaque code HBLD trouvé, chercher le contexte
    for match in all_hbld_matches:
        code = match.group(0).upper()
        
        # Prendre un gros contexte autour du code (200 caractères avant et après)
        start_context = max(0, match.start() - 100)
        end_context = min(len(full_text), match.end() + 150)
        context = full_text[start_context:end_context]
        
        # Nettoyer le contexte
        context = re.sub(r'\s+', ' ', context)
        
        # Chercher TOUS les montants dans ce contexte
        amounts = re.findall(r'\d+[.,]\d{2}', context)
        
        # Si on a au moins 3 montants, c'est probablement un traitement
        if len(amounts) >= 3:
            # Chercher un numéro de dent (1-48) près du code
            # On cherche dans les 30 caractères avant le code
            before_code = full_text[max(0, match.start()-30):match.start()]
            dent_match = re.search(r'\b(1[0-9]|2[0-9]|3[0-9]|4[0-8]|[1-9])\b', before_code)
            dent = dent_match.group(1) if dent_match else None
            
            # Si pas trouvé avant, chercher après
            if not dent:
                after_code = full_text[match.end():min(len(full_text), match.end()+30)]
                dent_match = re.search(r'\b(1[0-9]|2[0-9]|3[0-9]|4[0-8]|[1-9])\b', after_code)
                dent = dent_match.group(1) if dent_match else None
            
            # Description simple (50 premiers caractères du contexte)
            description = context[:100].strip()
            
            # Prendre les montants (au moins 4 si possible)
            treatment_data = {
                "code_acte": code,
                "dent": dent or "?",
                "description": description,
            }
            
            # Ajouter les montants disponibles
            if len(amounts) >= 1:
                treatment_data["honoraires"] = to_float_amount(amounts[0])
            if len(amounts) >= 2:
                treatment_data["prix_dispositif_medical"] = to_float_amount(amounts[1])
            if len(amounts) >= 3:
                treatment_data["base_remboursement"] = to_float_amount(amounts[2])
            if len(amounts) >= 4:
                treatment_data["reste_a_charge"] = to_float_amount(amounts[3])
            elif len(amounts) >= 1:
                # Si pas 4 montants, le dernier est peut-être le reste
                treatment_data["reste_a_charge"] = to_float_amount(amounts[-1])
            
            treatments.append(treatment_data)
    
    # 3. Recherche par LIGNES - méthode simple
    if len(treatments) < 3:  # Si pas assez trouvé
        print("  Recherche par lignes...")
        lines = full_text.split('\n')
        
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 30:  # Ignorer les lignes trop courtes
                continue
            
            # Vérifier si la ligne contient un HBLD
            if re.search(r'HBLD\d{3}', line_clean, re.IGNORECASE):
                # Chercher tous les montants dans cette ligne
                amounts = re.findall(r'\d+[.,]\d{2}', line_clean)
                
                if len(amounts) >= 2:  # Au moins 2 montants
                    # Code HBLD
                    code_match = re.search(r'(HBLD\d{3})', line_clean, re.IGNORECASE)
                    code = code_match.group(1).upper() if code_match else "HBLDxxx"
                    
                    # Numéro de dent (chercher n'importe quel nombre 1-48)
                    dent_match = re.search(r'\b(1[0-9]|2[0-9]|3[0-9]|4[0-8]|[1-9])\b', line_clean)
                    dent = dent_match.group(1) if dent_match else "?"
                    
                    treatments.append({
                        "code_acte": code,
                        "dent": dent,
                        "description": line_clean[:80],
                        "honoraires": to_float_amount(amounts[0]),
                        "reste_a_charge": to_float_amount(amounts[-1])
                    })
    
    # 4. DÉDUPLICATION intelligente
    print(f"  → {len(treatments)} traitements avant déduplication")
    
    unique_treatments = []
    seen = set()
    
    for t in treatments:
        # Créer une clé unique basée sur le code et les montants
        if t.get("honoraires"):
            key = f"{t['code_acte']}_{t.get('honoraires')}"
        else:
            key = f"{t['code_acte']}_{t.get('dent')}"
        
        if key not in seen:
            seen.add(key)
            unique_treatments.append(t)
    
    print(f"  → {len(unique_treatments)} traitements uniques après déduplication")
    
    # 5. AFFICHER CE QU'ON A TROUVÉ (pour debug)
    if unique_treatments:
        print("\n  Détail des traitements trouvés:")
        for i, t in enumerate(unique_treatments, 1):
            print(f"    {i}. {t['code_acte']} - Dent {t.get('dent', '?')}: €{t.get('honoraires', '?')} (reste: €{t.get('reste_a_charge', '?')})")
    
    return unique_treatments
# -------------------------
# FONCTION PRINCIPALE - FINALE
# -------------------------
def extract_dental_quote(pdf_path: str) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"EXTRACTION DEVIS DENTAIRE: {os.path.basename(pdf_path)}")
    print(f"{'='*80}")
    
    try:
        # 1. EXTRACTION TEXTE
        print("1. Extraction du texte...")
        full_text, tables_data = extract_pdf_text_and_tables_optimized(pdf_path)
        
        # Sauvegarde DEBUG
        debug_file = f"DEBUG_{os.path.basename(pdf_path).replace('.pdf', '.txt')}"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"   ✓ Texte sauvegardé: {debug_file}")
        print(f"   ✓ Longueur: {len(full_text)} caractères")
        
        # 2. EXTRACTION DONNÉES DE BASE
        print("\n2. Extraction des informations de base...")
        basic = extract_basic_fields(full_text)  # Garder ta fonction actuelle
        
        # 3. EXTRACTION TRAITEMENTS (NOUVELLE MÉTHODE)
        print("\n3. Extraction des traitements...")
        treatments = extract_treatments_aggressive(full_text, tables_data)
        
        # 4. EXTRACTION TOTAUX
        print("\n4. Extraction des totaux...")
        totals = extract_totals_from_tables(tables_data, full_text)
        
        # Si pas de totaux, calculer
        if not totals.get("honoraires_total") and treatments:
            honoraires_sum = sum(t.get("honoraires", 0) or 0 for t in treatments)
            reste_sum = sum(t.get("reste_a_charge", 0) or 0 for t in treatments)
            totals["honoraires_total"] = honoraires_sum
            totals["reste_a_charge"] = reste_sum
        
        # 5. INFORMATIONS COMPLÉMENTAIRES
        misc = extract_consent_and_misc(full_text)
        
        # 6. RÉSULTAT FINAL
        result = {
            "source_file": pdf_path,
            "extraction_info": {
                "text_length": len(full_text),
                "treatments_found": len(treatments),
                "debug_file": debug_file
            },
            "dentiste": basic.get("dentiste"),
            "patient": basic.get("patient"),
            "devis": basic.get("devis"),
            "traitements": treatments,
            "financier": totals,
            "informations_complementaires": misc
        }
        
        # AFFICHAGE RÉCAPITULATIF
        print(f"\n{'='*80}")
        print("RÉSULTAT DE L'EXTRACTION")
        print(f"{'='*80}")
        print(f"• Dentiste: {basic.get('dentiste', {}).get('nom_praticien', 'Non trouvé')}")
        print(f"• Patient: {basic.get('patient', {}).get('nom', 'Non trouvé')}")
        print(f"• Devis n°: {basic.get('devis', {}).get('numero', 'Non trouvé')}")
        print(f"• Traitements trouvés: {len(treatments)}")
        
        if treatments:
            print("• Détail des traitements:")
            for i, t in enumerate(treatments, 1):
                print(f"  {i}. {t.get('code_acte', 'N/A')} - Dent {t.get('dent', 'N/A')}: "
                      f"€{t.get('honoraires', 0)} (reste: €{t.get('reste_a_charge', 0)})")
        
        print(f"• Total honoraires: €{totals.get('honoraires_total', 0)}")
        print(f"• Reste à charge: €{totals.get('reste_a_charge', 0)}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "source_file": pdf_path,
            "error": str(e),
            "extraction_info": {"error": "Échec de l'extraction"}
        }

# -------------------------
# EXÉCUTION
# -------------------------
def analyse_devis(pdf_path: str) -> Dict[str, Any]:
    print(f"\n=== ANALYSE DU DOCUMENT : {pdf_path} ===")

    # 1. Extraction OCR avancée
    full_text = ocr_from_pdf_enhanced(pdf_path)

    # 2. Champs de base (dentiste + patient + devis)
    basic_fields = extract_basic_fields(full_text)

    # 3. Extraction tables PDF (si PDF texte)
    tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    tables_data.append({
                        "page": i + 1,
                        "table": table
                    })
    except Exception as e:
        print(f"Erreur tables PDF : {e}")

    # 4. Totaux financiers
    totals = extract_totals_from_tables(tables_data, full_text)

    # 5. Consentement et diverses informations
    misc = extract_consent_and_misc(full_text)

    return {
        "texte_complet_ocr": full_text,
        "identification": basic_fields,
        "tables": tables_data,
        "totaux": totals,
        "informations_complementaires": misc
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main2.py fichier.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Erreur : fichier introuvable : {pdf_path}")
        sys.exit(1)

    result = analyse_devis(pdf_path)

    # Output JSON propre
    print("\n=== RÉSULTAT JSON ===")
    print(json.dumps(result, indent=4, ensure_ascii=False))
