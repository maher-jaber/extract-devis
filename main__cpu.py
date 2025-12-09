#!/usr/bin/env python3
# main2.py
# Extraction robuste de devis dentaires (PDF texte + PDF scannés)
# Usage: python main2.py <chemin_vers_pdf>

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

# -------------------------
# Initialisation OCR
# -------------------------
ocr_engine = PaddleOCR(lang='fr', use_angle_cls=True)
easyocr_reader = easyocr.Reader(['fr', 'en'])

# -------------------------
# Utilitaires généraux
# -------------------------
def safe_strip(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def norm_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def to_float_amount(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).replace('\u00A0', '').replace(' ', '').replace(',', '.')
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return None

def find_first_match(patterns: List[str], text: str, flags=0) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            g = m.group(1).strip()
            return norm_whitespace(g)
    return None

def find_all_matches(pattern: str, text: str, flags=0) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(pattern, text, flags)]

# -------------------------
# OCR sur image
# -------------------------
def preprocess_image(image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
    try:
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Vérifier et réduire la taille si nécessaire
        height, width = img.shape
        max_dimension = 2000  # Réduit pour éviter les problèmes
        
        if height > max_dimension or width > max_dimension:
            scale = min(max_dimension / height, max_dimension / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"  Image redimensionnée: {height}x{width} -> {new_height}x{new_width}")
        
        # Égalisation d'histogramme pour meilleur contraste
        img = cv2.equalizeHist(img)
        
        # Redimensionnement modéré
        scale_factor = 1.5  # Réduit
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Seuillage adaptatif pour documents scannés
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

        # Morphologie pour renforcer le texte
        kernel = np.ones((2,2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        return img
    except Exception as e:
        print(f"Erreur prétraitement image: {e}")
        return None

def run_paddleocr(img_path: str) -> str:
    """Exécuter PaddleOCR"""
    try:
        result = ocr_engine.predict(img_path)
        if not result:
            return ""
        
        text_parts = []
        for page in result:
            for line in page:
                if len(line) >= 2:
                    text_info = line[1]
                    if text_info and len(text_info) >= 2:
                        text = str(text_info[0]).strip()
                        confidence = float(text_info[1])
                        if confidence > 0.2 and text:
                            text_parts.append(text)
        return " ".join(text_parts).strip()
    except Exception as e:
        print(f"Erreur PaddleOCR: {e}")
        return ""

def run_easyocr(img_path: str) -> str:
    """Exécuter EasyOCR"""
    try:
        results = easyocr_reader.readtext(img_path, paragraph=False, width_ths=0.7)
        texts = [text.strip() for (bbox, text, conf) in results if conf > 0.2 and text.strip()]
        return " ".join(texts)
    except Exception as e:
        print(f"Erreur EasyOCR: {e}")
        return ""
    
def extract_treatments_from_ocr_text(full_text: str) -> List[Dict[str, Any]]:
    """Extrait les traitements directement depuis le texte OCR"""
    treatments = []
    
    # Pattern pour détecter les lignes de traitement
    # Format attendu: "14 HBLD350 Pose d'une couronne ... 440,00 120,00 72,00 368,00"
    treatment_pattern = r'''
        (?:\b(\d{1,2})\b\s+)?              # Numéro de dent optionnel
        (HBLD\d+|HBLD\s*\d+)\s+            # Code acte
        ([^\n]{20,100}?)                    # Description
        \s+                                 # Séparateur
        ([\d\s.,]+\s+[\d\s.,]+\s+[\d\s.,]+\s+[\d\s.,]+)  # 4 montants
    '''
    
    matches = re.finditer(treatment_pattern, full_text, re.IGNORECASE | re.VERBOSE | re.DOTALL)
    
    for match in matches:
        dent = match.group(1)
        code_acte = match.group(2).replace(' ', '') if match.group(2) else None
        description = norm_whitespace(match.group(3)) if match.group(3) else None
        montants = match.group(4)
        
        # Extraire les montants
        amounts = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2}))', montants)
        
        honoraires = to_float_amount(amounts[0]) if len(amounts) > 0 else None
        prix_dispositif = to_float_amount(amounts[1]) if len(amounts) > 1 else None
        base_remb = to_float_amount(amounts[2]) if len(amounts) > 2 else None
        reste_charge = to_float_amount(amounts[3]) if len(amounts) > 3 else None
        
        treatments.append({
            "code_acte": code_acte,
            "dent": dent,
            "description": description,
            "honoraires": honoraires,
            "prix_dispositif_medical": prix_dispositif,
            "base_remboursement": base_remb,
            "reste_a_charge": reste_charge
        })
    
    return treatments

def ocr_from_pdf(pdf_path: str) -> str:
    """OCR hybride sur PDF scanné multipages"""
    doc = fitz.open(pdf_path)
    all_text = []
    
    print(f"Traitement OCR du document scanné ({len(doc)} pages)...")
    
    for i, page in enumerate(doc):
        # Essayer plusieurs résolutions pour optimiser la détection des tables
        for zoom in [0.4, 0.6]:  # Essayer différentes tailles
            try:
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)
                
                temp_path = f"temp_page_{i+1}_zoom{zoom}.png"
                pix.save(temp_path)
                
                # Essayer EasyOCR qui est souvent meilleur pour les tables
                text_easy = run_easyocr(temp_path)
                
                if text_easy.strip():
                    # Ajouter des séparateurs pour indiquer les lignes de table
                    lines = text_easy.split('\n')
                    formatted_lines = []
                    for line in lines:
                        if re.search(r'\d+[\.,]\d{2}', line):  # Ligne avec montants
                            formatted_lines.append("| " + line + " |")
                        elif re.search(r'HBLD|Nature|Cotation', line, re.IGNORECASE):  # En-tête
                            formatted_lines.append("=== " + line + " ===")
                        else:
                            formatted_lines.append(line)
                    
                    page_text = "\n".join(formatted_lines)
                    all_text.append(f"=== PAGE {i+1} (zoom {zoom}) ===\n{page_text}")
                    print(f"  Page {i+1} (zoom {zoom}): {len(page_text)} caractères")
                    break
                    
            except Exception as e:
                print(f"  Erreur page {i+1} zoom {zoom}: {str(e)[:100]}...")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    doc.close()
    
    full_ocr_text = "\n\n".join(all_text)
    print(f"OCR terminé: {len(full_ocr_text)} caractères extraits au total")
    return full_ocr_text

# -------------------------
# Extraction texte PDF et tables
# -------------------------
def extract_pdf_text_and_tables(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    full_text_pdf = ""
    tables_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                full_text_pdf += text + "\n\n"
                page_tables = page.extract_tables()
                for t in page_tables:
                    if any(any(cell for cell in row if cell) for row in t):
                        tables_data.append({"page": i+1, "table": t})
    except Exception as e:
        print(f"Erreur pdfplumber: {e}")
        full_text_pdf = ""

    # Détection simple des PDF scannés
    # Vérifier s'il y a assez de texte structuré
    is_scanned = len(full_text_pdf.strip()) < 200
    
    # Vérifier également les mots clés
    if not is_scanned:
        keywords = ['DEVIS', 'dentaire', 'chirurgien', 'patient', 'honoraires']
        found_keywords = sum(1 for kw in keywords if kw.lower() in full_text_pdf.lower())
        if found_keywords < 2:
            is_scanned = True
    
    # OCR hybride si PDF scanné
    ocr_text = ""
    if is_scanned:
        print("Document détecté comme scanné - Application OCR...")
        ocr_text = ocr_from_pdf(pdf_path)
    else:
        print("Document détecté comme texte - Extraction directe...")
    
    # Fusion du texte
    if ocr_text:
        full_text = ocr_text
    else:
        full_text = full_text_pdf
    
    # Nettoyage final du texte
    full_text = full_text.replace('\xa0', ' ').replace('\r', '\n')
    
    print(f"Texte extrait: {len(full_text)} caractères")
    
    return full_text, tables_data

# -------------------------
# Normalisation & mapping colonnes
# -------------------------
def normalize_header_cells(header_row: List[Any]) -> List[str]:
    normalized = []
    for cell in header_row:
        if cell is None:
            normalized.append('')
            continue
        text = norm_whitespace(str(cell)).lower()
        text = text.replace('é', 'e').replace('è', 'e').replace('à', 'a').replace('ù', 'u').replace('ç', 'c')
        text = re.sub(r'[^a-z0-9 _-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        normalized.append(text)
    return normalized

def find_header_index_and_map(table: List[List[Any]]) -> Tuple[Optional[int], Optional[List[str]]]:
    header_keywords = ['cotation', 'nature', 'panier', 'honoraires', 'montant', 'base de remboursement', 'n° de traitement', 'n° dent', 'n° dent ou', 'numero', 'code']
    for i, row in enumerate(table):
        if not row:
            continue
        joined = ' '.join([str(c).lower() if c else '' for c in row])
        matches = sum(1 for kw in header_keywords if kw in joined)
        if matches >= 2:
            headers = normalize_header_cells(row)
            return i, headers
    if table and len(table[0]) >= 3:
        headers = normalize_header_cells(table[0])
        return 0, headers
    return None, None

# -------------------------
# Parsing traitements
# -------------------------
def parse_table_rows_with_header(table: List[List[Any]], header_index: int, headers: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for r in table[header_index + 1:]:
        if not any(c for c in r if c):
            continue
        row_map = {}
        for idx, cell in enumerate(r):
            key = headers[idx] if idx < len(headers) else f"col_{idx}"
            row_map[key] = safe_strip(str(cell)) if cell is not None else None
        rows.append(row_map)
    return rows

def best_guess_field_from_row(row_map: Dict[str, Any]) -> Dict[str, Any]:
    joined_vals = ' '.join([v for v in (row_map.values() if isinstance(row_map, dict) else []) if v])
    code = None
    code_match = re.search(r'(HBLD\d+|HBLD\s*\d+)', joined_vals, re.IGNORECASE)
    if code_match:
        code = code_match.group(1).replace(' ', '')

    dent = None
    for k, v in row_map.items():
        if v:
            if re.search(r'\b(n[°º]|numero|n°|nº|n)\b', k) or re.search(r'^\s*\d+\s*$', v):
                if re.match(r'^\s*\d{1,2}\s*$', v):
                    dent = v.strip()
                    break
            if 'dent' in k and v:
                dent = v.strip()
                break

    description = None
    for k, v in row_map.items():
        if v and re.search(r'couronn|pose|prothes|implant|inlay|infrastructure|acte', v, re.IGNORECASE):
            description = norm_whitespace(v)
            break

    panier = None
    for k, v in row_map.items():
        if v and 'panier' in k:
            panier = v.strip()
            break

    amounts = []
    for v in row_map.values():
        if not v:
            continue
        found = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2}))', v)
        for f in found:
            amounts.append(f.replace('\u00A0', '').replace(' ', ''))

    honoraires = to_float_amount(amounts[0]) if len(amounts) > 0 else None
    prix_dispositif = to_float_amount(amounts[1]) if len(amounts) > 1 else None
    base_remb = to_float_amount(amounts[2]) if len(amounts) > 2 else None
    reste_a_charge = to_float_amount(amounts[3]) if len(amounts) > 3 else None

    return {
        "code_acte": code,
        "dent": dent,
        "description": description,
        "panier": panier,
        "honoraires": honoraires,
        "prix_dispositif_medical": prix_dispositif,
        "base_remboursement": base_remb,
        "reste_a_charge": reste_a_charge
    }

def extract_treatments_from_tables(tables_data: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    treatments = []
    
    # 1. Essayer d'abord avec les tables structurées
    for tinfo in tables_data:
        table = tinfo['table']
        header_index, headers = find_header_index_and_map(table)
        if header_index is None or not headers:
            for row in table:
                row_text = ' '.join([str(c) for c in row if c])
                if 'HBLD' in row_text.upper() or re.search(r'HBLD\s*\d+', row_text, re.IGNORECASE):
                    row_map = {f"col_{i}": safe_strip(cell) for i, cell in enumerate(row)}
                    treatments.append(best_guess_field_from_row(row_map))
        else:
            parsed_rows = parse_table_rows_with_header(table, header_index, headers)
            for rm in parsed_rows:
                joined = ' '.join([v for v in rm.values() if v])
                if 'HBLD' in joined.upper() or re.search(r'HBLD\s*\d+', joined, re.IGNORECASE) or re.search(r'couronn|prothes|implant', joined, re.IGNORECASE):
                    treatments.append(best_guess_field_from_row(rm))
    
    # 2. Si aucune table trouvée, extraire directement depuis le texte OCR
    if not treatments:
        print("Aucune table détectée, tentative d'extraction depuis le texte brut...")
        
        # Chercher des lignes contenant HBLD codes
        hbld_pattern = r'(?:HBLD\d+|HBLD\s*\d+)\s+([^\n]+)'
        matches = re.findall(hbld_pattern, full_text, re.IGNORECASE)
        
        for match in matches:
            line = norm_whitespace(match)
            
            # Extraire le numéro de dent
            dent_match = re.search(r'\b(\d{1,2})\b', line[:10])
            dent = dent_match.group(1) if dent_match else None
            
            # Extraire les montants
            amounts = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2}))', line)
            honoraires = to_float_amount(amounts[0]) if len(amounts) > 0 else None
            base_remb = to_float_amount(amounts[1]) if len(amounts) > 1 else None
            reste_charge = to_float_amount(amounts[2]) if len(amounts) > 2 else None
            
            # Identifier le code acte
            code_match = re.search(r'(HBLD\d+|HBLD\s*\d+)', line, re.IGNORECASE)
            code = code_match.group(1).replace(' ', '') if code_match else None
            
            treatments.append({
                "code_acte": code,
                "dent": dent,
                "description": line[:100],  # Prendre les premiers 100 caractères comme description
                "honoraires": honoraires,
                "base_remboursement": base_remb,
                "reste_a_charge": reste_charge
            })
    
    # 3. Déduplication
    seen = set()
    dedup = []
    for t in treatments:
        key = (str(t.get('dent')), str(t.get('code_acte')))
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    
    return dedup
# -------------------------
# Extraction données dentiste / patient / devis
# -------------------------
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
    consent_text_present = bool(re.search(r'consentement[^:\n]*éclair', full_text, re.IGNORECASE))
    acompte_required = bool(re.search(r'acompte|avance|versement[^:\n]*initial', full_text, re.IGNORECASE))
    
    date_consent = None
    for pattern in [r'Fait[^,\n]*,\s*le\s*(\d{2}/\d{2}/\d{4})', r'Date[^:\n]*[:\-]\s*(\d{2}/\d{2}/\d{4})']:
        m = re.search(pattern, full_text)
        if m:
            date_consent = m.group(1)
            break
    
    phone = find_first_match([
        r'Téléphone[^:\n]*[:\-]\s*([+\d\s\-]+)',
        r'Tel[^:\n]*[:\-]\s*([+\d\s\-]+)',
        r'(\d{2}[ \.]\d{2}[ \.]\d{2}[ \.]\d{2})'
    ], full_text)
    
    signature_patient = bool(re.search(r'signature[^:\n]*patient|Lu et approuvé', full_text, re.IGNORECASE))
    
    return {
        "consentement_eclaire_present": consent_text_present,
        "date_consentement": date_consent,
        "acompte_requis": acompte_required,
        "telephone": phone,
        "signature_patient": signature_patient
    }

# -------------------------
# Nettoyage final
# -------------------------
def clean_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cleaned = clean_result(v)
            if cleaned is None and k not in ("traitements", "financier"):
                continue
            if isinstance(cleaned, str) and cleaned == "" and k not in ("source_file",):
                continue
            out[k] = cleaned
        return out if out else None
    elif isinstance(obj, list):
        nl = [clean_result(x) for x in obj]
        nl = [x for x in nl if x is not None and x != ""]
        return nl if nl else None
    elif isinstance(obj, str):
        s = obj.strip()
        return s if s else None
    else:
        return obj

# -------------------------
# Fonction principale
# -------------------------
def extract_dental_quote(pdf_path: str) -> Dict[str, Any]:
    print(f"=== Extraction du devis: {pdf_path} ===")
    
    try:
        # Extraction texte et tables
        full_text, tables_data = extract_pdf_text_and_tables(pdf_path)
        full_text_norm = full_text.replace('\xa0', ' ')
        
        # Extraction des données
        basic = extract_basic_fields(full_text_norm)
        
        # Essayer d'abord l'extraction par tables
        treatments = extract_treatments_from_tables(tables_data, full_text_norm)
        
        # Si échec, essayer l'extraction directe depuis le texte OCR
        if not treatments:
            print("Tentative d'extraction alternative depuis le texte OCR...")
            treatments = extract_treatments_from_ocr_text(full_text_norm)
        
        # Si toujours échec, essayer une dernière méthode
        if not treatments:
            print("Tentative d'extraction de dernière chance...")
            # Chercher simplement toutes les lignes avec HBLD
            lines = full_text_norm.split('\n')
            for line in lines:
                if 'HBLD' in line.upper():
                    # Extraction simple
                    code_match = re.search(r'(HBLD\d+)', line, re.IGNORECASE)
                    amounts = re.findall(r'(\d+[\.,]\d{2})', line)
                    if code_match and amounts:
                        treatments.append({
                            "code_acte": code_match.group(1),
                            "description": line[:80],
                            "honoraires": to_float_amount(amounts[0]) if amounts else None,
                            "reste_a_charge": to_float_amount(amounts[-1]) if amounts else None
                        })
        
        totals = extract_totals_from_tables(tables_data, full_text_norm)
        misc = extract_consent_and_misc(full_text_norm)

        result = {
            "source_file": pdf_path,
            "dentiste": basic.get("dentiste"),
            "patient": basic.get("patient"),
            "devis": basic.get("devis"),
            "traitements": treatments,
            "financier": totals,
            "informations_complementaires": misc
        }
        
        print(f"=== Extraction terminée ===")
        print(f"Traitements trouvés: {len(treatments)}")
        return clean_result(result)
    
    except Exception as e:
        print(f"ERREUR lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "source_file": pdf_path,
            "error": str(e)
        }
        
def debug_extraction(pdf_path: str):
    """Fonction de débogage pour voir ce qui est extrait"""
    print(f"\n=== DÉBOGAGE: {pdf_path} ===")
    
    full_text, tables_data = extract_pdf_text_and_tables(pdf_path)
    
    print(f"\n1. Longueur du texte: {len(full_text)} caractères")
    print(f"2. Nombre de tables détectées: {len(tables_data)}")
    
    print("\n3. Aperçu du texte (premiers 2000 caractères):")
    print(full_text[:2000])
    
    print("\n4. Recherche de HBLD dans le texte:")
    hbld_matches = re.findall(r'HBLD\d+', full_text, re.IGNORECASE)
    print(f"   Trouvé: {hbld_matches}")
    
    print("\n5. Recherche de montants:")
    amounts = re.findall(r'\d+[\.,]\d{2}', full_text)
    print(f"   Trouvé: {amounts[:10]}...")
# -------------------------
# CLI / Exécution
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main2.py <chemin_vers_pdf>")
        sys.exit(1)
    path = sys.argv[1]
    
    # Vérifier que le fichier existe
    if not os.path.exists(path):
        print(f"Erreur: Le fichier '{path}' n'existe pas.", file=sys.stderr)
        sys.exit(2)
    
    try:
        parsed = extract_dental_quote(path)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Erreur extraction PDF: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)