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

        # Redimensionnement pour améliorer OCR
        scale_factor = 2
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Morphologie pour renforcer petites lettres
        kernel = np.ones((2,2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        return img
    except Exception as e:
        return None

def run_paddleocr(img_path: str) -> str:
    """Exécuter PaddleOCR"""
    try:
        result = ocr_engine.predict(img_path)
        if not result:
            return ""
        data = result[0]
        text_parts = []
        if 'rec_texts' in data:
            text_parts = data['rec_texts']
        else:
            for page in result:
                for line in page:
                    if len(line) >= 2:
                        text_info = line[1]
                        if text_info and len(text_info) >= 2:
                            text = str(text_info[0]).strip()
                            confidence = float(text_info[1])
                            if confidence > 0.3 and text:
                                text_parts.append(text)
        return " ".join(text_parts).strip()
    except Exception as e:
        return ""

def run_easyocr(img_path: str) -> str:
    """Exécuter EasyOCR"""
    try:
        results = easyocr_reader.readtext(img_path)
        texts = [text.strip() for (bbox, text, conf) in results if conf > 0.3 and text.strip()]
        return " ".join(texts)
    except:
        return ""

def ocr_from_pdf(pdf_path: str) -> str:
    """OCR hybride sur PDF scanné multipages"""
    doc = fitz.open(pdf_path)
    all_text = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
        img = np.frombuffer(pix.tobytes("png"), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            continue
        processed = preprocess_image(img)
        if processed is None:
            continue
        temp_path = f"temp_page_{i+1}.png"
        cv2.imwrite(temp_path, processed)
        text_paddle = run_paddleocr(temp_path)
        if not text_paddle.strip():
            text_paddle = run_easyocr(temp_path)
        if text_paddle.strip():
            all_text.append(f"Page {i+1}:\n{text_paddle}")
    return "\n\n".join(all_text)

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
        raise RuntimeError(f"Erreur ouverture PDF: {e}")

    # OCR hybride si PDF scanné ou texte trop court
    ocr_text = ""
    if len(full_text_pdf.strip()) < 500:
        ocr_text = ocr_from_pdf(pdf_path)
    full_text = full_text_pdf + "\n" + ocr_text if ocr_text else full_text_pdf
    return full_text, tables_data

# -------------------------
# Parsing & traitement
# -------------------------
# ... ici tu peux copier toutes tes fonctions existantes :
# extract_basic_fields, extract_treatments_from_tables,
# extract_totals_from_tables, extract_consent_and_misc,
# best_guess_field_from_row, parse_table_rows_with_header, etc.
# Elles restent inchangées, car elles fonctionnent pour PDF texte et OCR maintenant
# -------------------------





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
    # ---- Dentiste ----
    nom_praticien = find_first_match([
        r"Nom\s*Pr[ée]nom\s*[:\-]?\s*([^\n\r]+)",
        r"Nom\s+Prénom\s*[:\-]?\s*([^\n\r]+)",
        r"Nom\s*[:\-]?\s*([A-Z][^\n\r]+)\nIdentifiant"
    ], full_text, flags=re.IGNORECASE | re.MULTILINE)
    if nom_praticien:
        nom_praticien = re.split(r'Nom et prénom|Date de naissance', nom_praticien, flags=re.IGNORECASE)[0].strip()

    rpps = find_first_match([
        r"RPPS\s*[:\-]?\s*([0-9]{6,})",
        r"identifiant du praticien rpps\s*[:\-]?\s*([0-9]{6,})"
    ], full_text, flags=re.IGNORECASE)
    adeli = find_first_match([
        r"ADELI\s*[:\-]?\s*([0-9]{6,})",
        r"adeli\s*[:\-]?\s*([0-9]{6,})"
    ], full_text, flags=re.IGNORECASE)

    etablissement_full = find_first_match([
        r"Raison sociale et adresse\s*[:\-]?\s*([^\n\r]+)", 
        r"raison sociale et adresse\s*[:\-]?\s*([^\n\r]+)"
    ], full_text, flags=re.IGNORECASE)

    finess_dentiste = None
    numero_etablissement_dentiste = None
    adresse_dentiste = None
    if etablissement_full:
        m = re.search(r'N[°º] de l[’\']etablissement \(FINESS\)\s*[:\-]?\s*(\d+)', etablissement_full)
        if m:
            finess_dentiste = m.group(1)
            numero_etablissement_dentiste = finess_dentiste
            adresse_dentiste = re.sub(r'-?\s*N[°º] de l[’\']etablissement \(FINESS\)\s*[:\-]?\s*\d+', '', etablissement_full).strip()
        else:
            adresse_dentiste = etablissement_full.strip()

    # ---- Patient ----
    patient_nom = find_first_match([
        r"Nom et prénom\s*[:\-]?\s*([^\n\r]+)",
        r"Nom et prénom\s*:\s*([A-Z][^\n\r]+)"
    ], full_text, flags=re.IGNORECASE)
    if patient_nom:
        patient_nom = re.split(r'Date de naissance', patient_nom, flags=re.IGNORECASE)[0].strip()
    else:
        pm = re.search(r"^([A-Z][A-Z \-]+)\s+Date de naissance", full_text, flags=re.MULTILINE)
        if pm:
            patient_nom = pm.group(1).strip()

    date_naissance = find_first_match([
        r"Date de naissance\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Date de naissance\s*[:\-]?\s*(\d{2}\-\d{2}\-\d{4})"
    ], full_text, flags=re.IGNORECASE)

    num_secu = find_first_match([
        r"S[eé]curit[eé]\s*sociale[^\n\r]*[:\-]?\s*([0-9]{10,})",
        r"n[°º] de S[eé]curit[eé]\s*sociale\s*[:\-]?\s*([0-9]{10,})"
    ], full_text, flags=re.IGNORECASE)

    adresse_patient_raw = find_first_match([
        r"Adresse du patient\s*[:\-]?\s*([^\n\r]+(?:\n[^\n\r]+)?)",
        r"Adresse\s*[:\-]?\s*([0-9A-Za-z ].+)\n\s*\d{5}"
    ], full_text, flags=re.IGNORECASE)

    numero_etablissement_patient = None
    adresse_patient = None
    if adresse_patient_raw:
        m = re.search(r'N[°º] de l[’\']etablissement \(FINESS\)\s*[:\-]?\s*(\d+)', adresse_patient_raw)
        if m:
            numero_etablissement_patient = m.group(1)
        adresse_patient = re.sub(r'N[°º] de l[’\']etablissement \(FINESS\)\s*[:\-]?\s*\d+', '', adresse_patient_raw)
        adresse_patient = re.sub(r'\s+', ' ', adresse_patient).strip()

    # ---- Devis ----
    numero_devis = find_first_match([
        r"Num[ée]ro\s+du\s+devis\s*[:\-]?\s*([0-9A-Za-z\-]+)",
        r"Num[ée]ro du devis\s*[:\-]?\s*([0-9A-Za-z\-]+)"
    ], full_text, flags=re.IGNORECASE)

    date_devis = find_first_match([
        r"Date\s+du\s+devis\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Date du devis\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})"
    ], full_text, flags=re.IGNORECASE)

    validite = find_first_match([
        r"Valable jusqu['’`]\s*(?:au)?\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        r"Valable jusqu'au\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})"
    ], full_text, flags=re.IGNORECASE)

    if numero_etablissement_dentiste is None and adresse_patient:
        match = re.search(r'FINESS\)\s*:\s*(\d+)', adresse_patient)
        if match:
            numero_etablissement_dentiste = match.group(1)
        adresse_patient = re.sub(r"N° de l'établissement \(FINESS\) : \d+", "", adresse_patient)
           
    return {
        "dentiste": {
            "nom_praticien": nom_praticien,
            "rpps": rpps,
            "adeli": adeli,
            "finess": finess_dentiste,
            "adresse": adresse_dentiste,
            "numero_etablissement": numero_etablissement_dentiste
        },
        "patient": {
            "nom": patient_nom,
            "date_naissance": date_naissance,
            "numero_securite_sociale": num_secu,
            "adresse": adresse_patient,
            "numero_etablissement": numero_etablissement_patient
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
                if totals["honoraires_total"] is not None and totals["prix_dispositif_medical"] is not None:
                    return totals

    m = re.search(r'TOTAL\s*€[^\n\r]*\n?([^\n\r]+)', full_text, flags=re.IGNORECASE)
    if m:
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
    consent_text_present = bool(re.search(r'consentement\s+eclair', full_text, re.IGNORECASE)) or bool(re.search(r'consentement éclair', full_text, re.IGNORECASE))
    acompte_required = bool(re.search(r'acompte', full_text, re.IGNORECASE))
    date_consent = None
    m = re.search(r'\b[A-ZÉÈÎÏ][A-ZÉÈÎÏa-zéèîï\-\s]+,\s*le\s*(\d{2}/\d{2}/\d{4})', full_text)
    if m:
        date_consent = m.group(1)
    phone = find_first_match([r'(\+?\d{1,3}[ \u00A0]?\d{1,3}[ \u00A0]?\d{2,3}[ \u00A0]?\d{2,3}[ \u00A0]?\d{2,3})', r'(\d{2,3}[ \u00A0]?\d{2}[ \u00A0]?\d{2}[ \u00A0]?\d{2})'], full_text)
    signature_patient = bool(re.search(r'signature du patient', full_text, re.IGNORECASE)) or bool(re.search(r'Lu et approuvé', full_text, re.IGNORECASE))
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
    full_text, tables_data = extract_pdf_text_and_tables(pdf_path)
    full_text_norm = full_text.replace('\xa0', ' ')
    basic = extract_basic_fields(full_text_norm)
    treatments = extract_treatments_from_tables(tables_data, full_text_norm)
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
    return clean_result(result)

# -------------------------
# CLI / Exécution
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main2.py <chemin_vers_pdf>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        parsed = extract_dental_quote(path)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Erreur extraction PDF: {e}", file=sys.stderr)
        sys.exit(2)