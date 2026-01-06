#!/usr/bin/env python3
# main_ultra.py - Version Ultra-Performante pour PDFs dentaires
# Extraction profonde avec analyse multi-techniques

import cv2
import re
import json
import sys
import pandas as pd
import numpy as np
import pdfplumber
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import easyocr
import os
import pytesseract
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
import traceback
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# INITIALISATION DES MOTEURS OCR
# -------------------------
class OCRManager:
    def __init__(self):
        self.engines = {}
        self.init_all_engines()
    
    def init_all_engines(self):
        """Initialise tous les moteurs OCR disponibles"""
        
        # Tesseract
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.name == "nt"
                else "/usr/bin/tesseract"
            )
            self.engines['tesseract'] = True
            logger.info("✓ Tesseract prêt")
        except Exception as e:
            logger.warning(f"✗ Tesseract: {e}")
            self.engines['tesseract'] = False
        
        # PaddleOCR
        try:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='fr',
                use_gpu=True,
                show_log=False
            )
            self.engines['paddle'] = True
            logger.info("✓ PaddleOCR initialisé")
        except Exception as e:
            logger.warning(f"✗ PaddleOCR: {e}")
            self.paddle_ocr = None
            self.engines['paddle'] = False
        
        # EasyOCR
        try:
            self.easy_reader = easyocr.Reader(['fr'], gpu=True)
            self.engines['easyocr'] = True
            logger.info("✓ EasyOCR initialisé")
        except Exception as e:
            logger.warning(f"✗ EasyOCR: {e}")
            self.easy_reader = None
            self.engines['easyocr'] = False
    
    def get_available_engines(self):
        return [name for name, available in self.engines.items() if available]

ocr_manager = OCRManager()

# -------------------------
# UTILITAIRES AVANCÉS
# -------------------------
class TextProcessor:
    """Classe pour le traitement avancé du texte"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalise le texte pour l'analyse"""
        if not text:
            return ""
        
        # Remplacements multi-espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Corrections OCR courantes
        corrections = {
            'HBLD030': 'HBLD350',
            'HBLD090': 'HBLD090',
            'HBLD038': 'HBLD038',
            'H B L D': 'HBLD',
            'HBL D': 'HBLD',
            'H BLD': 'HBLD',
            'couronn e': 'couronne',
            'dentair e': 'dentaire',
            'prothè se': 'prothèse',
            'zircon e': 'zircone',
            '€uro': 'euro',
            '€': '€',
            'l o': "l'",
            'd une': "d'une",
            'qu une': "qu'une",
            'Clairbaux': 'Clairbaux',
            'Clairbalix': 'Clairbaux',
            'Stratica': 'Stratica',
            'Emilie': 'Emile',
            'CLARBALIX': 'CLARBAUX',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text.strip()
    
    @staticmethod
    def extract_amounts(text: str) -> List[float]:
        """Extrait tous les montants d'un texte"""
        if not text:
            return []
        
        # Patterns pour les montants
        patterns = [
            r'\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})',
            r'\d+[.,]\d{2}',
            r'\b\d{2,}\b'
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Nettoyer le montant
                    clean = str(match).replace('\u00A0', '').replace(' ', '').replace(',', '.')
                    clean = re.sub(r'[^\d\.\-]', '', clean)
                    if clean and '.' in clean:
                        amount = float(clean)
                        if amount > 0:
                            amounts.append(amount)
                except:
                    continue
        
        return sorted(set(amounts))  # Retourne les montants uniques triés
    
    @staticmethod
    def find_pattern(text: str, pattern: str, context_chars: int = 100) -> List[Dict]:
        """Trouve un pattern avec son contexte"""
        results = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            context = text[start:end]
            
            results.append({
                'match': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'context': context,
                'amounts': TextProcessor.extract_amounts(context)
            })
        
        return results
    
    @staticmethod
    def extract_table_from_text(text: str) -> List[List[str]]:
        """Essaie d'extraire une table à partir du texte"""
        lines = text.split('\n')
        table = []
        
        for line in lines:
            # Chercher des séparateurs de colonnes
            if re.search(r'\s{3,}|\t|\|\s*', line):
                # Essayer de diviser la ligne en colonnes
                cols = re.split(r'\s{3,}|\t', line)
                cols = [col.strip() for col in cols if col.strip()]
                if len(cols) >= 2:  # Au moins 2 colonnes
                    table.append(cols)
        
        return table

# -------------------------
# PRÉ-TRAITEMENT D'IMAGE AVANCÉ
# -------------------------
class ImagePreprocessor:
    """Classe pour le pré-traitement avancé des images"""
    
    @staticmethod
    def enhance_image_for_ocr(image_path: str) -> Optional[np.ndarray]:
        """Améliore l'image pour l'OCR"""
        try:
            # Charger l'image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Conserver une copie originale
            original = img.copy()
            
            # Liste des techniques à essayer
            enhanced_images = []
            
            # Technique 1: CLAHE + Denoising
            try:
                # Conversion LAB pour améliorer le contraste
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # CLAHE sur le canal L
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                lab = cv2.merge((l, a, b))
                img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                enhanced_images.append(('clahe', img_clahe))
            except Exception as e:
                logger.debug(f"CLAHE échoué: {e}")
            
            # Technique 2: Seuillage adaptatif
            try:
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray, h=30)
                
                # Multiple seuillages
                binary1 = cv2.adaptiveThreshold(denoised, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
                enhanced_images.append(('binary1', binary1))
                
                binary2 = cv2.adaptiveThreshold(denoised, 255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 15, 3)
                enhanced_images.append(('binary2', binary2))
            except Exception as e:
                logger.debug(f"Seuillage échoué: {e}")
            
            # Technique 3: Amélioration des bords
            try:
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                enhanced_images.append(('edges', edges))
            except Exception as e:
                logger.debug(f"Détection bords échouée: {e}")
            
            return enhanced_images if enhanced_images else [('original', original)]
            
        except Exception as e:
            logger.error(f"Erreur pré-traitement image: {e}")
            return None
    
    @staticmethod
    def save_images(images: List, prefix: str = "debug"):
        """Sauvegarde les images pour debug"""
        for i, (name, img) in enumerate(images):
            filename = f"{prefix}_{name}_{i}.png"
            cv2.imwrite(filename, img)
            logger.debug(f"Image sauvegardée: {filename}")

# -------------------------
# OCR MULTI-MOTEUR
# -------------------------
class MultiEngineOCR:
    """Classe pour l'OCR multi-moteur"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def ocr_with_all_engines(self, image_path: str) -> Dict[str, str]:
        """Exécute l'OCR avec tous les moteurs disponibles"""
        results = {}
        
        # Pré-traiter l'image
        preprocessor = ImagePreprocessor()
        enhanced_images = preprocessor.enhance_image_for_ocr(image_path)
        
        if not enhanced_images:
            logger.warning(f"Aucune image améliorée pour {image_path}")
            return results
        
        # Tester avec chaque image améliorée
        for img_name, img in enhanced_images:
            temp_path = f"temp_{img_name}.png"
            cv2.imwrite(temp_path, img)
            
            # PaddleOCR
            if ocr_manager.engines.get('paddle') and ocr_manager.paddle_ocr:
                try:
                    result = ocr_manager.paddle_ocr.ocr(temp_path, cls=True)
                    if result and result[0]:
                        paddle_text = " ".join([line[1][0] for line in result[0] 
                                              if len(line) >= 2 and line[1][0]])
                        paddle_text = self.text_processor.normalize_text(paddle_text)
                        if paddle_text:
                            results[f'paddle_{img_name}'] = paddle_text
                            logger.debug(f"PaddleOCR ({img_name}): {len(paddle_text)} chars")
                except Exception as e:
                    logger.debug(f"PaddleOCR erreur ({img_name}): {e}")
            
            # EasyOCR
            if ocr_manager.engines.get('easyocr') and ocr_manager.easy_reader:
                try:
                    easy_results = ocr_manager.easy_reader.readtext(
                        temp_path,
                        paragraph=True,
                        width_ths=0.7,
                        height_ths=0.7,
                        min_size=10,
                        text_threshold=0.3
                    )
                    easy_text = " ".join([res[1] for res in easy_results if res[1]])
                    easy_text = self.text_processor.normalize_text(easy_text)
                    if easy_text:
                        results[f'easy_{img_name}'] = easy_text
                        logger.debug(f"EasyOCR ({img_name}): {len(easy_text)} chars")
                except Exception as e:
                    logger.debug(f"EasyOCR erreur ({img_name}): {e}")
            
            # Tesseract
            if ocr_manager.engines.get('tesseract'):
                try:
                    tesseract_text = pytesseract.image_to_string(
                        temp_path,
                        lang='fra+eng',
                        config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
                    )
                    tesseract_text = self.text_processor.normalize_text(tesseract_text)
                    if tesseract_text:
                        results[f'tess_{img_name}'] = tesseract_text
                        logger.debug(f"Tesseract ({img_name}): {len(tesseract_text)} chars")
                except Exception as e:
                    logger.debug(f"Tesseract erreur ({img_name}): {e}")
            
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Fusionner tous les résultats
        all_text = " ".join(results.values())
        results['combined'] = self.text_processor.normalize_text(all_text)
        
        return results
    
    def ocr_pdf_page(self, pdf_path: str, page_num: int, zoom_levels: List[float] = None) -> str:
        """OCR d'une page PDF avec différents niveaux de zoom"""
        if zoom_levels is None:
            zoom_levels = [1.0, 1.5, 2.0]
        
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        all_page_texts = []
        
        for zoom in zoom_levels:
            try:
                # Rendre la page avec le zoom
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Sauvegarder temporairement
                temp_img = f"temp_page_{page_num}_z{zoom}.png"
                pix.save(temp_img)
                
                # OCR avec tous les moteurs
                ocr_results = self.ocr_with_all_engines(temp_img)
                
                # Prendre le texte combiné
                if ocr_results.get('combined'):
                    all_page_texts.append(f"--- ZOOM {zoom}x ---\n{ocr_results['combined']}")
                
                # Nettoyer
                if os.path.exists(temp_img):
                    os.remove(temp_img)
                    
            except Exception as e:
                logger.debug(f"Page {page_num} zoom {zoom} erreur: {e}")
                continue
        
        doc.close()
        
        # Fusionner les meilleurs résultats
        if all_page_texts:
            # Prendre le texte le plus long (le plus complet)
            best_text = max(all_page_texts, key=len)
            return best_text
        
        return ""

# -------------------------
# ANALYSEUR PDF PROFOND
# -------------------------
class DeepPDFAnalyzer:
    """Classe pour l'analyse profonde des PDFs"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.ocr_engine = MultiEngineOCR()
        
    def analyze_pdf_structure(self, pdf_path: str) -> Dict:
        """Analyse la structure du PDF"""
        analysis = {
            'is_text_based': False,
            'has_tables': False,
            'page_count': 0,
            'text_density': 0,
            'table_structures': [],
            'detected_sections': []
        }
        
        try:
            # Essayer pdfplumber d'abord
            with pdfplumber.open(pdf_path) as pdf:
                analysis['page_count'] = len(pdf.pages)
                total_text = ""
                total_tables = 0
                
                for page_num, page in enumerate(pdf.pages):
                    # Extraire le texte
                    page_text = page.extract_text() or ""
                    total_text += page_text
                    
                    # Chercher des tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table and any(any(cell for cell in row if cell) for row in table):
                            total_tables += 1
                            analysis['table_structures'].append({
                                'page': page_num + 1,
                                'row_count': len(table),
                                'col_count': max(len(row) for row in table) if table else 0
                            })
                    
                    # Détecter les sections
                    sections = self._detect_sections(page_text)
                    analysis['detected_sections'].extend(sections)
                
                # Calculer la densité de texte
                if analysis['page_count'] > 0:
                    analysis['text_density'] = len(total_text) / analysis['page_count']
                
                # Déterminer si c'est un PDF texte
                analysis['is_text_based'] = (
                    analysis['text_density'] > 500 or  # Plus de 500 caractères par page
                    'DEVIS' in total_text or
                    'dentaire' in total_text.lower() or
                    'HBLD' in total_text
                )
                
                analysis['has_tables'] = total_tables > 0
                
                logger.info(f"Structure analysée: {analysis['page_count']} pages, "
                          f"text_based={analysis['is_text_based']}, "
                          f"tables={total_tables}, density={analysis['text_density']:.0f}")
                
                return analysis
                
        except Exception as e:
            logger.warning(f"Analyse structure PDF échouée: {e}")
            # Si pdfplumber échoue, ouvrir avec PyMuPDF pour compter les pages
            try:
                doc = fitz.open(pdf_path)
                analysis['page_count'] = len(doc)
                doc.close()
                analysis['is_text_based'] = False  # Supposer scanné si pdfplumber échoue
            except:
                pass
        
        return analysis
    
    def _detect_sections(self, text: str) -> List[str]:
        """Détecte les sections dans le texte"""
        sections = []
        patterns = [
            (r'Identification du (?:chirurgien-dentiste|praticien|patient)', 'Identification'),
            (r'Traitement proposé|Actes proposés', 'Traitements'),
            (r'DEVIS|Devis', 'Devis'),
            (r'Consentement éclairé', 'Consentement'),
            (r'TOTAL|Total|Montant total', 'Totaux'),
            (r'Remboursement|AMO|Sécurité sociale', 'Remboursements'),
        ]
        
        for pattern, section_name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sections.append(section_name)
        
        return list(set(sections))
    
    def extract_all_text(self, pdf_path: str, analysis: Dict) -> str:
        """Extrait tout le texte du PDF selon son type"""
        
        # Pour les PDF texte, utiliser pdfplumber
        if analysis.get('is_text_based', False):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    all_text = []
                    for page in pdf.pages:
                        text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                        if text:
                            all_text.append(self.text_processor.normalize_text(text))
                    
                    full_text = "\n\n".join(all_text)
                    logger.info(f"Texte extrait (pdfplumber): {len(full_text)} caractères")
                    return full_text
            except Exception as e:
                logger.warning(f"pdfplumber échoué, passage à OCR: {e}")
        
        # Pour les PDF scannés, utiliser OCR multi-moteur
        logger.info("Extraction OCR complète...")
        all_pages_text = []
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                logger.info(f"  OCR page {page_num + 1}/{len(doc)}...")
                page_text = self.ocr_engine.ocr_pdf_page(pdf_path, page_num)
                if page_text:
                    all_pages_text.append(f"\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n{page_text}")
            
            doc.close()
            
            full_text = "\n".join(all_pages_text)
            full_text = self.text_processor.normalize_text(full_text)
            
            logger.info(f"Texte OCR extrait: {len(full_text)} caractères")
            return full_text
            
        except Exception as e:
            logger.error(f"OCR échoué: {e}")
            return ""
    
    def extract_tables_deep(self, pdf_path: str, analysis: Dict) -> List[Dict]:
        """Extrait les tables en profondeur"""
        tables = []
        
        # Pour les PDF texte avec tables
        if analysis.get('has_tables', False) and analysis.get('is_text_based', False):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        for table_num, table in enumerate(page_tables):
                            if table and any(any(cell for cell in row if cell) for row in table):
                                # Nettoyer la table
                                cleaned_table = []
                                for row in table:
                                    cleaned_row = []
                                    for cell in row:
                                        if cell:
                                            cleaned_cell = self.text_processor.normalize_text(str(cell))
                                            cleaned_row.append(cleaned_cell)
                                        else:
                                            cleaned_row.append("")
                                    cleaned_table.append(cleaned_row)
                                
                                tables.append({
                                    'page': page_num + 1,
                                    'table_num': table_num,
                                    'table': cleaned_table,
                                    'source': 'pdfplumber',
                                    'row_count': len(cleaned_table),
                                    'col_count': max(len(row) for row in cleaned_table) if cleaned_table else 0
                                })
                                
                                logger.debug(f"Table extraite page {page_num+1}: {len(cleaned_table)}x{max(len(row) for row in cleaned_table) if cleaned_table else 0}")
            except Exception as e:
                logger.warning(f"Extraction tables pdfplumber échouée: {e}")
        
        return tables

# -------------------------
# EXTRACTEUR DE DONNÉES DENTAIRES
# -------------------------
class DentalDataExtractor:
    """Extracteur spécialisé pour les données dentaires"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def extract_basic_info(self, full_text: str) -> Dict:
        """Extrait les informations de base"""
        info = {
            'dentiste': {},
            'patient': {},
            'devis': {}
        }
        
        # Dentiste
        dentiste_patterns = [
            (r'Nom\s*Pr[ée]nom\s*[:\-]?\s*([^\n:]+?)(?=\s*(?:Identifiant|RPPS|ADELI|$))', 'nom_praticien'),
            (r'RPPS[^\d]*(\d{8,})', 'rpps'),
            (r'ADELI[^\d]*(\d{8,})', 'adeli'),
            (r'FINESS[^\d]*(\d{8,})', 'finess'),
            (r'Raison[^:\n]*[:\-]\s*([^\n]+?)(?=\s*(?:N[°º]|$))', 'adresse'),
        ]
        
        for pattern, field in dentiste_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip() if match.group(1) else match.group(0).strip()
                info['dentiste'][field] = self.text_processor.normalize_text(value)
        
        # Patient
        patient_patterns = [
            (r'Patient[^:\n]*[:\-]\s*([^\n]+?)(?=\s*(?:Date|Naissance|$))', 'nom'),
            (r'Nom[^:\n]*[:\-]\s*([^\n]+?)(?=\s*(?:Date|Naissance|$))', 'nom'),
            (r'Date[^:\n]*naissance[^:\n]*[:\-]?\s*(\d{2}/\d{2}/\d{4})', 'date_naissance'),
            (r'S[ée]curit[ée][^:\n]*[:\-]?\s*(\d{13,15})', 'numero_securite_sociale'),
            (r'Adresse[^:\n]*patient[^:\n]*[:\-]\s*([^\n]+)', 'adresse'),
        ]
        
        for pattern, field in patient_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip() if match.group(1) else match.group(0).strip()
                info['patient'][field] = self.text_processor.normalize_text(value)
        
        # Devis
        devis_patterns = [
            (r'Devis[^:\n]*[:\-]\s*([^\n]+?)(?=\s*(?:Date|$))', 'numero'),
            (r'R[ée]f[ée]rence[^:\n]*[:\-]\s*([^\n]+)', 'numero'),
            (r'Date[^:\n]*devis[^:\n]*[:\-]?\s*(\d{2}/\d{2}/\d{4})', 'date_devis'),
            (r'Valable[^\n]*(\d{2}/\d{2}/\d{4})', 'validite'),
        ]
        
        for pattern, field in devis_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip() if match.group(1) else match.group(0).strip()
                info['devis'][field] = self.text_processor.normalize_text(value)
        
        return info
    
    def extract_treatments_advanced(self, full_text: str, tables: List[Dict]) -> List[Dict]:
        """Extrait les traitements avec une approche avancée"""
        treatments = []
        
        # Stratégie 1: Analyser les tables extraites
        for table_info in tables:
            table = table_info.get('table', [])
            if not table or len(table) < 2:
                continue
            
            # Chercher une table de traitements (contient HBLD)
            table_text = "\n".join([" | ".join(row) for row in table])
            if 'HBLD' in table_text.upper():
                treatments_from_table = self._extract_from_table(table, table_info['page'])
                treatments.extend(treatments_from_table)
        
        # Stratégie 2: Analyser le texte ligne par ligne
        if len(treatments) < 3:  # Si pas assez de traitements trouvés
            treatments_from_text = self._extract_from_text_lines(full_text)
            treatments.extend(treatments_from_text)
        
        # Stratégie 3: Recherche par contexte HBLD
        if len(treatments) < 3:
            treatments_from_context = self._extract_by_hbld_context(full_text)
            treatments.extend(treatments_from_context)
        
        # Déduplication et nettoyage
        unique_treatments = self._deduplicate_treatments(treatments)
        
        logger.info(f"Traitements extraits: {len(unique_treatments)}")
        return unique_treatments
    
    def _extract_from_table(self, table: List[List[str]], page: int) -> List[Dict]:
        """Extrait les traitements d'une table"""
        treatments = []
        
        # Trouver les indices de colonnes
        header_row = None
        for i, row in enumerate(table):
            row_text = " ".join([str(cell) for cell in row if cell])
            if any(keyword in row_text.upper() for keyword in ['HBLD', 'COTATION', 'ACTE']):
                header_row = i
                break
        
        if header_row is None or header_row >= len(table):
            return treatments
        
        # Analyser les lignes de données
        for row_idx in range(header_row + 1, len(table)):
            row = table[row_idx]
            if not row or not any(cell for cell in row):
                continue
            
            # Chercher un code HBLD
            code = None
            for cell in row:
                if cell and re.search(r'HBLD\d{3,}', str(cell), re.IGNORECASE):
                    code_match = re.search(r'(HBLD\d{3,})', str(cell), re.IGNORECASE)
                    if code_match:
                        code = code_match.group(1).upper()
                        break
            
            if not code:
                continue
            
            # Extraire les informations
            treatment = {
                'code_acte': code,
                'page': page,
                'row': row_idx + 1,
                'source': 'table'
            }
            
            # Numéro de dent
            for cell in row:
                if cell and str(cell).isdigit():
                    dent = int(str(cell))
                    if 1 <= dent <= 48:
                        treatment['dent'] = str(dent)
                        break
            
            # Montants
            all_amounts = []
            for cell in row:
                if cell:
                    amounts = self.text_processor.extract_amounts(str(cell))
                    all_amounts.extend(amounts)
            
            # Assigner les montants
            if all_amounts:
                all_amounts = sorted(set(all_amounts), reverse=True)
                if len(all_amounts) >= 1:
                    treatment['honoraires'] = all_amounts[0]
                if len(all_amounts) >= 2:
                    treatment['reste_a_charge'] = all_amounts[-1]
                if len(all_amounts) >= 3:
                    treatment['prix_dispositif_medical'] = all_amounts[1]
            
            # Description (premier texte non numérique)
            description_parts = []
            for cell in row:
                if cell and not re.search(r'\d+[.,]\d{2}', str(cell)) and not str(cell).isdigit():
                    desc_part = str(cell).strip()
                    if desc_part and desc_part != code:
                        description_parts.append(desc_part)
            
            if description_parts:
                treatment['description'] = " ".join(description_parts[:3])
            
            treatments.append(treatment)
        
        return treatments
    
    def _extract_from_text_lines(self, text: str) -> List[Dict]:
        """Extrait les traitements des lignes de texte"""
        treatments = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Vérifier si la ligne contient un HBLD
            hbld_match = re.search(r'(HBLD\d{3,})', line, re.IGNORECASE)
            if hbld_match:
                code = hbld_match.group(1).upper()
                
                # Extraire les montants
                amounts = self.text_processor.extract_amounts(line)
                
                if amounts:
                    treatment = {
                        'code_acte': code,
                        'source': 'text_line',
                        'line': line_num + 1
                    }
                    
                    # Numéro de dent
                    dent_match = re.search(r'\b(1[0-9]|2[0-9]|3[0-9]|4[0-8]|[1-9])\b', line)
                    if dent_match:
                        treatment['dent'] = dent_match.group(1)
                    
                    # Assigner les montants
                    amounts = sorted(set(amounts), reverse=True)
                    if len(amounts) >= 1:
                        treatment['honoraires'] = amounts[0]
                    if len(amounts) >= 2:
                        treatment['reste_a_charge'] = amounts[-1]
                    
                    # Description
                    description = line[:200].strip()
                    treatment['description'] = description
                    
                    treatments.append(treatment)
        
        return treatments
    
    def _extract_by_hbld_context(self, text: str) -> List[Dict]:
        """Extrait les traitements par contexte HBLD"""
        treatments = []
        
        # Trouver tous les HBLD avec contexte
        hbld_matches = self.text_processor.find_pattern(text, r'HBLD\d{3,}', context_chars=200)
        
        for match_info in hbld_matches:
            code = match_info['match'].upper()
            context = match_info['context']
            amounts = match_info['amounts']
            
            if amounts:
                treatment = {
                    'code_acte': code,
                    'source': 'context',
                    'context': context
                }
                
                # Numéro de dent
                dent_match = re.search(r'\b(1[0-9]|2[0-9]|3[0-9]|4[0-8]|[1-9])\b', context)
                if dent_match:
                    treatment['dent'] = dent_match.group(1)
                
                # Assigner les montants
                amounts = sorted(set(amounts), reverse=True)
                if len(amounts) >= 1:
                    treatment['honoraires'] = amounts[0]
                if len(amounts) >= 2:
                    treatment['reste_a_charge'] = amounts[-1]
                
                treatments.append(treatment)
        
        return treatments
    
    def _deduplicate_treatments(self, treatments: List[Dict]) -> List[Dict]:
        """Déduplique les traitements"""
        unique_treatments = []
        seen_keys = set()
        
        for treatment in treatments:
            # Créer une clé unique
            key_parts = []
            if 'code_acte' in treatment:
                key_parts.append(treatment['code_acte'])
            if 'dent' in treatment:
                key_parts.append(str(treatment['dent']))
            if 'honoraires' in treatment:
                key_parts.append(f"{treatment['honoraires']:.2f}")
            
            key = "_".join(key_parts) if key_parts else str(treatment)
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_treatments.append(treatment)
        
        return unique_treatments
    
    def extract_totals(self, full_text: str, treatments: List[Dict]) -> Dict:
        """Extrait les totaux financiers"""
        totals = {
            'honoraires_total': None,
            'prix_dispositif_medical_total': None,
            'base_remboursement_total': None,
            'montant_rembourse_total': None,
            'reste_a_charge_total': None
        }
        
        # Chercher les totaux dans le texte
        total_patterns = [
            r'TOTAL[^\n€]*€[^\n]*([0-9\s.,]+)',
            r'Total[^\n€]*€[^\n]*([0-9\s.,]+)',
            r'Montant total[^\n€]*€[^\n]*([0-9\s.,]+)',
        ]
        
        for pattern in total_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    amounts = self.text_processor.extract_amounts(match.group(1))
                    if amounts:
                        # Assigner selon la position
                        if len(amounts) >= 1 and not totals['honoraires_total']:
                            totals['honoraires_total'] = amounts[0]
                        if len(amounts) >= 2 and not totals['reste_a_charge_total']:
                            totals['reste_a_charge_total'] = amounts[1]
        
        # Si pas trouvé, calculer à partir des traitements
        if not totals['honoraires_total'] and treatments:
            honoraires_sum = sum(t.get('honoraires', 0) or 0 for t in treatments)
            totals['honoraires_total'] = honoraires_sum
        
        if not totals['reste_a_charge_total'] and treatments:
            reste_sum = sum(t.get('reste_a_charge', 0) or 0 for t in treatments)
            totals['reste_a_charge_total'] = reste_sum
        
        return totals
    
    def extract_additional_info(self, full_text: str) -> Dict:
        """Extrait les informations complémentaires"""
        info = {
            'consentement_eclaire': False,
            'mutuelle': None,
            'acompte': None,
            'mentions_legales': False
        }
        
        # Consentement
        if re.search(r'consentement.*éclairé|éclairé.*consentement', full_text, re.IGNORECASE):
            info['consentement_eclaire'] = True
        
        # Mutuelle
        mutuelle_match = re.search(r'Mutuelle[^:\n]*[:\-]?\s*([^\n]+)', full_text, re.IGNORECASE)
        if mutuelle_match:
            info['mutuelle'] = self.text_processor.normalize_text(mutuelle_match.group(1))
        
        # Acompte
        acompte_match = re.search(r'Acompte[^:\n]*([0-9\s.,]+)', full_text, re.IGNORECASE)
        if acompte_match:
            amounts = self.text_processor.extract_amounts(acompte_match.group(1))
            if amounts:
                info['acompte'] = amounts[0]
        
        # Mentions légales
        legal_keywords = ['information précontractuelle', 'notice HAS', 'code de la santé']
        for keyword in legal_keywords:
            if keyword.lower() in full_text.lower():
                info['mentions_legales'] = True
                break
        
        return info

# -------------------------
# SYSTÈME PRINCIPAL
# -------------------------
class DentalQuoteExtractor:
    """Système principal d'extraction de devis dentaires"""
    
    def __init__(self):
        self.analyzer = DeepPDFAnalyzer()
        self.extractor = DentalDataExtractor()
        self.results_cache = {}
    
    def extract_from_pdf(self, pdf_path: str, debug: bool = False) -> Dict:
        """Extrait toutes les informations d'un PDF"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYSE PROFONDE: {os.path.basename(pdf_path)}")
        logger.info(f"{'='*80}")
        
        try:
            # Étape 1: Analyse de la structure
            logger.info("1. Analyse de la structure du PDF...")
            structure_analysis = self.analyzer.analyze_pdf_structure(pdf_path)
            
            # Étape 2: Extraction du texte
            logger.info("2. Extraction du texte...")
            full_text = self.analyzer.extract_all_text(pdf_path, structure_analysis)
            
            if debug:
                debug_file = f"DEBUG_FULL_{os.path.basename(pdf_path).replace('.pdf', '.txt')}"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                logger.info(f"  ✓ Texte complet sauvegardé: {debug_file}")
            
            # Étape 3: Extraction des tables
            logger.info("3. Extraction des tables...")
            tables = self.analyzer.extract_tables_deep(pdf_path, structure_analysis)
            
            # Étape 4: Extraction des informations de base
            logger.info("4. Extraction des informations de base...")
            basic_info = self.extractor.extract_basic_info(full_text)
            
            # Étape 5: Extraction des traitements
            logger.info("5. Extraction des traitements...")
            treatments = self.extractor.extract_treatments_advanced(full_text, tables)
            
            # Étape 6: Extraction des totaux
            logger.info("6. Extraction des totaux...")
            totals = self.extractor.extract_totals(full_text, treatments)
            
            # Étape 7: Informations complémentaires
            logger.info("7. Extraction des informations complémentaires...")
            additional_info = self.extractor.extract_additional_info(full_text)
            
            # Construction du résultat final
            result = {
                'metadata': {
                    'source_file': pdf_path,
                    'extraction_date': datetime.now().isoformat(),
                    'pdf_structure': structure_analysis,
                    'text_length': len(full_text),
                    'tables_found': len(tables),
                    'treatments_found': len(treatments)
                },
                'basic_info': basic_info,
                'treatments': treatments,
                'financial_summary': totals,
                'additional_info': additional_info,
                'raw_analysis': {
                    'tables_count': len(tables),
                    'extraction_method': 'text_based' if structure_analysis.get('is_text_based') else 'ocr_based'
                }
            }
            
            # Affichage du résumé
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'source_file': pdf_path
            }
    
    def _print_summary(self, result: Dict):
        """Affiche un résumé de l'extraction"""
        logger.info(f"\n{'='*80}")
        logger.info("RÉSUMÉ DE L'EXTRACTION")
        logger.info(f"{'='*80}")
        
        basic = result.get('basic_info', {})
        
        # Dentiste
        dentiste = basic.get('dentiste', {})
        logger.info(f"• Dentiste: {dentiste.get('nom_praticien', 'Non trouvé')}")
        logger.info(f"  RPPS: {dentiste.get('rpps', 'Non trouvé')}")
        
        # Patient
        patient = basic.get('patient', {})
        logger.info(f"• Patient: {patient.get('nom', 'Non trouvé')}")
        
        # Devis
        devis = basic.get('devis', {})
        logger.info(f"• Devis n°: {devis.get('numero', 'Non trouvé')}")
        
        # Traitements
        treatments = result.get('treatments', [])
        logger.info(f"• Traitements trouvés: {len(treatments)}")
        
        if treatments:
            logger.info("• Détail des traitements:")
            for i, t in enumerate(treatments[:10], 1):  # Limite à 10 pour la lisibilité
                code = t.get('code_acte', 'N/A')
                dent = t.get('dent', 'N/A')
                honoraires = t.get('honoraires', 0)
                reste = t.get('reste_a_charge', 0)
                logger.info(f"  {i}. {code} - Dent {dent}: €{honoraires} (reste: €{reste})")
            
            if len(treatments) > 10:
                logger.info(f"  ... et {len(treatments) - 10} autres traitements")
        
        # Totaux
        totals = result.get('financial_summary', {})
        logger.info(f"• Total honoraires: €{totals.get('honoraires_total', 0)}")
        logger.info(f"• Total reste à charge: €{totals.get('reste_a_charge_total', 0)}")
        
        logger.info(f"{'='*80}\n")

# -------------------------
# POINT D'ENTRÉE
# -------------------------
def main():
    """Fonction principale"""
    
    if len(sys.argv) < 2:
        print("Usage: python main_ultra.py fichier.pdf [--debug]")
        print("Exemples:")
        print("  python main_ultra.py devis.pdf")
        print("  python main_ultra.py devis2.pdf --debug")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    if not os.path.exists(pdf_path):
        print(f"Erreur: fichier introuvable: {pdf_path}")
        sys.exit(1)
    
    # Initialiser l'extracteur
    extractor = DentalQuoteExtractor()
    
    # Exécuter l'extraction
    start_time = datetime.now()
    result = extractor.extract_from_pdf(pdf_path, debug=debug_mode)
    end_time = datetime.now()
    
    # Afficher le temps d'exécution
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Temps d'exécution: {duration:.2f} secondes")
    
    # Sauvegarder le résultat JSON
    output_file = f"RESULT_{os.path.basename(pdf_path).replace('.pdf', '.json')}"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False, default=str)
    
    logger.info(f"\nRésultat sauvegardé: {output_file}")
    
    # Afficher un aperçu JSON
    if debug_mode:
        print("\n" + "="*80)
        print("APERÇU DU RÉSULTAT JSON")
        print("="*80)
        print(json.dumps({
            'basic_info': result.get('basic_info'),
            'treatments_count': len(result.get('treatments', [])),
            'financial_summary': result.get('financial_summary'),
            'metadata': result.get('metadata', {})
        }, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()