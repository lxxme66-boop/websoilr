#!/usr/bin/env python3
"""
Script to download required models for QA evaluation
"""

import os
import sys
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import nltk
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_sentence_transformer():
    """Download sentence transformer model"""
    logger.info("Downloading sentence transformer model...")
    try:
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully downloaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to download sentence transformer: {e}")
        return False
    return True


def download_bert_model():
    """Download BERT model"""
    logger.info("Downloading BERT model...")
    try:
        model_name = 'bert-base-chinese'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        logger.info(f"Successfully downloaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to download BERT model: {e}")
        return False
    return True


def download_nltk_data():
    """Download NLTK data"""
    logger.info("Downloading NLTK data...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False
    return True


def download_spacy_model():
    """Download spaCy model"""
    logger.info("Downloading spaCy model...")
    try:
        # Try to load the model first
        try:
            spacy.load("zh_core_web_sm")
            logger.info("spaCy Chinese model already installed")
            return True
        except:
            # If not found, download it
            os.system("python -m spacy download zh_core_web_sm")
            logger.info("Successfully downloaded spaCy Chinese model")
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False
    return True


def main():
    """Main function"""
    logger.info("Starting model download process...")
    
    success = True
    
    # Download models
    if not download_sentence_transformer():
        success = False
    
    if not download_bert_model():
        success = False
    
    if not download_nltk_data():
        success = False
    
    if not download_spacy_model():
        success = False
    
    if success:
        logger.info("All models downloaded successfully!")
    else:
        logger.error("Some models failed to download. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()