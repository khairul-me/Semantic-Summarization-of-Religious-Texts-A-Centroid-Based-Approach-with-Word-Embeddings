import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from typing import List, Dict, Tuple
import re
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the NLTK data path
nltk_data_path = 'K:/Khairul_Etin_research'
nltk.data.path = [nltk_data_path]

# Ensure required NLTK data is downloaded
def setup_nltk():
    try:
        if not os.path.exists(f"{nltk_data_path}/tokenizers/punkt"):
            nltk.download('punkt', download_dir=nltk_data_path)
        if not os.path.exists(f"{nltk_data_path}/corpora/stopwords"):
            nltk.download('stopwords', download_dir=nltk_data_path)
        logging.info("NLTK resources checked and loaded successfully")
    except Exception as e:
        logging.error(f"Error setting up NLTK: {e}")
        raise

class PDFSummarizer:
    def __init__(self):
        """Initialize the summarizer with required models."""
        try:
            self.nlp = spacy.load('en_core_web_md')
            self.stop_words = set(stopwords.words('english'))
            logging.info("PDFSummarizer initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing PDFSummarizer: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            if not text.strip():
                raise ValueError("Extracted text is empty")
            
            logging.info(f"Successfully extracted text from {pdf_path}")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and split text into sentences."""
        try:
            # Basic cleaning
            text = text.replace('\n', ' ')
            text = re.sub(r'\s+', ' ', text)
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Clean sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                # Keep sentences that have meaningful content
                if len(sentence.split()) > 3 and any(c.isalnum() for c in sentence):
                    cleaned_sentences.append(sentence)
            
            if not cleaned_sentences:
                raise ValueError("No valid sentences found after preprocessing")
            
            logging.info(f"Preprocessed text into {len(cleaned_sentences)} sentences")
            return cleaned_sentences
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            raise

    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Convert sentences to embeddings using spaCy."""
        try:
            embeddings = []
            for sentence in sentences:
                doc = self.nlp(sentence)
                if len(doc) > 0:
                    # Get vectors for non-stop words
                    vectors = [token.vector for token in doc 
                             if not token.is_stop and token.has_vector]
                    if vectors:
                        sent_vector = np.mean(vectors, axis=0)
                        embeddings.append(sent_vector)
            
            if not embeddings:
                raise ValueError("No valid embeddings generated")
            
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"Error generating sentence embeddings: {e}")
            raise

    def calculate_centroid(self, sentences: List[str]) -> np.ndarray:
        """Calculate centroid using TF-IDF weighted word embeddings."""
        try:
            # Calculate TF-IDF
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(sentences)
            
            # Get important words
            feature_names = tfidf.get_feature_names_out()
            important_words = []
            for i in range(len(sentences)):
                top_indices = np.argsort(tfidf_matrix[i].toarray()[0])[-5:]
                important_words.extend([feature_names[idx] for idx in top_indices])
            
            # Calculate centroid
            word_vectors = [self.nlp(word).vector for word in set(important_words)]
            centroid = np.mean(word_vectors, axis=0)
            return centroid
        except Exception as e:
            logging.error(f"Error calculating centroid: {e}")
            raise

    def generate_summary(self, sentences: List[str], embeddings: np.ndarray, 
                        centroid: np.ndarray, num_sentences: int = 5) -> List[str]:
        """Select top sentences based on similarity to centroid."""
        try:
            # Calculate similarities
            similarities = []
            for embedding in embeddings:
                similarity = np.dot(embedding, centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(centroid))
                similarities.append(similarity)
            
            # Select top sentences
            num_sentences = min(num_sentences, len(sentences))
            top_indices = np.argsort(similarities)[-num_sentences:]
            summary = [sentences[idx] for idx in sorted(top_indices)]
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            raise

    def visualize_sentences(self, embeddings: np.ndarray, centroid: np.ndarray, 
                          summary_indices: List[int], output_path: str):
        """Create visualization of sentence embeddings."""
        try:
            # Prepare data for visualization
            all_vectors = np.vstack([embeddings, centroid])
            tsne = TSNE(n_components=2, random_state=42)
            vectors_2d = tsne.fit_transform(all_vectors)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot regular sentences
            plt.scatter(vectors_2d[:-1, 0], vectors_2d[:-1, 1], 
                       c='blue', alpha=0.5, label='Sentences')
            
            # Plot summary sentences
            plt.scatter(vectors_2d[summary_indices, 0], vectors_2d[summary_indices, 1],
                       c='green', marker='*', s=200, label='Summary Sentences')
            
            # Plot centroid
            plt.scatter(vectors_2d[-1, 0], vectors_2d[-1, 1],
                       c='red', marker='X', s=200, label='Centroid')
            
            plt.title('Sentence Embeddings Visualization')
            plt.legend()
            
            # Save plot
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            
            logging.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
            raise

    def process_pdf(self, pdf_path: str, output_dir: str, num_sentences: int = 5):
        """Main method to process PDF and generate summary with visualizations."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process text
            text = self.extract_text_from_pdf(pdf_path)
            sentences = self.preprocess_text(text)
            
            # Generate embeddings and summary
            embeddings = self.get_sentence_embeddings(sentences)
            centroid = self.calculate_centroid(sentences)
            summary = self.generate_summary(sentences, embeddings, centroid, num_sentences)
            
            # Create visualization
            summary_indices = [sentences.index(sent) for sent in summary]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save outputs
            viz_path = os.path.join(output_dir, f"visualization_{timestamp}.png")
            self.visualize_sentences(embeddings, centroid, summary_indices, viz_path)
            
            summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
            with open(summary_path, "w", encoding='utf-8') as f:
                f.write("=== Document Summary ===\n\n")
                for i, sent in enumerate(summary, 1):
                    f.write(f"{i}. {sent}\n\n")
            
            # Print summary to console
            print("\n=== Generated Summary ===\n")
            for i, sent in enumerate(summary, 1):
                print(f"{i}. {sent}\n")
            
            logging.info(f"Summary and visualization saved to {output_dir}")
            return summary
            
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise

def main():
    """Main function to run the summarizer."""
    try:
        # Setup NLTK
        setup_nltk()
        
        # Initialize summarizer
        summarizer = PDFSummarizer()
        
        # Process PDF
        pdf_path = "K:/Khairul_Etin_research/AppiahChapter04.pdf"
        output_dir = "K:/Khairul_Etin_research/output"
        
        summary = summarizer.process_pdf(
            pdf_path,
            output_dir,
            num_sentences=5
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"\nError: {e}")
        print("Please check the log file for more details.")

if __name__ == "__main__":
    main()