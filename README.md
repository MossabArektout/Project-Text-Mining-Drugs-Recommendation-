# Drug Recommendation System - NLP Project

**Authors:** Mossab Arektout and Abderrahim Mabrouk
**Course:** Natural Language Processing / Text Mining

## Overview

This project implements a content-based drug recommendation system using Natural Language Processing (NLP) techniques. The system uses TF-IDF vectorization and cosine similarity to recommend similar medicines based on their names, reasons for use, and descriptions.

## Features

- **Text Preprocessing Pipeline**: Lowercase conversion, special character removal, whitespace normalization
- **TF-IDF Vectorization**: Advanced feature extraction with bigrams and optimized parameters
- **Cosine Similarity**: Efficient similarity calculation for 9,720+ medicines
- **Baseline Comparison**: Random recommendation baseline to validate TF-IDF approach
- **Precision@K and Recall@K**: Comprehensive evaluation metrics
- **Interactive UI**: Dropdown-based interface for testing recommendations
- **Visualizations**: Multiple charts showing similarity distributions and performance metrics
- **Future Work**: Detailed roadmap for advanced NLP techniques (BERT, Word2Vec, etc.)

## Project Structure

```
Text_Mining/
├── Drug_Recommendation.ipynb       # Main Jupyter notebook with all code and analysis
├── medicine.csv            # Dataset with 9,720 medicine records
├── Drugs_Rec.pdf          # Presentation slides (French)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Text_Mining.ipynb
   ```

## Requirements

- Python 3.8+
- pandas 2.2.2
- numpy 2.0.2
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0
- ipywidgets >= 8.0.0

See [requirements.txt](requirements.txt) for complete list.

## Dataset

- **Total records:** 9,720 medicines
- **Unique drug names:** 9,626
- **Medical conditions:** 50 categories
- **Top conditions:** Hypertension (2,505), Infection (1,109), Pain (1,072)
- **No missing values**

## Usage

### Running the Notebook

Open [Text_Mining.ipynb](Text_Mining.ipynb) and run all cells sequentially to:

1. Load and explore the dataset
2. Preprocess text data
3. Generate TF-IDF vectors
4. Calculate similarity matrix
5. Test the recommendation system
6. View evaluation metrics and visualizations

### Interactive Testing

The notebook includes an interactive widget where you can:
- Select any medicine from a dropdown menu
- View top 10 recommendations with similarity scores
- See visual similarity bars
- Get statistics about recommendation quality

### Example Usage

```python
# Get recommendations for a medicine
result = recommend("A CN Gel(Topical) 20gmA CN Soap 75gm", medicines, similarity, top_n=5)

# Display results
for i, rec in enumerate(result['recommendations'], 1):
    print(f"{i}. {rec['name']} - Similarity: {rec['score']:.4f}")
```

## Results

### Performance Metrics

- **Average similarity score:** 0.8362
- **High similarity ratio (>0.5):** 96.20%
- **Recommendations evaluated:** 500

### Precision@K and Recall@K

| K  | Precision@K | Recall@K |
|----|-------------|----------|
| 1  | High        | Low      |
| 3  | High        | Medium   |
| 5  | High        | Medium   |
| 10 | Medium      | High     |

*Run the notebook to see exact values*

### Baseline Comparison

The TF-IDF approach significantly outperforms random recommendations by ensuring recommended medicines treat the same medical condition with high textual similarity.

## Methodology

### 1. Text Preprocessing
- Convert to lowercase
- Remove special characters
- Normalize whitespace
- Combine multiple text fields (name + reason + description)

### 2. Feature Extraction
- **TF-IDF Vectorization** with:
  - English stop words removal
  - Unigrams and bigrams (1,2)
  - 5,000 max features
  - Min document frequency: 2
  - Max document frequency: 0.95

### 3. Similarity Calculation
- **Cosine Similarity** matrix (9,720 × 9,720)
- Efficient sparse matrix representation
- 99.55% sparsity

### 4. Recommendation Generation
- Rank by similarity score
- Filter top-N results
- Return with medical condition and scores

## Evaluation

### Metrics Implemented

1. **Similarity Score Distribution**: Histogram and box plots
2. **Baseline Comparison**: TF-IDF vs Random recommendations
3. **Precision@K**: Proportion of relevant items in top-K
4. **Recall@K**: Proportion of relevant items retrieved
5. **Heatmap Visualization**: Similarity matrix sample

## Future Enhancements

The project identifies several advanced techniques for future work:

### Advanced NLP Techniques
- **Word2Vec / GloVe**: Semantic word embeddings
- **BioBERT**: Medical domain transformer models
- **Graph Neural Networks**: Knowledge graph-based recommendations

### Additional Features
- Drug interaction analysis
- Side effect considerations
- Multilingual support
- Explainable AI (LIME/SHAP)
- Hybrid recommendation systems

See the notebook's "Future Work" section for detailed descriptions and implementation approaches.

## Presentation

The project includes professional presentation slides ([Drugs_Rec.pdf](Drugs_Rec.pdf)) covering:
1. Problem statement and objectives
2. Dataset characteristics
3. NLP pipeline methodology
4. Results and visualizations
5. Conclusion

## Key Achievements

1. **Comprehensive NLP Pipeline**: End-to-end implementation from preprocessing to evaluation
2. **High-Quality Recommendations**: 96.20% of recommendations have high similarity (>0.5)
3. **Rigorous Evaluation**: Multiple metrics including baseline comparison and precision/recall
4. **Interactive Demo**: User-friendly interface for testing
5. **Academic Rigor**: Mathematical formulas, statistical analysis, and professional documentation

## License

This project is for educational purposes as part of a Natural Language Processing course.

## Contact

For questions or feedback about this project:
- **Mossab Arektout**
- **Abderrahim Mabrouk**

## Acknowledgments

- Course: Natural Language Processing / Text Mining
- Dataset: Medicine information with drug names, conditions, and descriptions
- Tools: Python, scikit-learn, pandas, matplotlib, Jupyter

---

**Last Updated:** January 2025
