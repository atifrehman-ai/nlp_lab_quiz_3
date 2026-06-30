# NLP Text Similarity App

A Streamlit app that computes semantic similarity between words or sentences using the free pretrained model `sentence-transformers/all-MiniLM-L6-v2`.

## Features

- Enter multiple text items, one per line.
- Compute cosine similarity between sentence embeddings.
- Display results as:
  - similarity matrix
  - bar chart
  - heatmap
  - 2D PCA embedding visualization
- Includes a short summary section with analysis highlights.

## Requirements

- Python 3.10+ (tested with Anaconda / base environment)
- Streamlit
- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- torch

## Installation

1. Open a terminal in the project folder:

   ```powershell
   cd C:\Users\atifk\Desktop\Quiz3_NLP
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   If you prefer installing only the packages used by the app:

   ```powershell
   pip install streamlit sentence-transformers scikit-learn pandas numpy matplotlib seaborn torch
   ```

## Running the App

Start the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local URL displayed in the terminal, usually `http://localhost:8502`.

## Usage

- Enter one sentence or phrase per line in the text input box.
- Click **Analyze Similarity**.
- The app will generate similarity results and visualization charts.

### Example input

```text
A neon librarian danced through the maze of glowing books at midnight.
The quick brown fox jumps over the lazy dog.
A vintage camera captures light in a quiet museum.
```

## Notes

- The first time you run the app, the model weights are downloaded from Hugging Face and may take a few minutes.
- If the app warns about HF Hub authentication, it is only a rate-limit notice; the app will still work.
- For a faster and cleaner install, use a dedicated Python virtual environment.

## Troubleshooting

- `ModuleNotFoundError: No module named 'sentence_transformers'`
  - Make sure you installed `sentence-transformers` in the same Python environment used by Streamlit.
- `streamlit` starts but page does not load
  - Confirm the terminal output shows `Local URL: http://localhost:8502`.

## License

This project is provided as-is for educational use.
