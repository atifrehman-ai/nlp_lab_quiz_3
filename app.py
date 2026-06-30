import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="NLP Text Similarity App",
    layout="wide"
)

st.title("🔍 NLP Text Similarity using Free Pretrained Model")

st.write("""
This application uses the free pretrained model
**sentence-transformers/all-MiniLM-L6-v2**
to calculate similarity between words or sentences.
""")


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


st.header("Input Text")

text_input = st.text_area(
    "Enter one word or sentence per line:",
    height=200,
    placeholder="""Apple
Orange
Banana
Fruit
Car
Vehicle"""
)


if st.button("Analyze Similarity"):

    texts = [line for line in text_input.split("\n") if line.strip()]

    if len(texts) < 2:
        st.warning("Please enter at least two words or sentences.")
        st.stop()

    
    embeddings = model.encode(texts)

   
    similarity_matrix = cosine_similarity(embeddings)

    st.header("Similarity Matrix")

    df = pd.DataFrame(
        similarity_matrix,
        index=texts,
        columns=texts
    )

    st.dataframe(df)

    
    st.header(" Graph 1 : Bar Chart")

    reference = 0

    scores = similarity_matrix[reference]

    result = pd.DataFrame({
        "Text": texts,
        "Similarity": scores
    })

    result = result.sort_values(
        by="Similarity",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(result["Text"], result["Similarity"])

    ax.set_xlabel("Words / Sentences")
    ax.set_ylabel("Similarity Score")
    ax.set_title(f"Similarity with '{texts[reference]}'")

    plt.xticks(rotation=45)

    st.pyplot(fig)

   
    st.header("🔥 Graph 2 : Heatmap")

    fig2, ax2 = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df,
        annot=True,
        cmap="YlGnBu",
        ax=ax2
    )

    st.pyplot(fig2)

    
    st.header("🌍 Graph 3 : 2D PCA Embedding")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(8,6))

    ax3.scatter(
        reduced[:,0],
        reduced[:,1]
    )

    for i, txt in enumerate(texts):
        ax3.annotate(
            txt,
            (reduced[i,0], reduced[i,1])
        )

    ax3.set_title("2D Embedding Visualization")

    st.pyplot(fig3)

   
    st.header("🏆 Top Similar Results")

    top_results = result.iloc[1:]

    st.table(top_results)

   
    st.header(" Paul's Critical Thinking Standards")

    st.subheader("1. Clarity")
    st.write(
        f"The user entered {len(texts)} text items. "
        "The application compares them using a pretrained NLP model."
    )

    st.subheader("2. Accuracy")
    st.write(
        "Model Used: sentence-transformers/all-MiniLM-L6-v2."
        " No training or preprocessing was performed."
    )

    st.subheader("3. Precision")
    st.write(
        f"Highest similarity score: "
        f"{top_results.iloc[0]['Similarity']:.4f}"
    )

    st.subheader("4. Relevance")
    st.write(
        "The bar chart, heatmap, and PCA plot directly support the similarity results."
    )

    st.subheader("5. Logic")
    st.write(
        "Texts with similar meanings have embeddings that are closer together, resulting in higher cosine similarity scores."
    )

    st.subheader("6. Significance")
    st.write(
        f"The most significant result is the text pair with the highest non-identical similarity score: "
        f"'{top_results.iloc[0]['Text']}' with a score of {top_results.iloc[0]['Similarity']:.4f}."
    )

    st.subheader("7. Fairness")
    st.write(
        "The pretrained model may not fully understand domain-specific language or very rare terms."
    )

    st.success("Analysis Completed Successfully!")
