import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(
    page_title="InsightSphere",
    layout="wide"
)

st.title("InsightSphere")

st.subheader(
    "AI-Powered Customer Sentiment & Business Intelligence Dashboard"
)

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

sample_size = st.sidebar.slider(
    "Select Sample Size",
    1000,
    30000,
    10000,
    1000
)

cluster_count = st.sidebar.slider(
    "Customer Groups",
    2,
    10,
    4
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.sidebar.subheader("Select Dataset Columns")

    columns = df.columns.tolist()

    optional_columns = ["N/A"] + columns

    review_column = st.sidebar.selectbox(
        "Select Review Column",
        columns
    )

    rating_column = st.sidebar.selectbox(
        "Select Rating Column",
        columns
    )

    sentiment_column = st.sidebar.selectbox(
        "Select Sentiment Column",
        optional_columns
    )

    product_price_column = st.sidebar.selectbox(
        "Select Product Price Column",
        optional_columns
    )

    product_name_column = st.sidebar.selectbox(
        "Select Product Name Column",
        optional_columns
    )

    run_analysis = st.sidebar.button("Run Analysis")

    if run_analysis:

        df = df.sample(
            min(sample_size, len(df)),
            random_state=42
        )

        selected_columns = [
            review_column,
            rating_column
        ]

        if sentiment_column != "N/A":
            selected_columns.append(sentiment_column)

        if product_price_column != "N/A":
            selected_columns.append(product_price_column)

        if product_name_column != "N/A":
            selected_columns.append(product_name_column)

        df = df[selected_columns]

        rename_dict = {
            review_column: 'Review',
            rating_column: 'Rate'
        }

        if sentiment_column != "N/A":
            rename_dict[sentiment_column] = 'Sentiment'
        else:
            df['Sentiment'] = 'neutral'

        if product_price_column != "N/A":
            rename_dict[product_price_column] = 'product_price'
        else:
            df['product_price'] = 0

        if product_name_column != "N/A":
            rename_dict[product_name_column] = 'product_name'
        else:
            df['product_name'] = 'Unknown Product'

        df.rename(columns=rename_dict, inplace=True)

        df.dropna(inplace=True)

        lemmatizer = WordNetLemmatizer()

        stop_words = set(stopwords.words('english'))

        def clean_text(text):

            text = str(text).lower()

            text = re.sub(r'\d+', '', text)

            text = re.sub(r'[^\w\s]', '', text)

            words = text.split()

            words = [
                lemmatizer.lemmatize(word)
                for word in words
                if word not in stop_words
            ]

            return " ".join(words)

        df['clean_review'] = df['Review'].apply(clean_text)

        tfidf = TfidfVectorizer(max_features=3000)

        X = tfidf.fit_transform(df['clean_review'])

        y = df['Sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LogisticRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }

        df['sentiment_score'] = (
            df['Sentiment']
            .astype(str)
            .str.lower()
            .map(sentiment_map)
        )

        df['review_length'] = df['clean_review'].apply(len)

        df['Rate'] = pd.to_numeric(
            df['Rate'],
            errors='coerce'
        )

        df['product_price'] = pd.to_numeric(
            df['product_price'],
            errors='coerce'
        )

        df.dropna(inplace=True)

        cluster_data = df[
            [
                'Rate',
                'sentiment_score',
                'review_length',
                'product_price'
            ]
        ]

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(cluster_data)

        kmeans = KMeans(
            n_clusters=cluster_count,
            random_state=42
        )

        clusters = kmeans.fit_predict(scaled_data)

        df['cluster'] = clusters

        persona_names = {
            0: "Loyal Customers",
            1: "Budget-Conscious Buyers",
            2: "Dissatisfied Customers",
            3: "Premium Buyers",
            4: "Frequent Reviewers",
            5: "High Value Customers",
            6: "Occasional Buyers",
            7: "Neutral Customers",
            8: "Price Sensitive Users",
            9: "Returning Customers"
        }

        df['customer_group'] = df['cluster'].map(persona_names)

        st.header("Business Insights Dashboard")

        positive_pct = round(
            (
                len(df[df['sentiment_score'] == 1]) / len(df)
            ) * 100,
            1
        )

        negative_pct = round(
            (
                len(df[df['sentiment_score'] == -1]) / len(df)
            ) * 100,
            1
        )

        avg_rating = round(df['Rate'].mean(), 2)

        best_product = (
            df.groupby('product_name')['sentiment_score']
            .mean()
            .idxmax()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Positive Customers",
            f"{positive_pct}%"
        )

        col2.metric(
            "Negative Customers",
            f"{negative_pct}%"
        )

        col3.metric(
            "Average Rating",
            f"{avg_rating}/5"
        )

        col4.metric(
            "Top Product",
            str(best_product)[:20]
        )

        st.divider()

        st.subheader("Overall Customer Mood")

        fig, ax = plt.subplots(figsize=(4,3))

        sns.countplot(
            x='Sentiment',
            data=df,
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

        st.markdown("### Key Takeaways")

        if positive_pct > 70:

            st.success(
                "Customers are generally satisfied with the products and services."
            )

        if negative_pct > 25:

            st.warning(
                "A significant number of customers are unhappy and may require attention."
            )

        st.info(
            "Reviews mainly reflect customer experience, pricing, quality, and delivery feedback."
        )

        st.subheader("Ratings Distribution")

        rating_data = df.dropna(subset=['Rate'])

        fig, ax = plt.subplots(figsize=(5,3))

        sns.countplot(
            x=rating_data['Rate'],
            ax=ax
        )

        ax.set_xlabel("Rating")

        ax.set_ylabel("Count")

        plt.tight_layout()

        st.pyplot(fig)

        st.subheader("Product Performance Analysis")

        product_sentiment = (
            df.groupby('product_name')['sentiment_score']
            .mean()
            .sort_values(ascending=False)
        )

        best_products = product_sentiment.head(5)

        worst_products = product_sentiment.tail(5)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Best Performing Products")

            for item in best_products.index:

                st.success(item)

        with col2:

            st.markdown("### Products Needing Attention")

            for item in worst_products.index:

                st.error(item)


        st.subheader("Customer Behavior Groups")

        group_summary = df.groupby('customer_group')[
            [
                'Rate',
                'sentiment_score',
                'review_length'
            ]
        ].mean()

        st.dataframe(group_summary)

        fig, ax = plt.subplots(figsize=(6,3))

        sns.countplot(
            y='customer_group',
            data=df,
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

        st.markdown("### Customer Group Insights")

        for group in group_summary.index:

            avg_sentiment = group_summary.loc[
                group,
                'sentiment_score'
            ]

            if avg_sentiment > 0.5:

                st.success(
                    f"{group}: Customers are highly satisfied and likely to return."
                )

            elif avg_sentiment < -0.5:

                st.warning(
                    f"{group}: Customers are unhappy and may stop purchasing."
                )

            else:

                st.info(
                    f"{group}: Customers show mixed or neutral opinions."
                )

        st.subheader("Top Customer Concerns & Trends")

        positive_reviews = " ".join(
            df[
                df['Sentiment']
                .astype(str)
                .str.lower() == 'positive'
            ]['clean_review']
        )

        negative_reviews = " ".join(
            df[
                df['Sentiment']
                .astype(str)
                .str.lower() == 'negative'
            ]['clean_review']
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Positive Trends")

            if len(positive_reviews) > 0:

                wordcloud = WordCloud(
                    width=500,
                    height=250,
                    background_color='white'
                ).generate(positive_reviews)

                fig, ax = plt.subplots(figsize=(5,2.5))

                ax.imshow(wordcloud)

                ax.axis('off')

                plt.tight_layout()

                st.pyplot(fig)

        with col2:

            st.markdown("### Customer Complaints")

            if len(negative_reviews) > 0:

                wordcloud = WordCloud(
                    width=500,
                    height=250,
                    background_color='white'
                ).generate(negative_reviews)

                fig, ax = plt.subplots(figsize=(5,2.5))

                ax.imshow(wordcloud)

                ax.axis('off')

                plt.tight_layout()

                st.pyplot(fig)

        st.header("AI-Powered Business Recommendations")

        st.write("• Improve products receiving consistently negative reviews.")
        st.write("• Investigate recurring customer complaints.")
        st.write("• Reward loyal and highly satisfied customers.")
        st.write("• Improve delivery, packaging, or pricing strategies where needed.")
        st.write("• Focus marketing campaigns on high-performing products.")

        with st.expander("View Detailed AI/ML Analysis"):

            st.subheader("Model Accuracy")

            st.metric(
                "Accuracy",
                f"{round(accuracy * 100, 2)}%"
            )

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(4,3))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax
            )

            plt.tight_layout()

            st.pyplot(fig)

            st.subheader("Elbow Method")

            wcss = []

            for i in range(1, 11):

                temp_model = KMeans(
                    n_clusters=i,
                    random_state=42
                )

                temp_model.fit(scaled_data)

                wcss.append(temp_model.inertia_)

            fig, ax = plt.subplots(figsize=(5,3))

            ax.plot(
                range(1,11),
                wcss,
                marker='o'
            )

            plt.tight_layout()

            st.pyplot(fig)

            st.subheader("PCA Cluster Visualization")

            pca = PCA(n_components=2)

            pca_result = pca.fit_transform(scaled_data)

            pca_df = pd.DataFrame()

            pca_df['PCA1'] = pca_result[:,0]

            pca_df['PCA2'] = pca_result[:,1]

            pca_df['Cluster'] = clusters

            fig, ax = plt.subplots(figsize=(5,3.5))

            sns.scatterplot(
                x='PCA1',
                y='PCA2',
                hue='Cluster',
                data=pca_df,
                palette='Set1',
                ax=ax
            )

            plt.tight_layout()

            st.pyplot(fig)

        st.success(
            "Business insight generation completed successfully."
        )

    else:

        st.info(
            "Please configure dataset columns and click Run Analysis"
        )

else:

    st.info("Please upload a CSV dataset to continue")