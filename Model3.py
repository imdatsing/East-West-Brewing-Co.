import streamlit as st
import pandas as pd
import pickle
from openai import OpenAI
import matplotlib.pyplot as plt

from deep_translator import GoogleTranslator

from collections import Counter
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

def app():
    st.title("Model 3: Customer Feedback Analysis")
    
    df = st.file_uploader(label='Upload your dataset:', type=['csv'])
    
    if df is not None:
        df = pd.read_csv(df, encoding='Windows-1254')
        df2 = df.iloc[:, 0]
        reviews = df2.tolist()
        st.session_state['reviews'] = reviews
        st.session_state['df'] = df
    elif 'df' in st.session_state:
        df = st.session_state['df']
        reviews = st.session_state['reviews']
    else:
        st.error("Please upload a CSV file.")
        return
    
    st.write(df.head(5))

    if st.button('Predict'):
        if 'aspect_category' in df.columns:
            df = df.drop(columns=['aspect_category'])
        df = df.dropna()
        df = drop_exceed(df)
        df = split_by_bar(df)
        # df = translate_column(df, 'review_text', source_lang='auto', target_lang='en')
        gen_ug(df)
        feature_dic_dev = [featurize(x) for x in df['ug']]

        # Load model
        with open('Model3Vectorizer.pkl', 'rb') as model_file:
            vectorizer = pickle.load(model_file)
        feature_vec_dev = vectorizer.transform(feature_dic_dev)
        with open('Model3.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        predict_aspect = model.predict(feature_vec_dev)
        df['predicted_aspect'] = predict_aspect
        df = sentiment_analysis(df)
        st.write(df[['review_text', 'review_rating', 'predicted_aspect', 'review_date', 'sentiment']])

        st.header("Word Cloud")
        chart3(df, 'review_text')

        st.header("Number of Reviews by Day of the Week and Sentiment")
        chart1(df)

        st.header("Sentiment Distribution Across Aspects")
        chart2(df)

        st.session_state['df'] = df

    if st.button('Recommendation'):
        if 'summary' not in st.session_state or 'recommendation' not in st.session_state:
            if 'df' in st.session_state:
                reviews = st.session_state['reviews']
                summary = summarize_reviews(reviews)
                recommendation = rcm(reviews)
                st.session_state['summary'] = summary
                st.session_state['recommendation'] = recommendation
            else:
                st.error("Please predict first to get recommendations.")
                return

        st.header("Summarized Analysis")
        st.write(st.session_state['summary'])
        st.header("Key Takeaways and Recommendation")
        st.write(st.session_state['recommendation'])
        

### --- Drop review_text that exceed 5000 characters (because translate only work no more than 5000)
def drop_exceed(df):
    df['review_text'] = df['review_text'].astype(str)
    df = df[df['review_text'].str.len() <= 5000]
    return df

### --- Translate review_text
# def translate_column(df, column_name, source_lang='auto', target_lang='en'):
#     translator = GoogleTranslator(source='auto', target='en')
#     translated_text = []
    
    # Duyệt qua từng dòng trong cột và dịch
    for index, row in df.iterrows():
        text = row[column_name]
        translated = translator.translate(text, src=source_lang, dest=target_lang)
        translated_text.append(translated)
    
    # Tạo một cột mới chứa văn bản đã dịch
    df[column_name] = translated_text
    
    return df  

### --- Tokenize Review
def split_by_bar(df, column_name='review_text'):
    df['tokens'] = df[column_name].apply(lambda x: x.split())
    return df

### --- Unigram review
def unigram(lst):
  n_gram = 1
  return Counter(ngrams(lst, n_gram))

def gen_ug(df, column_name='tokens'):
  df['ug'] = df['tokens'].apply(unigram)
  return df

### --- Featurize
def featurize(token_list): 
  features = {} # add output into dict = dict of features = list of feature dict
  for i in token_list:
    features[i] = 1 
  return features

sia = SentimentIntensityAnalyzer()
def sentiment_analysis(df):
    sentiments = []
    sentiment_labels = []
    scores = []
    for review_text in df['review_text']:
        sentiment = sia.polarity_scores(review_text)
        score = sia.polarity_scores(review_text)
        score = score['compound']
        sentiments.append(sentiment)
        scores.append(score)
    for sentiment in sentiments:
        compound_score = sentiment['compound']
        if compound_score >= 0.05:
            sentiment_labels.append('Positive')
        elif compound_score <= -0.05:
            sentiment_labels.append('Negative')
        else:
            sentiment_labels.append('Neutral')
    df['sentiment'] = sentiment_labels
    df['score'] = scores
    return df

def summarize_reviews(reviews):
    client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])
    review_texts = "\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews)])
    prompt = (
        f"Here are several reviews of a restaurant:\n\n"
        f"{review_texts}\n\n"
        "You are an expert in sentiment analysis and feedback summarization. Your task is to analyze the customer feedback provided below and summarize the main points, highlighting both positive and negative aspects. The goal is to understand the overall sentiment and key points of the feedback."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    summary = response.choices[0].message.content.strip()
    return summary

def rcm(reviews):
    client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])
    review_texts = "\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews)])
    prompt = (
        f"Here are several reviews of a restaurant:\n\n"
        f"{review_texts}\n\n"
        "You are an experienced restaurant management consultant with deep knowledge of the hospitality industry. You specialize in analyzing customer feedback and providing actionable recommendations to improve the dining experience, operational efficiency, and overall customer satisfaction."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    recommendation = response.choices[0].message.content.strip()
    return recommendation

def chart1(df):
    # Convert the ReviewDate column to datetime
    df['review_date'] = pd.to_datetime(df['review_date'])

    # Extract the day of the week from the ReviewDate
    df['DayOfWeek'] = df['review_date'].dt.day_name()

    # Count the number of reviews per day of the week for each sentiment
    sentiment_counts = df.groupby(['DayOfWeek', 'sentiment']).size().unstack(fill_value=0)

    # Reorder the DataFrame to have the days of the week in the correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sentiment_counts = sentiment_counts.reindex(days_order)

    # Define sentiment colors
    sentiments = ['Positive', 'Neutral', 'Negative']
    colors = {'Positive': 'lightskyblue', 'Neutral': 'lightpink', 'Negative': 'navajowhite'}  # Blue, Pink, Orange
    days = range(len(days_order))
    bar_width = 0.25

    # Plot the data in a grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate bars for each sentiment
    for i, sentiment in enumerate(sentiments):
        bars = ax.bar([d + i * bar_width for d in days], sentiment_counts[sentiment], width=bar_width, label=sentiment, color=colors[sentiment])
        
        # Add annotations
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:  # Only annotate bars with a height greater than zero
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Adding labels and title
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Reviews')
    ax.set_xticks([d + bar_width for d in days])
    ax.set_xticklabels(days_order)
    ax.legend()

    # Display the chart
    st.pyplot(fig)
    
def chart2(df):
    # Aggregate the data to count the number of sentiments for each aspect
    aspect_sentiment_counts = df.groupby(['predicted_aspect', 'sentiment']).size().unstack(fill_value=0)

    # Define a consistent color mapping for sentiments
    colors = {
        'Positive': 'lightskyblue',
        'Neutral': 'lightpink',
        'Negative': 'navajowhite'
    }

    # Calculate the number of rows needed
    num_aspects = len(aspect_sentiment_counts)
    num_cols = 3  # Adjust the number of columns
    num_rows = (num_aspects + num_cols - 1) // num_cols

    # Plot larger pie charts for each aspect
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))  # Adjusted figure size and layout

    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for ax, (aspect, counts) in zip(axs, aspect_sentiment_counts.iterrows()):
        # Filter out zero values
        counts = counts[counts > 0]
        counts_labels = counts.index.tolist()
        counts_colors = [colors[sentiment] for sentiment in counts_labels]
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=counts_colors, 
            labeldistance=1,  # Distance of the labels from the center
            pctdistance=0.7,    # Distance of the percentage labels from the center
            textprops={'fontsize': 14}
        )
        ax.set_title(aspect, fontsize=16)  # Adjusted font size of the title
        
        # Add legend for each subplot
        ax.legend(wedges, counts_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14, title_fontsize=14, handletextpad=0.5, labelspacing=0.5)

    # Hide any unused subplots
    for i in range(len(aspect_sentiment_counts), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()  # Adjust layout to prevent overlap
    st.pyplot(fig)
    
def chart3(df, column):
    # Join all the text in the specified column
    mega_text = ' '.join(df[column])

    # Generate the word cloud
    wordcloud = WordCloud(width=1200, height=500,background_color='white').generate(mega_text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))  # Set the facecolor to #0e1117
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove the axis
    st.pyplot(plt)
 
