import streamlit as st
import requests
import os
import re
import json
import math
import time
from collections import Counter
from dotenv import load_dotenv

# Load Together API key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY") or st.secrets.get("TOGETHER_API_KEY")

# Constants
FEEDBACK_FILE = 'feedback_data.json'

# Streamlit setup
st.set_page_config(page_title="NeuraNest - Your YouTube AI Agent")
st.title("🎬 CreatorNest (by NeuraNest Labs)")
st.caption("Smart AI for Smarter Creators — Beta")
st.warning("Beta notice: Results may be inaccurate. Don’t paste sensitive data. "
           "By using this app you agree to our basic privacy note: inputs may be processed by API providers.")

# Task selector with Analytics added
task = st.selectbox("Choose a task", [
    "Generate Video Ideas",
    "Write YouTube Description",
    "Generate Comment Reply",
    "Generate Hashtags",
    "Create Video Script",
    "Analytics-Based Decision Engine"  # New functionality
])

# Topic input (not needed for Analytics, but harmless)
topic = st.text_input("Enter your video topic or comment:")

# Duration input for script generation
script_duration = ""
if task == "Create Video Script":
    duration_option = st.radio(
        "How would you like to specify the script duration?",
        ["Select from list", "Enter custom duration"]
    )

    if duration_option == "Select from list":
        script_duration = st.selectbox(
            "Select desired video script duration:",
            [f"{i} minute{'s' if i > 1 else ''}" for i in range(1, 61)]
        )
    else:
        script_duration = st.text_input(
            "Enter custom duration (e.g., 4.5 minutes, 90 seconds):",
            placeholder="e.g., 15 minutes"
        )

# Helper: Convert duration to minutes
def get_minutes(duration_str):
    try:
        number_match = re.search(r"[\d.]+", duration_str)
        unit_match = re.search(r"(second|minute|hour)", duration_str.lower())

        if not number_match or not unit_match:
            return 5  # fallback

        value = float(number_match.group())
        unit = unit_match.group()

        if "second" in unit:
            return value / 60
        elif "minute" in unit:
            return value
        elif "hour" in unit:
            return value * 60
    except:
        return 5

# Together API function to generate text chunks
def generate_chunk(prompt, max_tokens):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful AI YouTube assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception: {e}"

# Save feedback locally in JSON file
def save_feedback(entry, filename=FEEDBACK_FILE):
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    data.append(entry)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Get prompt style adaptively based on feedback ratings
def get_prompt_style(filename=FEEDBACK_FILE):
    if not os.path.exists(filename):
        return "engaging and informative"
    with open(filename, 'r') as f:
        data = json.load(f)
    ratings = [entry.get('rating', 0) for entry in data if entry.get('rating') is not None]
    if not ratings:
        return "engaging and informative"
    avg_rating = sum(ratings) / len(ratings)
    if avg_rating < 3:
        return "simple and beginner-friendly"
    else:
        return "engaging and informative"

# Load feedback data for analytics
def load_feedback(filename=FEEDBACK_FILE):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return json.load(f)

# Summarize feedback for analytics dashboard
def summarize_feedback(data):
    ratings = [entry.get('rating') for entry in data if entry.get('rating') is not None]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    count = len(ratings)

    topics = [entry.get('topic') for entry in data if entry.get('topic')]
    topic_counts = Counter(topics).most_common(5)

    comments = [entry.get('comments', '') for entry in data if entry.get('comments')]
    stopwords = {'the', 'and', 'to', 'a', 'is', 'in', 'of', 'for', 'it', 'on', 'with', 'this', 'but', 'too'}
    words = []
    for comment in comments:
        words.extend([w.lower() for w in re.findall(r'\b\w+\b', comment) if w.lower() not in stopwords])
    word_freq = Counter(words).most_common(10)

    return {
        'average_rating': avg_rating,
        'feedback_count': count,
        'top_topics': topic_counts,
        'common_comment_words': word_freq
    }

# Main generation and feedback handling
if st.button("Generate") and topic:
    with st.spinner("Generating..."):

        if task == "Generate Video Ideas":
            prompt = f"Give 3 trending YouTube video ideas for: {topic}."
            result = generate_chunk(prompt, max_tokens=300)
            st.success("✅ Generated!")
            st.markdown(result)

        elif task == "Write YouTube Description":
            prompt = f"Write an SEO-optimized YouTube video description for: {topic}."
            result = generate_chunk(prompt, max_tokens=400)
            st.success("✅ Generated!")
            st.markdown(result)

        elif task == "Generate Comment Reply":
            prompt = f"Reply to this YouTube comment: '{topic}'"
            result = generate_chunk(prompt, max_tokens=200)
            st.success("✅ Generated!")
            st.markdown(result)

        elif task == "Generate Hashtags":
            prompt = f"Generate 10 trending and relevant hashtags for: {topic}"
            result = generate_chunk(prompt, max_tokens=150)
            st.success("✅ Generated!")
            st.markdown(result)

        elif task == "Create Video Script":
            total_minutes = get_minutes(script_duration)
            chunk_minutes = 5
            num_chunks = math.ceil(total_minutes / chunk_minutes)

            # Get adaptive prompt style based on feedback
            prompt_style = get_prompt_style()

            full_script = ""
            for i in range(num_chunks):
                start_time = i * chunk_minutes
                end_time = min((i + 1) * chunk_minutes, total_minutes)

                chunk_prompt = (
                    f"Write the script for a YouTube video titled: '{topic}'. "
                    f"Make it {prompt_style}. "
                    f"This is part {i+1} of {num_chunks}, covering minute {int(start_time)} to {int(end_time)}."
                )

                chunk_result = generate_chunk(chunk_prompt, max_tokens=1200)
                full_script += f"\n\n### Part {i+1} ({int(start_time)}–{int(end_time)} mins):\n{chunk_result}"
                time.sleep(1.5)  # Pause between API calls

            st.success("✅ Full Script Generated!")
            st.markdown(full_script)

            # Feedback collection UI
            st.markdown("---")
            st.header("Help NeuraNest Improve!")
            rating = st.slider("Rate this script from 1 (poor) to 5 (excellent):", 1, 5, 3)
            comments = st.text_area("Additional comments (optional):")

            if st.button("Submit Feedback"):
                feedback_entry = {
                    "topic": topic,
                    "script": full_script,
                    "rating": rating,
                    "comments": comments
                }
                save_feedback(feedback_entry)
                st.success("Thanks for your feedback! NeuraNest will learn and improve over time.")

# Analytics dashboard
if task == "Analytics-Based Decision Engine":
    st.header("📊 Analytics-Based Decision Engine")

    data = load_feedback()
    if not data:
        st.info("No feedback data available yet. Generate some scripts and submit feedback first!")
    else:
        summary = summarize_feedback(data)

        st.subheader("Feedback Summary")
        st.write(f"**Total Feedback Entries:** {summary['feedback_count']}")
        st.write(f"**Average Rating:** {summary['average_rating']:.2f} / 5")

        st.subheader("Top 5 Topics by Feedback Count")
        for topic_name, count in summary['top_topics']:
            st.write(f"- {topic_name}: {count} feedback entries")

        st.subheader("Most Common Words in Comments")
        common_words = summary['common_comment_words']
        if common_words:
            words_str = ", ".join([f"{word} ({freq})" for word, freq in common_words])
            st.write(words_str)
        else:
            st.write("No comments submitted yet.")

        # Simple actionable insights based on rating
        if summary['average_rating'] < 3:
            st.warning("Average rating is below 3. Consider simplifying your scripts and making them more beginner-friendly.")
        elif summary['average_rating'] < 4:
            st.info("Average rating is decent. Keep refining content style for better engagement.")
        else:
            st.success("Great average rating! Keep up the engaging content style.")

