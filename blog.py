import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set up YouTube API client
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]  # Add your YouTube API key in .streamlit/secrets.toml
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

SYSTEM_INSTRUCTION = """
You are a skilled blog post writer. Follow these guidelines:
1. Use the exact title provided in the title area.
2. Convert the provided YouTube video transcript into a well-structured blog post.
3. Organize the content logically with appropriate headings and paragraphs.
4. Remove filler words and repetitive content common in spoken language.
5. Maintain the key points and important information from the video.
"""

def generate_article_from_transcript(title, transcript):
    summary_prompt = f"First, summarize the key points from this transcript in 3-4 sentences: {transcript[:1000]}..."
    
    # Get a summary first to help with context
    summary_response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "Summarize the key points from this video transcript."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    summary = summary_response.choices[0].message.content
    
    # Now generate the full article
    content_prompt = f"""Write a detailed blog post with the title '{title}'.
    Here's a summary of the video content: {summary}
    
    Here's the full transcript to reference: {transcript}
    
    Convert this into a well-structured blog post while maintaining the key information and insights from the video."""
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": content_prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    st.title("YouTube Video to Blog Post Generator")
    
    video_url = st.text_input("Enter YouTube Video URL:")
    title = st.text_input("Enter the blog post title:")
    
    if st.button("Generate Article"):
        if video_url and title:
            with st.spinner("Processing video..."):
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL")
                    return
                
                transcript = get_video_transcript(video_id)
                if not transcript:
                    st.error("Could not fetch video transcript")
                    return
                
                article_content = generate_article_from_transcript(title, transcript)
            
            st.subheader("Generated Article:")
            st.markdown(article_content)
        else:
            st.warning("Please enter both a YouTube URL and a title.")

if __name__ == "__main__":
    main()
