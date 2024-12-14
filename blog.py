import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import os
import whisper
import time

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set up YouTube API client
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
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

def transcribe_audio(video_id):
    """Download and transcribe video audio using Whisper"""
    try:
        # Get video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Configure pytube with custom parameters
        yt = YouTube(video_url)
        
        # Add retry logic for audio download
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get the audio stream with lowest bitrate to speed up download
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
                
                if not audio_stream:
                    raise Exception("No audio stream available")
                
                # Use a unique temporary filename
                temp_file = f"temp_audio_{video_id}_{int(time.time())}.mp4"
                
                # Download with timeout
                audio_stream.download(filename=temp_file, timeout=30)
                
                # Check if file exists and has size
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    raise Exception("Download failed or empty file")
                
                # Load Whisper model and transcribe
                model = whisper.load_model("base")
                result = model.transcribe(temp_file)
                
                # Clean up
                os.remove(temp_file)
                return result["text"]
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        # Ensure cleanup even if error occurs
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)

def get_video_transcript(video_id):
    """Get transcript from YouTube video with fallback to audio transcription"""
    try:
        # First try getting transcript through YouTube API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If no English transcript, get the first available and translate it
            transcript = transcript_list.find_transcript(['en'])
            transcript = transcript.translate('en')
        
        transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
        return transcript_text
        
    except Exception as e:
        st.info("No transcript available. Attempting to transcribe audio...")
        # Try the fallback method with additional error context
        transcript = transcribe_audio(video_id)
        if not transcript:
            st.error("""
            Unable to process video. This could be due to:
            - Video is private or restricted
            - No available transcripts
            - Audio transcription failed
            
            Please try another video or ensure the video is public.
            """)
        return transcript

def get_video_details(video_id):
    """Get video title and description from YouTube API"""
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            snippet = response['items'][0]['snippet']
            return {
                'title': snippet['title'],
                'description': snippet['description']
            }
        return None
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None

SYSTEM_INSTRUCTION = """
You are a skilled blog post writer. Follow these guidelines:
1. Use the exact title provided in the title area.
2. Convert the provided YouTube video transcript into a well-structured blog post.
3. Organize the content logically with appropriate headings and paragraphs.
4. Remove filler words and repetitive content common in spoken language.
5. Maintain the key points and important information from the video.
6. Use markdown formatting for headers (##) and emphasis (*).
7. Include a brief introduction and conclusion.
8. Break down complex topics into digestible sections.
"""

def generate_article_from_transcript(title, transcript, video_details=None):
    """Generate a blog post from the transcript and video details"""
    
    # Create a context-rich prompt using video details if available
    context = ""
    if video_details:
        context = f"""
        Video Title: {video_details['title']}
        Video Description: {video_details['description']}
        """
    
    summary_prompt = f"""First, summarize the key points from this transcript and context:
    {context}
    Transcript excerpt: {transcript[:1500]}..."""
    
    # Get a summary first to help with context
    summary_response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "Summarize the key points from this video transcript and context."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.7
    )
    summary = summary_response.choices[0].message.content
    
    # Now generate the full article
    content_prompt = f"""Write a detailed blog post with the title: '{title}'
    
    Context: {context}
    
    Summary of content: {summary}
    
    Full transcript: {transcript}
    
    Convert this into a well-structured blog post while maintaining the key information and insights from the video.
    Use proper markdown formatting and create a engaging, easy-to-read article."""
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": content_prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="YouTube to Blog Post Generator", page_icon="📝", layout="wide")
    
    st.title("📝 YouTube Video to Blog Post Generator")
    st.markdown("""
    Transform any YouTube video into a well-structured blog post. 
    Simply enter the video URL and desired blog title below.
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        video_url = st.text_input("YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        title = st.text_input("Blog Post Title:", placeholder="Enter your desired blog post title...")
    
    if st.button("Generate Article", type="primary"):
        if video_url and title:
            with st.spinner("Processing video... This may take a few minutes."):
                # Extract video ID and get details
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please check the URL and try again.")
                    return
                
                # Get video details
                video_details = get_video_details(video_id)
                
                # Get transcript
                transcript = get_video_transcript(video_id)
                if not transcript:
                    return
                
                # Generate article
                article_content = generate_article_from_transcript(title, transcript, video_details)
            
            # Display results
            st.success("✅ Article generated successfully!")
            
            # Show the article in a nice format
            st.markdown("---")
            st.markdown("## Generated Article")
            st.markdown(article_content)
            
            # Add download button
            st.download_button(
                label="Download Article as Markdown",
                data=article_content,
                file_name="generated_article.md",
                mime="text/markdown"
            )
        else:
            st.warning("Please enter both a YouTube URL and a title.")
    
    # Add footer with usage instructions
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Paste a YouTube video URL in the input field
    2. Enter your desired blog post title
    3. Click "Generate Article" and wait for processing
    4. Download the generated article in Markdown format
    
    Note: Processing time may vary depending on video length and transcript availability.
    """)

if __name__ == "__main__":
    main()
