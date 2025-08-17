import streamlit as st
from googleapiclient.discovery import build
import openai
import numpy as np
import json
import re
import os
from openai import OpenAI
from urllib.parse import urlparse, parse_qs

# ---- CONFIGURATION ----
from dotenv import load_dotenv
load_dotenv()  # Load from .env file if exists

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print(client)

# ---- UTILITY FUNCTIONS ----

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_youtube_id(url):
    """Extract the video ID from any YouTube URL."""
    # Parse the URL and get the query parameters
    query = urlparse(url).query
    params = parse_qs(query)
    
    # 'v' is the video id parameter
    video_id = params.get('v')
    if video_id:
        return video_id[0]
    
    # Fallback for youtu.be short links
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    
    raise ValueError("Could not extract video ID from URL!")

def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def analyze_chunk_sizes(video_id):
    """Analyze the current chunk sizes in the embeddings file."""
    try:
        with open(f'embeddings_{video_id}.json', 'r') as f:
            chunks = json.load(f)
        
        word_counts = [len(chunk['text'].split()) for chunk in chunks]
        durations = [chunk['end'] - chunk['start'] for chunk in chunks]
        
        st.write(f"**📊 Current Chunk Analysis:**")
        st.write(f"- **Total chunks**: {len(chunks)}")
        st.write(f"- **Avg words per chunk**: {np.mean(word_counts):.1f}")
        st.write(f"- **Avg duration per chunk**: {np.mean(durations):.1f} seconds")
        st.write(f"- **Min/Max words**: {min(word_counts)} - {max(word_counts)}")
        
        # Show sample chunks
        st.write("**🔍 Sample chunks:**")
        for i, chunk in enumerate(chunks[:3]):
            st.write(f"{i+1}. `[{chunk['start']:.1f}s]` \"{chunk['text'][:100]}...\" ({len(chunk['text'].split())} words)")
            
    except Exception as e:
        st.error(f"Could not analyze chunks: {e}")

# ---- EMBEDDING FUNCTIONS ----

def get_embedding(text):
    """Get embedding for a given text using OpenAI's embedding model."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def load_embedded_chunks(video_id):
    """Load pre-computed embeddings from file."""
    # Try larger chunks first, fallback to original
    try:
        with open(f'embeddings_{video_id}_large.json', 'r') as f:
            return json.load(f)
    except:
        try:
            with open(f'embeddings_{video_id}.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Embedding file not found: {e}")
            st.stop()

def create_and_save_embeddings(video_id, video_content, chunk_duration=30):
    """Create larger chunks and generate embeddings from video content."""
    # Since we have video description instead of transcript, create chunks differently
    larger_chunks = create_content_chunks(video_content, chunk_duration)
    
    # Generate embeddings for each chunk
    embedded_chunks = []
    for chunk in larger_chunks:
        emb = get_embedding(chunk["text"])
        embedded_chunks.append({
            "embedding": emb,
            "start": chunk["start"],
            "end": chunk["end"], 
            "text": chunk["text"]
        })
    
    # Save to file
    filename = f'embeddings_{video_id}_large.json'
    with open(filename, 'w') as f:
        json.dump(embedded_chunks, f)
    
    return embedded_chunks

# ---- YOUTUBE API FUNCTIONS ----

def get_youtube_content(youtube_url):
    """Get YouTube video title and description using YouTube Data API"""
    try:
        # Extract video ID
        video_id = get_youtube_id(youtube_url)
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None, None, None, None
        
        # Build YouTube API client
        youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
        
        # Get video details
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            st.error("Video not found or is private")
            return None, None, None, None
        
        # Extract data
        snippet = video_response['items'][0]['snippet']
        stats = video_response['items'][0].get('statistics', {})
        
        title = snippet['title']
        description = snippet['description']
        channel = snippet['channelTitle']
        published = snippet['publishedAt']
        view_count = stats.get('viewCount', 'N/A')
        
        # Create structured content
        video_content = {
            'title': title,
            'description': description,
            'channel': channel,
            'published': published,
            'view_count': view_count,
            'full_text': f"Title: {title}\n\nChannel: {channel}\n\nDescription: {description}"
        }
        
        return video_id, video_content, title, channel
        
    except Exception as e:
        st.error(f"Error fetching YouTube data: {str(e)}")
        return None, None, None, None

# ---- CONTENT PROCESSING FUNCTIONS ----

def create_content_chunks(video_content, chunk_duration=30):
    """Create chunks from video content (title + description)."""
    full_text = video_content['full_text']
    
    # Split description into sentences for better chunking
    sentences = re.split(r'[.!?]+', full_text)
    
    chunks = []
    current_chunk = {"start": 0, "end": 0, "text": ""}
    chunk_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        current_chunk["text"] += " " + sentence
        current_chunk["end"] = chunk_count * chunk_duration + chunk_duration
        
        # Create chunk when we reach reasonable length
        if len(current_chunk["text"].split()) >= 50 or i == len(sentences) - 1:
            if current_chunk["text"].strip():
                chunks.append({
                    "start": current_chunk["start"],
                    "end": current_chunk["end"], 
                    "text": current_chunk["text"].strip()
                })
                chunk_count += 1
                current_chunk = {
                    "start": chunk_count * chunk_duration, 
                    "end": 0, 
                    "text": ""
                }
    
    return chunks

# ---- SIMILARITY SEARCH FUNCTIONS ----

def perform_similarity_search(question, embedded_chunks, top_k=3):
    """Perform similarity search to find most relevant chunks."""
    # Embed the question
    user_emb = get_embedding(question)
    
    # Calculate similarities
    similarities = []
    for chunk in embedded_chunks:
        sim = cosine_similarity(user_emb, chunk['embedding'])
        similarities.append((sim, chunk))
    
    # Sort by similarity (highest first) and return top_k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def create_context_from_chunks(top_chunks):
    """Create context string from top similarity chunks."""
    context_parts = []
    for _, chunk in top_chunks:
        timestamp_range = f"[Section {chunk['start']:.0f}-{chunk['end']:.0f}]"
        context_parts.append(f"{timestamp_range} {chunk['text']}")
    return "\n".join(context_parts)

# ---- LLM FUNCTIONS ----

def generate_answer(question, context):
    """Generate answer using LLM with the given context."""
    llm_prompt = f"""
You are an expert assistant. Answer the question using ONLY the following context (video content).

Question: {question}

Context:
{context}

Format it in an easy to understand manner
Answer:
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Fixed model name
        messages=[{"role": "user", "content": llm_prompt}]
    )
    return response.choices[0].message.content

def generate_social_media_post(video_content, video_url):
    """Generate a LinkedIn-style social media post from the video content."""
    
    full_text = video_content['full_text']
    
    social_media_prompt = f"""
Given the following video information, write a LinkedIn post in this style:
---
What if your next model training loop could run 100B-parameter LLMs without wrestling separate stacks for training and inference, and without vendor lock-in? 
This is what we covered during this session.
Hope you find this helpful!
See vid and blog post here:
➡️ [LINK_HERE]
MLOps World | GenAI Summit
Presentation Highlights
• [bullet 1]
• [bullet 2]
• [bullet 3]
...
---
Please extract the main problem or question the video addresses and phrase it as a bold, open-ended hook in the first line. Then summarize the main themes or solutions discussed in 5–6 short bullet points (Presentation Highlights), written for LinkedIn. 
At the end, include 'See vid and blog post here: ➡️ {video_url}' and add 'MLOps World | GenAI Summit'.

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Video Content:
{full_text}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": social_media_prompt}]
    )
    return response.choices[0].message.content

def generate_detailed_summary(video_content, video_url):
    """Generate a comprehensive detailed summary from the video content."""
    
    full_text = video_content['full_text']
    
    detailed_summary_prompt = f"""
Create a comprehensive summary of this video following this structure:

## Summary
Write a detailed 3-4 paragraph overview that covers:
- The presenter/channel: {video_content['channel']}
- The main topic: {video_content['title']}
- Key themes and approaches discussed based on the description
- Potential business applications and value
- Conclusion and takeaways for the audience

## Highlights
Create 7-10 bullet points with emojis that capture the most important takeaways from the video description:
🚀 [Key insight or technique]
🎯 [Important concept or method]
🐍 [Technical detail or tool]
etc.

## Key Insights
Write 5-7 detailed insights with descriptive headings and emoji, each containing:
- A bold heading with emoji that captures the core concept
- 2-3 sentences explaining the insight in depth
- Why this matters for businesses/practitioners
- How it relates to broader trends or challenges

## Conclusion
Write a thoughtful 2-3 sentence conclusion that ties everything together and emphasizes the practical value for viewers.

Please make the summary comprehensive, professional, and valuable for someone who wants to understand the core concepts.

Video Information:
Title: {video_content['title']}
Channel: {video_content['channel']}
Published: {video_content['published']}
Views: {video_content['view_count']}

Content:
{full_text}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": detailed_summary_prompt}]
    )
    return response.choices[0].message.content

# ---- UI DISPLAY FUNCTIONS ----

def display_answer(answer):
    """Display the generated answer."""
    st.markdown("### 🧑‍💻 Answer")
    st.write(answer)

def display_social_media_post(post_content):
    """Display the generated social media post."""
    st.markdown("### 📱 LinkedIn Post")
    st.text_area("Generated LinkedIn Post:", post_content, height=300)
    st.info("Copy the text above to share on LinkedIn!")

def display_detailed_summary(summary_content):
    """Display the generated detailed summary."""
    st.markdown("### 📝 Detailed Summary")
    st.markdown(summary_content)
    st.info("This comprehensive summary covers all key points from the video!")

def display_source_timestamps(top_chunks):
    """Display source sections."""
    st.markdown("### ⏰ Source Sections")
    for _, chunk in top_chunks:
        st.write(f"Section — {chunk['text'][:200]}...")
    st.info("These are the most relevant sections from the video content!")

# ---- MAIN APPLICATION ----

def main():
    """Main Streamlit application."""
    st.title("🎬 YouTube RAG - TMLS Video Marketing Tool")
    
    # User inputs
    video_url = st.text_input("Paste YouTube video link:", "")
    
    # Add tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["❓ Q&A", "📱 Social Media Post", "📝 Detailed Summary"])
    
    with tab1:
        question = st.text_input("Ask a question about the video:", "")
        
        # Add chunk analysis button
        if st.button("🔍 Analyze Current Chunks") and video_url:
            video_id, video_content, title, channel = get_youtube_content(video_url)
            if video_id:
                analyze_chunk_sizes(video_id)
        
        # Add regenerate embeddings button
        if st.button("🔄 Generate Content Chunks") and video_url:
            video_id, video_content, title, channel = get_youtube_content(video_url)
            if video_content:
                with st.spinner("Creating content chunks and embeddings..."):
                    embedded_chunks = create_and_save_embeddings(video_id, video_content, chunk_duration=30)
                st.success(f"✅ Created {len(embedded_chunks)} content chunks! Now try asking your question again.")
        
        if st.button("Get Answer") and video_url and question:
            # Get video content
            video_id, video_content, title, channel = get_youtube_content(video_url)
            
            if video_content:
                # Load pre-computed embeddings or create new ones
                with st.spinner("Loading embeddings..."):
                    try:
                        embedded_chunks = load_embedded_chunks(video_id)
                    except:
                        st.info("Creating new embeddings from video content...")
                        embedded_chunks = create_and_save_embeddings(video_id, video_content)
                
                # Perform similarity search
                with st.spinner("Searching for relevant content..."):
                    top_chunks = perform_similarity_search(question, embedded_chunks)
                    context = create_context_from_chunks(top_chunks)
                
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, context)
                
                # Display results
                display_answer(answer)
                display_source_timestamps(top_chunks)
    
    with tab2:
        if st.button("Generate LinkedIn Post") and video_url:
            # Get video content
            video_id, video_content, title, channel = get_youtube_content(video_url)
            
            if video_content:
                # Generate social media post
                with st.spinner("Generating LinkedIn post..."):
                    social_post = generate_social_media_post(video_content, video_url)
                
                # Display social media post
                display_social_media_post(social_post)
    
    with tab3:
        if st.button("Generate Detailed Summary") and video_url:
            # Get video content
            video_id, video_content, title, channel = get_youtube_content(video_url)
            
            if video_content:
                # Generate detailed summary
                with st.spinner("Generating comprehensive summary..."):
                    detailed_summary = generate_detailed_summary(video_content, video_url)
                
                # Display detailed summary
                display_detailed_summary(detailed_summary)

# ---- RUN APPLICATION ----
if __name__ == "__main__":
    main()