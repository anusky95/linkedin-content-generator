import anthropic
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
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---- UTILITY FUNCTIONS ----

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

# ---- TEMPLATE GENERATION FUNCTIONS ----

def generate_template_1(video_content, video_url):
    """Generate Authority + Contradiction template"""
    prompt = f"""
Create a viral LinkedIn post using the Authority + Contradiction approach. 

IMPORTANT: Output clean text that can be copied directly to LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Structure:
- Start with a hook about the topic being positive but having a hidden problem
- Add an empty line
- Add a broader appeal statement
- Establish widespread problem with numbers and expert signposting like "After X years helping Y companies..."
- Introduce the main concept with one-line analogy and brief expert credibility
- Create 3 bullet points with emojis: "• Component: Method → Benefit"
- Include results section: "Real client results I've witnessed:" followed by 3-4 metrics with emojis (⚡💾✨💰) using before→after format
- Add reality check stating one limitation and mitigation strategy  
- End with: "Want the complete deliverable? Expert name (credentials) is revealing specific value including concrete examples. 🎯 Presentation type at TMLS Conference. Register here → {video_url}"
- Close with urgency statement

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean, copy-paste ready text for LinkedIn. No markdown formatting.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_template_2(video_content, video_url):
    """Generate Death + Rebirth template"""
    prompt = f"""
Create a viral LinkedIn post using the Death + Rebirth approach.

IMPORTANT: Output clean text for LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Structure:
- Start with "Old approach is dead. Long live new approach."
- Explain why old approach worked historically but is insufficient today
- Introduce new approach as natural evolution with expert credibility
- Break down system into 4-5 components with emojis (📊🔧🔍🧠📜): "Component: Description"
- Show how new approach solves old approach failures
- Mention one overlooked factor
- End with: "Expert is revealing the complete framework at TMLS... Register → {video_url}"
- Close with: "What's your take on this shift?"

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean, copy-paste ready text for LinkedIn.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_template_3(video_content, video_url):
    """Generate Pain Point + How-To template"""
    prompt = f"""
Create a viral LinkedIn post using the Pain Point + How-To approach.

IMPORTANT: Output clean text for LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Structure:
- Start with "90% of thing fail. Here's why."
- Explain 1-2 root causes with signposting
- Share tool/approach name with one-line benefit and expert mention
- Create numbered how-to guide with 3-4 steps using emojis (🔥⚡✨💎): "1. 🔥 Step Name: Description"
- State one risk/limitation and how to address it
- Add single-sentence takeaway of why this matters
- End with: "Want the advanced implementation? Expert breaks down the complete system at TMLS... Register → {video_url}"
- Close with: "What's been your biggest challenge with topic?"

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean, copy-paste ready text for LinkedIn.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_template_4(video_content, video_url):
    """Generate Impossible Feat template"""
    prompt = f"""
Create a viral LinkedIn post using the Impossible Feat approach.

IMPORTANT: Output clean text for LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Structure:
- Start with "How do you impossible task?" followed by "You can't... not all at once, anyway."
- Introduce real solution that makes it possible
- Explain concept in plain language with expert signposting
- List 3-4 methods with emojis (🔢📝📄🧠): "🔢 Method: Description + pro/con"
- Add advice on when to use which method
- Include truth bomb or memorable analogy
- End with: "Ready for the deep dive? Expert reveals the complete methodology at TMLS... Register → {video_url}"
- Close with: "Which method would you try first?"

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean, copy-paste ready text for LinkedIn.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_template_5(video_content, video_url):
    """Generate Provocative Vision template"""
    prompt = f"""
Create a viral LinkedIn post using the Provocative Vision approach.

IMPORTANT: Output clean text for LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Structure:
- Start with "Overlooked factor undermines even the best system/tool."
- Show why most people overlook this factor with signposting
- Create 5-7 variations: "Variation: Best for: X | Avoid for: Y"
- Provide starting point and iteration approach recommendation
- Tie micro-choice to macro outcomes (ROI, adoption, reliability)
- Discuss how this evolves over next 12-24 months
- End with: "Want to stay ahead of the curve? Expert shares the complete roadmap at TMLS... Register → {video_url}"
- Close with: "What are you doing about this today?"

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean, copy-paste ready text for LinkedIn.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_pinned_comment(video_content):
    """Generate strategic pinned comment"""
    prompt = f"""
Create a strategic pinned comment based on this video content.

IMPORTANT: Output clean text for LinkedIn. NO markdown formatting like **bold** or [brackets]. Use emojis and natural formatting only.

Format should be:
🔥 BONUS: Just for this community - the X biggest mistakes/insights about topic:

1. Specific insight #1
2. Specific insight #2  
3. Specific insight #3

Drop a 🚀 if you want me to break these down!

Plus - anyone attending TMLS can connect with me for a free 15-min consultation.

Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}

Output only clean text - no markdown formatting.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_hashtags(video_content):
    """Generate relevant hashtags"""
    prompt = f"""
Generate 5-8 relevant hashtags for this LinkedIn post about a technical/AI video.

Always include: #AI #MachineLearning #TMLS
Add 2-5 topic-specific hashtags based on the content.

Video Title: {video_content['title']}
Content: {video_content['full_text']}

Output format: #Hashtag1 #Hashtag2 #Hashtag3 etc.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content


def generate_workflow_html(video_content):
    """Generate workflow visualization HTML using Claude"""
    prompt = f"""
You are an expert workflow visualization designer. Create an interactive HTML workflow diagram based on this video content.Create a complete, self-contained HTML workflow diagram. The output must be a COMPLETE HTML document that starts with <!DOCTYPE html> and ends with </html>.Do not wrap in ```html``` or any other formatting.
# Updated Technical Infographic Design Prompt

Create a single-frame technical infographic in the style of professional software architecture diagrams with modern AI/ML visualization aesthetics.

## CANVAS SPECIFICATIONS:
- Size: 1200x1200px (square format for LinkedIn)
- Background: Clean white with subtle gradient overlay OR dark navy/purple gradient (based on content type)
- Grid system: Invisible 12-column layout for precise alignment
- Margins: 60px on all sides for breathing room

## VISUAL STYLE REQUIREMENTS:
- Design aesthetic: Professional software documentation style with modern AI/ML touches
- Color palette options:

### SCHEME A - Professional Tech (Clean):
- Primary: #008B8B (Dark Teal)
- Secondary: #20B2AA (Light Teal/Cyan)
- Accent: #9370DB (Medium Purple)
- Background: #FFFFFF to #F0F8FF gradient
- Text: Dark grays (#1A202C, #4A5568, #718096)

### SCHEME B - Modern AI/ML (Dark):
- Primary: #06B6D4 (Bright Cyan)
- Secondary: #3B82F6 (Blue)
- Accent: #10B981 (Green) or #F59E0B (Orange)
- Background: #0F172A to #1E293B gradient (dark navy)
- Text: Light colors (#FFFFFF, #E2E8F0, #94A3B8)

### SCHEME C - Comparison Contrast:
- Side A: #EF4444 (Red/Orange) for traditional/old methods
- Side B: #06B6D4 (Cyan/Blue) for new/improved methods
- Accent: #9333EA (Purple) for highlights
- Background: Clean white with subtle tints

## TYPOGRAPHY HIERARCHY:
- Main title: 42-48px, 700 weight, gradient text effect
- Section titles: 20-24px, 600 weight
- Process labels: 12-16px, 500 weight
- Technical details: 10-12px, 400 weight
- Code text: 10-12px, monospace font (Monaco, Menlo)
- Annotations: 9-11px, 400 weight

## LAYOUT ARCHITECTURE OPTIONS:

### OPTION 1: Split Comparison Layout (45% | 10% | 45%)
- Left side: Traditional/existing method workflow
- Center: VS comparison with key metrics
- Right side: New/improved method workflow
- Use for: Traditional vs GRPO, Naive RAG vs Graph RAG

### OPTION 2: Process Flow Layout (Top-to-Bottom)
- 3-4 major horizontal sections
- Each section shows sequential steps
- Clear input → processing → output flow
- Use for: GRPO process, MUVERA steps, training pipelines

### OPTION 3: Architecture Diagram Layout
- Layered system components
- Data flow between layers with arrows
- Component responsibilities clearly labeled
- Integration points highlighted
- Use for: Multimodal RAG, Elysia framework, system architectures

### OPTION 4: Step-by-Step Process (4-6 Steps)
- Numbered sections with colored backgrounds
- Each step shows sub-processes
- Progressive revelation of complexity
- Use for: MUVERA explained, training workflows

## COMPONENT SPECIFICATIONS:

### TITLE SECTION:
- Main title: 44px, 700 weight, gradient or solid color
- Subtitle: 16-18px, normal weight, muted color
- Hook statistic: Prominent callout box with key metric
- Accent line: 4-6px colored bar under title

### SECTION CONTAINERS:
- Background: Subtle colored tint (5-10% opacity) or solid dark panels
- Border: 2-3px solid colored border with rounded corners (12-16px)
- Padding: 25-40px internal spacing
- Section numbers: Large (60-80px) or circular badges (40px)
- Hover effects: Subtle lift and shadow enhancement

### WORKFLOW ELEMENTS:
- Process boxes: 120-200px width, 40-80px height
- Box styling: White/dark background, colored border, subtle shadow
- Text: 12-16px, medium weight, high contrast
- Connection arrows: 2-3px stroke, colored, with arrowheads
- Flow indicators: Curved or straight paths showing direction

### CODE BLOCKS (for technical content):
- Background: Dark (#1a1a1a) with colored border
- Font: Monospace (Monaco, Menlo, Consolas)
- Syntax highlighting: Blue keywords, green strings, yellow numbers
- Size: 10-12px for readability
- Comments: Gray color (#9ca3af)

### TECHNICAL ANNOTATIONS:
- Leader lines: Thin colored lines (1-2px)
- Annotation boxes: Small rounded rectangles
- Performance metrics: Highlighted in colored boxes
- Cost/time indicators: Badge-style callouts
- Success rates: Progress bar or percentage displays

## CONTENT STRUCTURE TEMPLATES:

### FOR AI/ML COMPARISON INFOGRAPHICS:
- [MAIN_TITLE]: "Traditional [Method] vs [New Method]"
- [HOOK_STAT]: Cost reduction, time savings, or performance improvement
- [SECTION_1]: Traditional approach workflow (4-6 steps)
- [SECTION_2]: New approach workflow (4-6 steps)
- [METRICS]: Side-by-side performance comparison
- [KEY_INSIGHT]: Bottom summary of main advantage

### FOR PROCESS/ARCHITECTURE INFOGRAPHICS:
- [MAIN_TITLE]: "[Technology/Method] Explained"
- [SUBTITLE]: Brief description of what it does
- [PROCESS_STEPS]: 3-6 numbered stages
- [TECHNICAL_DETAILS]: Code snippets, formulas, or specifications
- [FLOW_ARROWS]: Clear data/process flow indicators
- [PERFORMANCE_NOTES]: Speed, accuracy, efficiency metrics

### FOR SYSTEM ARCHITECTURE:
- [MAIN_TITLE]: "Introducing [System Name]"
- [ARCHITECTURE_LAYERS]: Data → Processing → Output layers
- [COMPONENT_LABELS]: Clear naming of each system component
- [DATA_FLOW]: Arrows showing information movement
- [INTEGRATION_POINTS]: Highlighted connection areas

## SPECIFIC AI/ML VISUAL ELEMENTS:

### Neural Network Style:
- Nodes: Circles connected with lines
- Layers: Grouped node collections
- Activation: Color-coded node states
- Connections: Weighted line thickness

### Data Flow Indicators:
- Vector embeddings: Small dot patterns
- Model training: Circular progress indicators
- Feedback loops: Curved arrows returning to start
- Parallel processing: Multiple parallel paths

### Performance Visualizations:
- Training curves: Simple line charts
- Accuracy metrics: Percentage circles or bars
- Cost comparisons: Dollar amounts in contrasting colors
- Time savings: Clock icons with before/after

## MODERN DESIGN ENHANCEMENTS:

### Interactive Elements:
- Hover states: Subtle transforms (translateY(-3px))
- Color transitions: Smooth border/background changes
- Shadow effects: Multi-layered drop shadows
- Glow effects: Subtle colored glows on key elements

### Advanced Visual Features:
- Glassmorphism: Semi-transparent panels with backdrop blur
- Gradient borders: Multi-color border effects
- Animated connectors: Subtle pulse or flow animations
- Emphasis circles: Thin colored circles highlighting key areas

### Technical Styling:
- Code syntax highlighting: Consistent color coding
- System diagrams: Clean geometric shapes
- Flow charts: Professional connector styles
- Metric displays: Dashboard-style number presentations

## EXPORT REQUIREMENTS:
- High resolution: 2400x2400px for Retina displays
- Sharp text: Vector-based typography
- Social media optimized: Readable at 300x300px mobile size
- Print quality: Professional presentation ready
- Format options: PNG for social, SVG for scalability

## CONTENT ADAPTATION GUIDELINES:

### For DeepSeek/GRPO Content:
- Emphasize code-driven vs data-driven approach
- Show reward function examples in code blocks
- Highlight cost savings ($512 vs $50K)
- Include reasoning emergence visualization

### For RAG System Comparisons:
- Vector database representations
- Query → Retrieval → Generation flow
- Knowledge graph visualizations
- Multi-step retrieval processes

### For Training Pipeline Architecture:
- GPU cluster representations
- Data preprocessing stages
- Model versioning workflows
- Evaluation loop visualizations

## FINAL QUALITY CHECKLIST:
✓ Consistent 8px grid spacing throughout
✓ Readable text at all zoom levels
✓ High contrast ratios (4.5:1 minimum)
✓ Professional color palette (2-3 colors max)
✓ Clear information hierarchy
✓ Actionable insights prominently displayed
✓ Technical accuracy in all diagrams
✓ Modern, cutting-edge aesthetic
✓ LinkedIn-optimized dimensions and quality

## OUTPUT FORMAT:
Generate complete HTML/CSS with inline styles for immediate use, ensuring:
- Pixel-perfect alignment to grid
- Crisp vector-style graphics
- Responsive hover interactions
- Professional documentation quality
- Ready for high-resolution export

## CONTENT-SPECIFIC EXAMPLES:

### GRPO vs Traditional Fine-tuning:
```
Title: "GRPO vs Traditional Fine-tuning: The $512 Revolution"
Layout: Split comparison (45% | 10% | 45%)
Left: Traditional (Data Creation → Training → Iteration)
Right: GRPO (Reward Function → Sampling → Optimization)
Metrics: Cost ($50K vs $512), Time (6 months vs 32 hours), Team (15+ vs 1)
```

### RAG Architecture:
```
Title: "Naive RAG vs Graph RAG: Enhanced Retrieval"
Layout: Process flow with branching
Top: Query processing
Middle: Vector search vs Graph traversal
Bottom: Context generation and response
```

### Training Pipeline:
```
Title: "AI Model Training Architecture"
Layout: Layered system components
Layers: Data → Processing → Training → Evaluation → Deployment
Components: GPU clusters, model stores, evaluation loops
```

## TECHNICAL SPECIFICATIONS:

### CSS Variables for Consistency:
[css]
:root {{
  --primary-teal: #008B8B;
  --secondary-cyan: #20B2AA;
  --accent-purple: #9370DB;
  --text-dark: #1A202C;
  --border-radius: 12px;
  --shadow-subtle: 0 4px 12px rgba(0,0,0,0.1);
  --spacing-unit: 8px;
}}
[/css]

### Grid System:
- 12-column grid with 8px base spacing
- Section widths: multiples of 96px (8px × 12)
- Consistent gaps: 16px, 24px, 32px, 40px
- Responsive breakpoints at 768px and 1024px

### Animation Guidelines:
- Hover transitions: 0.2-0.3s ease
- Transform effects: translateY(-3px) for lift
- Color transitions: smooth border/background changes
- Stagger delays: 0.1s increments for sequential elements

## ACCESSIBILITY REQUIREMENTS:
- Minimum contrast ratio: 4.5:1 for normal text, 3:1 for large text
- Semantic HTML structure with proper headings
- Alt text for all visual elements
- Keyboard navigation support
- Screen reader compatible markup
- Focus indicators for interactive elements

## EXPORT OPTIMIZATION:
- SVG elements for scalable graphics
- Optimized PNG for photographic elements
- WebP format for modern browsers
- Retina display support (2x pixel density)
- Compression optimized for social media platforms
- Print-ready CMYK color profiles available

## QUALITY ASSURANCE CHECKLIST:
✓ Information hierarchy is clear and logical
✓ All technical details are accurate
✓ Color coding is consistent throughout
✓ Typography scales properly at all sizes
✓ Interactive elements provide clear feedback
✓ Loading performance is optimized
✓ Cross-browser compatibility verified
✓ Mobile responsiveness tested
✓ Professional presentation standards met
✓ Brand guidelines compliance (if applicable)

## VERSIONING AND ITERATION:
- Version 1.0: Core structure and content
- Version 1.1: Visual polish and refinements
- Version 1.2: Performance optimizations
- Include version stamp in footer
- Maintain design system documentation
- Track performance metrics and engagement

This comprehensive prompt ensures professional-quality technical infographics that combine cutting-edge design with clear information architecture, suitable for LinkedIn sharing and professional presentations.
Generate complete HTML/CSS with inline styles that creates a STATIC INFOGRAPHIC IMAGE suitable for screenshot/export to LinkedIn. The HTML should render as a single 1200x1200px infographic that captures the entire workflow/process in one comprehensive visual.

use the information below to create, do not output
Video Title: {video_content['title']}
Channel: {video_content['channel']}
Content: {video_content['full_text']}
Design Requirements:
- Professional software architecture diagram style
- Modern AI/ML visualization aesthetics
- Square format (1200x1200px) for LinkedIn
- Clean layout with proper spacing
- Professional color scheme (teals, blues, purples)
- Clear workflow steps with arrows
- Technical details in code blocks where relevant
- Metrics and performance indicators
- Modern gradient backgrounds
- Complete HTML with embedded CSS

Output only the complete HTML code starting with <!DOCTYPE html>
"""
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        html_content = response.content[0].text
        
        # Clean up any markdown formatting that might slip through
        if html_content.startswith('```html'):
            html_content = html_content.replace('```html', '').replace('```', '')
        
        # Ensure it starts with DOCTYPE
        html_content = html_content.strip()
        if not html_content.startswith('<!DOCTYPE html>'):
            # If it doesn't start with DOCTYPE, add it
            if html_content.startswith('<html>'):
                html_content = '<!DOCTYPE html>\n' + html_content
        
        return html_content
        # return response.content[0].text
    except Exception as e:
        return f"Error generating workflow: {str(e)}"

# ---- MAIN GENERATION FUNCTION ----

def generate_all_template_posts(video_content, video_url):
    """Generate LinkedIn posts using all 5 templates for user selection."""
    
    templates = {
        "Template 1: Authority + Contradiction": generate_template_1,
        "Template 2: Death + Rebirth": generate_template_2,
        "Template 3: Pain Point + How-To": generate_template_3,
        "Template 4: Impossible Feat": generate_template_4,
        "Template 5: Provocative Vision": generate_template_5
    }

    # Generate all posts
    all_posts = {}
    
    for template_name, generator_func in templates.items():
        try:
            all_posts[template_name] = generator_func(video_content, video_url)
        except Exception as e:
            all_posts[template_name] = f"Error generating this template: {str(e)}"
    
    # Generate pinned comment and hashtags
    try:
        pinned_comment = generate_pinned_comment(video_content)
        hashtags = generate_hashtags(video_content)
    except Exception as e:
        pinned_comment = f"Error generating pinned comment: {str(e)}"
        hashtags = "#AI #MachineLearning #TMLS"
    
    return all_posts, pinned_comment, hashtags

# ---- UI DISPLAY FUNCTIONS ----

def display_all_linkedin_templates(all_posts, pinned_comment, hashtags):
    """Display all template variations for user selection."""
    st.markdown("### 🎯 Choose Your Favorite LinkedIn Post Template")
    
    # Create tabs for each template
    template_names = list(all_posts.keys())
    tabs = st.tabs([f"📋 {name.split(':')[1].strip()}" for name in template_names])
    
    for i, (template_name, post_content) in enumerate(all_posts.items()):
        with tabs[i]:
            st.markdown(f"#### {template_name}")
            
            # Description of when to use this template
            template_descriptions = {
                "Template 1: Authority + Contradiction": "🎯 **Best for**: Established concepts with hidden problems. Most versatile template (40% usage).",
                "Template 2: Death + Rebirth": "🔄 **Best for**: Industry shifts, paradigm changes, new frameworks (20% usage).",
                "Template 3: Pain Point + How-To": "📝 **Best for**: Educational content, tutorials, step-by-step guides (20% usage).",
                "Template 4: Impossible Feat": "🤔 **Best for**: Simplifying complex problems, breaking down processes (10% usage).",
                "Template 5: Provocative Vision": "🔮 **Best for**: Future trends, thought leadership, big-picture posts (10% usage)."
            }
            
            st.info(template_descriptions.get(template_name, ""))
            
            # Display the post content
            st.text_area(
                f"📱 LinkedIn Post:", 
                post_content, 
                height=400,
                key=f"post_{i}"
            )
            
            # Copy button simulation
            # st.code(post_content, language=None)
            
            # Engagement prediction
            engagement_tips = {
                "Template 1: Authority + Contradiction": "💡 **High engagement potential**: Authority + specific metrics usually perform best",
                "Template 2: Death + Rebirth": "🔥 **Viral potential**: Controversial takes drive comments and shares",
                "Template 3: Pain Point + How-To": "👥 **Community building**: How-to posts generate helpful discussions",
                "Template 4: Impossible Feat": "🤓 **Educational value**: Great for building thought leadership",
                "Template 5: Provocative Vision": "🚀 **Forward-thinking**: Positions you as industry visionary"
            }
            
            st.success(engagement_tips.get(template_name, ""))
    
    # Show pinned comment and hashtags
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Pinned Comment")
        st.text_area("Strategic pinned comment:", pinned_comment, height=200)
        # st.code(pinned_comment, language=None)
    
    with col2:
        st.markdown("### #️⃣ Hashtags")
        st.text_area("Recommended hashtags:", hashtags, height=100)
        # st.code(hashtags, language=None)
    
    # Overall recommendation
    st.markdown("---")
    st.markdown("### 💡 Template Selection Guide")
    st.markdown("""
    **🎯 For maximum impact, rotate between templates:**
    - **Template 1 (Authority + Contradiction)**: Your go-to workhorse - use 40% of the time
    - **Template 2 (Death + Rebirth)**: When discussing industry changes - 20% usage  
    - **Template 3 (Pain Point + How-To)**: For educational content - 20% usage
    - **Template 4 (Impossible Feat)**: For complex simplification - 10% usage
    - **Template 5 (Provocative Vision)**: For thought leadership - 10% usage
    
    **💡 Pro tip**: Never use the same template twice in a row to avoid pattern recognition!
    """)

# ---- MAIN APPLICATION ----

def main():
    """Main Streamlit application."""
    st.title("🎬 Viral LinkedIn Post Generator")
    st.subheader("Transform YouTube videos into irresistible LinkedIn posts")
    
    # User inputs
    video_url = st.text_input("📺 Paste YouTube video link:", placeholder="https://www.youtube.com/watch?v=...")
    
    # Add tabs for different functionality
    tab1, tab2 = st.tabs(["📱 LinkedIn Posts", "🔄 Workflow Diagram"])
    
    with tab1:
        if st.button("🚀 Generate All Template Variations", type="primary") and video_url:
        # Get video content
            video_id, video_content, title, channel = get_youtube_content(video_url)
            
            if video_content:
                # Show video info
                st.success(f"✅ Successfully fetched: **{title}** by **{channel}**")
                
                # Generate all template variations
                with st.spinner("🔥 Generating all 5 template variations... This may take a moment"):
                    all_posts, pinned_comment, hashtags = generate_all_template_posts(video_content, video_url)
                
                # Display all templates
                display_all_linkedin_templates(all_posts, pinned_comment, hashtags)
                
                st.balloons()  # Celebration effect
                st.success("🎉 All templates generated! Pick your favorite and copy to LinkedIn!")
            else:
                st.error("❌ Could not fetch video content. Please check the URL and try again.")
                
    with tab2:
        st.markdown("### 🔄 Interactive Workflow Generator")
        st.markdown("Generate a beautiful, interactive workflow diagram from the video content!")
        
        if st.button("🎨 Generate Workflow Diagram", type="primary") and video_url:
            # Get video content
            video_id, video_content, title, channel = get_youtube_content(video_url)
            
            if video_content:
                # Show video info
                st.success(f"✅ Creating workflow for: **{title}** by **{channel}**")
                
                # Generate workflow HTML
                with st.spinner("🎨 Generating interactive workflow diagram..."):
                    workflow_html = generate_workflow_html(video_content)
                
                if "Error generating" not in workflow_html:
                    st.markdown("### 🎯 Your Interactive Workflow")
                    st.success("✅ Workflow generated! You can copy the HTML code below:")
                    
                    # Display the HTML
                    # st.components.v1.html(workflow_html, height=600, scrolling=True)
                    st.components.v1.html(workflow_html, height=1500, scrolling=True)
                    # Provide download option
                    st.download_button(
                        label="📥 Download HTML File",
                        data=workflow_html.encode('utf-8'),
                        file_name=f"workflow_{video_id}.html",
                        mime="text/html"
                    )
                    
                    # Show code for copying
                    with st.expander("📝 View/Copy HTML Code"):
                        st.code(workflow_html, language='html')
                        
                else:
                    st.error(workflow_html)
    
    # Instructions
    if not video_url:
        st.markdown("""
        ### 📋 How to Use:
        1. **Paste a YouTube URL** in the input above
        2. **Click "Generate All Template Variations"**
        3. **Browse through 5 different post styles** in the tabs
        4. **Copy your favorite** directly to LinkedIn
        5. **Use the pinned comment** for extra engagement
        6. **Add the hashtags** to maximize reach
        
        ### 🎯 What You'll Get:
        - **5 unique LinkedIn post variations** using proven viral templates
        - **Strategic pinned comment** with bonus content
        - **Optimized hashtags** for maximum reach
        - **Engagement tips** for each template style
        
        #### 🎨 Workflow Diagram Tab:
        - **Professional workflow diagrams** (1200x1200px LinkedIn-optimized)
        - **Technical process visualizations** perfect for sharing
        - **Modern AI/ML aesthetic** with clean, professional design
        - **Screenshot-ready infographics** that explain complex concepts visually
        - **Perfect for technical content creators** and conference speakers
        """)

# ---- RUN APPLICATION ----
if __name__ == "__main__":
    main()
    