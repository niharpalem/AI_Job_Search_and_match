import streamlit as st
import groq
from jobspy import scrape_jobs
import pandas as pd
import json
from typing import List, Dict
import numpy as np
import time

def make_clickable(url: str) -> str:
    """
    Convert a URL to a clickable HTML link.
    
    Args:
        url (str): The URL to make clickable
    
    Returns:
        str: HTML anchor tag with the URL
    """
    return f'<a href="{url}" target="_blank" style="color: #4e79a7;">Link</a>'

def convert_prompt_to_parameters(client, prompt: str) -> Dict[str, str]:
    """
    Convert user input prompt to structured job search parameters using AI.
    
    Args:
        client: Groq AI client
        prompt (str): User's job search description
    
    Returns:
        Dict[str, str]: Extracted search parameters with search_term and location
    """
    system_prompt = """
    You are a language decoder. Extract:
    - search_term: job role/keywords (expand abbreviations)
    - location: mentioned place or 'USA'
    Return only: {"search_term": "term", "location": "location"}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract from: {prompt}"}
        ],
        max_tokens=1024,
        model='llama3-70b-8192',
        temperature=0.2
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"search_term": prompt, "location": "USA"}

def analyze_resume(client, resume: str) -> str:
    """
    Generate a comprehensive resume analysis using AI.
    
    Args:
        client: Groq AI client
        resume (str): Full resume text
    
    Returns:
        str: Concise professional overview of the resume
    """
    system_prompt = """Analyze resume comprehensively in 150 words:
    1. Professional Profile Summary
    2. Key Technical Skills
    3. Educational Background
    4. Core Professional Experience Highlights
    5. Unique Strengths/Achievements
    Return a concise, structured professional overview."""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resume}
        ],
        max_tokens=400,
        model='llama3-70b-8192',
        temperature=0.3
    )
    
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def get_job_data(search_params: Dict[str, str]) -> pd.DataFrame:
    """
    Fetch job listings from multiple sources based on search parameters.
    
    Args:
        search_params (Dict[str, str]): Search parameters including term and location
    
    Returns:
        pd.DataFrame: Scraped job listings
    """
    try:
        return scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter"],
            search_term=search_params["search_term"],
            location=search_params["location"],
            results_wanted=60,
            hours_old=24,
            country_indeed='USA'
        )
    except Exception as e:
        st.warning(f"Error in job scraping: {str(e)}")
        return pd.DataFrame()

def analyze_job_batch(
    client, 
    resume: str, 
    jobs_batch: List[Dict], 
    start_index: int, 
    retry_count: int = 0
) -> pd.DataFrame:
    """
    Analyze a batch of jobs against the resume with retry logic.
    
    Args:
        client: Groq AI client
        resume (str): Resume text
        jobs_batch (List[Dict]): Batch of job listings
        start_index (int): Starting index of the batch
        retry_count (int, optional): Number of retry attempts. Defaults to 0.
    
    Returns:
        pd.DataFrame: Job match analysis results
    """
    if retry_count >= 3:
        return pd.DataFrame()
        
    system_prompt = """Rate resume-job matches. Return only JSON array:
[{"job_index": number, "match_score": 0-100, "reason": "brief reason"}]"""
    
    jobs_info = [
        {
            'index': idx + start_index,
            'title': job['title'],
            'desc': job.get('description', '')[:400],
        }
        for idx, job in enumerate(jobs_batch)
    ]
    
    resume_summary = analyze_resume(client, resume)
    
    analysis_prompt = f"Resume: {resume_summary}\nJobs: {json.dumps(jobs_info)}"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=1024,
            model='llama3-70b-8192',
            temperature=0.3
        )
        
        matches = json.loads(response.choices[0].message.content)
        return pd.DataFrame(matches)
    except Exception as e:
        if retry_count < 3:
            time.sleep(2)
            return analyze_job_batch(client, resume, jobs_batch, start_index, retry_count + 1)
        st.warning(f"Batch {start_index} failed after retries: {str(e)}")
        return pd.DataFrame()

def analyze_jobs_in_batches(
    client, 
    resume: str, 
    jobs_df: pd.DataFrame, 
    batch_size: int = 3
) -> pd.DataFrame:
    """
    Process job listings in batches and analyze match with resume.
    
    Args:
        client: Groq AI client
        resume (str): Resume text
        jobs_df (pd.DataFrame): DataFrame of job listings
        batch_size (int, optional): Number of jobs to process in each batch. Defaults to 3.
    
    Returns:
        pd.DataFrame: Sorted job matches by match score
    """
    all_matches = []
    jobs_dict = jobs_df.to_dict('records')
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(jobs_dict), batch_size):
        batch = jobs_dict[i:i + batch_size]
        status_text.text(f"Processing batch {i//batch_size + 1} of {len(jobs_dict)//batch_size + 1}")
        
        batch_matches = analyze_job_batch(client, resume, batch, i)
        if not batch_matches.empty:
            all_matches.append(batch_matches)
            
        progress = min((i + batch_size) / len(jobs_dict), 1.0)
        progress_bar.progress(progress)
        time.sleep(1)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
        return final_matches.sort_values('match_score', ascending=False)
    return pd.DataFrame()

def main():
    """
    Main Streamlit application entry point for Smart Job Search.
    Handles user interface, job search, and AI-powered job matching.
    """
    st.set_page_config(
        layout="wide",
        page_title="Smart Job Search with AI Matching",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS with reduced text sizes
    st.markdown("""
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1200px;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.375rem;
            padding: 0.75rem 1.5rem;
            border: none;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            margin: 0.5rem;
            min-width: 200px;
            font-size: 0.875rem;
        }
        [data-testid="stFileUploader"] {
            border: 2px dashed #e5e7eb;
            border-radius: 0.5rem;
            padding: 0.875rem;
            min-height: 220px;
            font-size: 0.875rem;
        }
        .stTextArea>div>div {
            border-radius: 0.5rem;
            min-height: 220px !important;
            font-size: 0.875rem;
        }
        .stTextInput>div>div>input {
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }
        .resume-html {
            padding: 1.5rem;
            max-width: 800px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }
        h1 {font-size: 3rem !important;  /* Adjust this value to increase the font size */
        } h2 {font-size: 1.5rem !important;  /* Adjust this value to increase the font size */
        h3, h4, h5, h6 {
            font-size: 80% !important;
        }
        p, li {
            font-size: 0.875rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with smaller text
    st.markdown("""
        <h1 style='text-align: center; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.875rem;'>
        🚀 Smart Job Search with AI Matching 
        </h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        user_input = st.text_area(
            "Describe the job you're looking for",
            placeholder="E.g., 'Senior Python developer with React experience in San Francisco'",
            height=150
        )
    
    with col2:
        user_resume = st.text_area(
            "Paste your resume here (for AI-powered matching)",
            placeholder="Paste your resume for AI-powered job matching",
            height=150
        )
        
    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        help="Your API key will be used to process the job search query"
    )

   # Add this CSS styling right after st.set_page_config()
    

    if st.button("🔍 Search Jobs", disabled=not api_key):
        st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Modify tab creation to use descriptive names
        tab1, tab2, tab3 = st.tabs([
        "🔍 Job Listings", 
        "📄 Resume Summary", 
        "🤖 AI Job Matching"
    ])
        if user_input and api_key:
            try:
                client = groq.Client(api_key=api_key)
                
                with st.spinner("Processing search parameters..."):
                    processed_params = convert_prompt_to_parameters(client, user_input)
                
                with st.spinner("Searching for jobs..."):
                    jobs_data = get_job_data(processed_params)
                    
                    if not jobs_data.empty:
                        data = pd.DataFrame(jobs_data)
                        data = data[data['description'].notna()].reset_index(drop=True)
                        
                        with tab1:
                            st.success(f"Found {len(data)} jobs!")
                            display_df = data[['site', 'job_url', 'title', 'company', 'location', 'job_type', 'date_posted']]
                            display_df['job_url'] = display_df['job_url'].apply(make_clickable)
                            st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
                        
                        if user_resume:
                            with tab2:
                                st.info("Analyzing resume summary...")
                                resume_summary = analyze_resume(client, user_resume)
                                st.success("Resume summary:")
                                st.write(resume_summary)
                            
                            with tab3:
                                st.info("Analyzing job matches in small batches...")
                                matches_df = analyze_jobs_in_batches(client, resume_summary, data, batch_size=3)
                                
                                if not matches_df.empty:
                                    matched_jobs = data.iloc[matches_df['job_index']].copy()
                                    matched_jobs['match_score'] = matches_df['match_score']
                                    matched_jobs['match_reason'] = matches_df['reason']
                                    
                                    st.success(f"Found {len(matched_jobs)} recommended matches!")
                                    display_cols = ['site', 'job_url', 'title', 'company', 'location', 'match_score', 'match_reason']
                                    display_df = matched_jobs[display_cols].sort_values('match_score', ascending=False)
                                    display_df['job_url'] = display_df['job_url'].apply(make_clickable)
                                    st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
                                else:
                                    st.warning("Could not process job matches. Please try again.")
                    else:
                        st.warning("No jobs found with the given parameters.")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif not api_key:
            st.warning("Please enter your API key.")
        else:
            st.warning("Please enter a job description.")

if __name__ == "__main__":
    main()