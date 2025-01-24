import streamlit as st
import groq
from jobspy import scrape_jobs
import pandas as pd
import json
from typing import List, Dict
import numpy as np
import time
from flowchart import create_job_search_architecture

def make_clickable(url):
    return f'<a href="{url}" target="_blank" style="color: #4e79a7;">Link</a>'

def convert_prompt_to_parameters(client, prompt):
    system_prompt = """
    You are a language decoder. From the given prompt, extract the following information:
    - search_term: the job role or keywords mentioned in the prompt 
    if the search term is in short form like DA expand it like Data analyst etc
    - location: if any place is mentioned in the prompt, otherwise return 'USA'
    if the location is in short abbreviate it like say CA to california etc
    
    Return the response in this exact format (just the dictionary, no other text):
    {"search_term": "extracted term", "location": "extracted location"}
    """
    
    user_prompt = f"Analyze this text: {prompt}"
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        model='llama-3.3-70b-versatile',
        temperature=0.2
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"search_term": prompt, "location": "USA"}

def analyze_resume(client, resume: str) -> str:
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
        model='llama-3.3-70b-versatile',
        temperature=0.3
    )
    
    return response.choices[0].message.content

def get_job_data(search_params):
    try:
        return scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
            google_search_term=search_params["search_term"],
            search_term=search_params["search_term"],
            location=search_params["location"],
            results_wanted=50,
            hours_old=24,
            country_indeed='USA'
        )
    except Exception:
        return scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
            search_term=search_params["search_term"],
            location=search_params["location"],
            results_wanted=50,
            hours_old=24,
            country_indeed='USA'
        )

def analyze_job_batch(client, resume: str, jobs_batch: List[Dict], start_index: int, resume_summary: str = None) -> pd.DataFrame:
    if not jobs_batch or not resume:
        return pd.DataFrame()
    
    resume_summary = analyze_resume(client, resume)
    
    system_prompt = """Analyze job-resume compatibility PRECISELY. 
    Return a list of job matches with these keys:
    - job_index (integer)
    - match_score (integer 0-100)
    - short_reason (string, max 5 words explaining match)
    - key_changes (string, specific skills/changes making candidate a good fit)
    - title (string)
    - company (string)"""
    
    jobs_info = [
        {
            'index': idx + start_index,
            'title': str(job.get('title', 'Untitled'))[:50],
            'company': str(job.get('company', 'Unknown'))[:50],
            'description': str(job.get('description', ''))[:300]
        }
        for idx, job in enumerate(jobs_batch)
    ]
    
    def flexible_json_parse(content: str) -> List[Dict]:
        parsing_attempts = [
            lambda: json.loads(content),
            lambda: json.loads(content.strip()),
            lambda: json.loads(content.replace("'", '"')),
            lambda: json.loads(content[content.find('['):content.rfind(']')+1]),
            lambda: [
                {
                    'job_index': int(match.get('job_index', 0)),
                    'match_score': int(match.get('match_score', 50)),
                    'short_reason': str(match.get('short_reason', 'Good potential match')),
                    'key_changes': str(match.get('key_changes', 'Relevant skills align')),
                    'title': str(match.get('title', 'Unknown Title')),
                    'company': str(match.get('company', 'Unknown Company'))
                }
                for match in (json.loads(match_str) if match_str else {} 
                               for match_str in content.split('},') if match_str.strip())
            ]
        ]
        
        for attempt in parsing_attempts:
            try:
                result = attempt()
                if (isinstance(result, list) and 
                    all('job_index' in item and 'match_score' in item for item in result)):
                    return result
            except Exception:
                continue
        
        return []
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resume Summary: {resume_summary}\nJobs: {json.dumps(jobs_info)}"}
            ],
            max_tokens=512,
            model='llama-3.3-70b-versatile',
            temperature=0.2
        )
        
        response_content = response.choices[0].message.content
        
        matches = flexible_json_parse(response_content)
        
        if matches:
            df = pd.DataFrame(matches)
            
            required_columns = ['job_index', 'match_score', 'short_reason', 'key_changes', 'title', 'company']
            df = df.reindex(columns=required_columns)
            df['match_score'] = pd.to_numeric(df['match_score'], errors='coerce').fillna(50)
            df['job_index'] = pd.to_numeric(df['job_index'], errors='coerce')
            
            return df
        
        return pd.DataFrame()
    
    except Exception as e:
        st.warning(f"Batch analysis error: {str(e)}")
        return pd.DataFrame()

def analyze_jobs_in_batches(client, resume: str, jobs_df: pd.DataFrame, batch_size: int = 3) -> pd.DataFrame:
    jobs_df = jobs_df.head(50)
    if jobs_df.empty or not resume:
        st.warning("No jobs or resume provided.")
        return pd.DataFrame()
    
    resume_summary = analyze_resume(client, resume)
    jobs_dict = jobs_df.to_dict('records')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_matches = []
    
    for i in range(0, len(jobs_dict), batch_size):
        batch = jobs_dict[i:i + batch_size]
        
        status_text.text(f"Processing batch {i//batch_size + 1} of {len(jobs_dict)//batch_size + 1}")
        
        batch_matches = analyze_job_batch(
            client, 
            resume, 
            batch, 
            i, 
            resume_summary
        )
        
        if not batch_matches.empty:
            all_matches.append(batch_matches)
        
        progress = min((i + batch_size) / len(jobs_dict), 1.0)
        progress_bar.progress(progress)
        
        time.sleep(2)
    
    progress_bar.empty()
    status_text.empty()
    
    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
        return final_matches.sort_values('match_score', ascending=False)
    
    st.warning("No job matches found.")
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="Advanced Job Search", layout="wide")
    
    st.title("🔍 Advanced Job Search with AI Matching")
    st.markdown("Intelligent job search powered by resume analysis")
     # Centered flowchart toggle
    col_toggle, _, _ = st.columns([1, 1, 1])
    with col_toggle:
        show_flowchart = st.toggle('Show System Architecture', False)
    
    if show_flowchart:
        diagram = create_job_search_architecture()
        st.graphviz_chart(diagram)
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_input = st.text_area(
            "Job Search Description",
            placeholder="E.g., 'Data Analyst in San Francisco'",
            height=150
        )
    
    with col2:
        user_resume = st.text_area(
            "Upload Resume",
            placeholder="Paste your complete resume for AI analysis",
            height=150
        )
    
    api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        help="Required for AI-powered job search"
    )

    if st.button("🚀 Start Advanced Search", disabled=not (user_input and api_key)):
        tab1, tab2, tab3 = st.tabs(["Job Search", "Resume Analysis", "Match Insights"])
        
        try:
            client = groq.Client(api_key=api_key)
            
            with st.spinner("Processing search parameters..."):
                search_params = convert_prompt_to_parameters(client, user_input)
            
            with st.spinner("Searching jobs..."):
                jobs_data = get_job_data(search_params)
                
                if not jobs_data.empty:
                    with tab1:
                        st.success(f"Found {len(jobs_data)} matching jobs!")
                        display_df = jobs_data[['site', 'job_url', 'title', 'company', 'location']]
                        display_df['job_url'] = display_df['job_url'].apply(make_clickable)
                        display_df=display_df.head(50)
                        st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
                    
                    if user_resume:
                        with tab2:
                            resume_summary = analyze_resume(client, user_resume)
                            st.markdown("### Resume Insights")
                            st.info(resume_summary)
                        
                        with tab3:
                            st.markdown("### Job Match Analysis")
                            matches = analyze_jobs_in_batches(
                                client, 
                                user_resume, 
                                jobs_data
                            )
                            
                            if not matches.empty:
                                st.dataframe(matches)
                            else:
                                st.warning("No matching jobs found.")
                
                else:
                    st.warning("No jobs found matching your search.")
        
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

if __name__ == "__main__":
    main()