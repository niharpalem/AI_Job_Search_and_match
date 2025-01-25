


# AI-Job-Search and Match

An AI-powered job search assistant that analyzes resumes, matches job listings, and provides intelligent job search insights.  

## Features  

- Analyze resumes and extract key information.  
- Search for job listings based on user input.  
- Match job listings with resumes and provide detailed insights.  
- Visualize the job search architecture.  

## Requirements  

- Docker  
- Docker Compose  

## Getting Started  

### Clone the Repository  

```bash  
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Job-Search.git  
cd AI-Job-Search  
```  

### Build the Docker Image  

```bash  
docker build -t ai-job-search .  
```  

### Run the Docker Container  

```bash  
docker run -p 8504:8504 ai-job-search  
```  

### Access the Application  

Open your web browser and go to [http://localhost:8504](http://localhost:8504).  

---

## Project Structure  

```
.
├── Dockerfile               # Docker configuration file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── main.py                  # Main application code
├── flowchart.py             # Code for generating the job search architecture diagram
└── ...                      # Additional files as needed
```  

---

## Usage  

1. **Enter Job Search Description**: Provide a description of the job you are looking for.  
2. **Upload Resume**: Upload or paste your complete resume for AI analysis.  
3. **Enter Groq API Key**: Provide your Groq API key for AI-powered job search and matching.  
4. **Start Advanced Search**: Click the button to initiate the job search and receive insights.  

---

## Contact  

For any questions or inquiries, please contact me on [LinkedIn](https://www.linkedin.com/in/nihar-palem-1b955a183/).  

---

