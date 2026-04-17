"""
LangChain utilities for resume processing
"""

import os
import re

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def _tokenize(text):
    """Tokenize text for lightweight lexical relevance scoring fallback."""
    tokens = re.findall(r"[a-zA-Z0-9_+#.-]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def _lexical_relevance_chunks(processed_resume, job_description, top_k=3):
    """Fallback ranking when vector embeddings are unavailable."""
    job_tokens = _tokenize(job_description)
    if not job_tokens:
        return []

    scored = []
    for doc in processed_resume["chunks"]:
        chunk_tokens = _tokenize(doc.page_content)
        if not chunk_tokens:
            continue

        overlap = len(job_tokens & chunk_tokens)
        overlap_ratio = overlap / len(job_tokens)
        density_ratio = overlap / len(chunk_tokens)
        score = min(1.0, (0.85 * overlap_ratio) + (0.15 * density_ratio))
        scored.append((doc.page_content, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


# Initialize LangChain components
def init_langchain_components(
    api_key,
    chat_model=DEFAULT_OPENROUTER_MODEL,
    embedding_model=DEFAULT_OPENROUTER_EMBEDDING_MODEL,
):
    """Initialize LangChain components.
    
    Args:
        api_key: OpenRouter API key
        chat_model: OpenRouter model for chat completions
        embedding_model: OpenRouter model for embeddings
        
    Returns:
        tuple: (embeddings, llm) or (None, None) if error
    """
    if not api_key:
        return None, None

    default_headers = {
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "resume-shortlister-mcp"),
    }

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers,
    )
    llm = ChatOpenAI(
        temperature=0,
        model=chat_model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers,
    )
    return embeddings, llm

def prepare_resume_documents(resume_text, filename):
    """
    Split resume text into chunks and wrap them as LangChain Document objects.
    
    Args:
        resume_text: Raw resume text
        filename: Name of the resume file
    
    Returns:
        dict: Contains original text and chunked Document list
    """
    # Step 1: Chunk the resume
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(resume_text)

    # Step 2: Wrap each chunk in a Document with metadata
    documents = [
        Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    return {
        "text": resume_text,
        "chunks": documents
    }

def find_relevant_sections(processed_resume, job_description, embeddings):
    """
    Find top 3 resume chunks relevant to a job description.
    
    Args:
        processed_resume: Output of process_resume_with_langchain (includes chunks)
        job_description: Job description string
        embeddings: OpenRouter-backed embeddings object
    
    Returns:
        List of (chunk_text, similarity_score) tuples
    """
    if embeddings:
        try:
            # Build FAISS index from processed chunks
            vectorstore = FAISS.from_documents(processed_resume["chunks"], embeddings)

            # Relevance scores are normalized to [0, 1]
            results = vectorstore.similarity_search_with_relevance_scores(job_description, k=3)
            return [
                (doc.page_content, max(0.0, min(1.0, float(score))))
                for doc, score in results
            ]
        except Exception:
            # OpenRouter embeddings may not be available on every account/model plan.
            pass

    return _lexical_relevance_chunks(processed_resume, job_description, top_k=3)


def extract_skills_with_langchain(resume_text, llm):
    """Extract skills from resume text using LangChain.
    
    Args:
        resume_text: Resume text content
        llm: LangChain language model
        
    Returns:
        str: Extracted skills or error message
    """
    if not llm:
        return "LangChain LLM not available for skill extraction."
    
    try:
        # Create a skill extraction chain
        prompt = PromptTemplate.from_template(
            """
            Extract the skills from the following resume. 
            Organize them into categories like:
            - Technical Skills
            - Soft Skills
            - Languages
            - Tools & Platforms
            
            Resume:
            {resume_text}
            
            Extracted Skills:
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain
        skills = chain.invoke({"resume_text": resume_text})
        return skills
        
    except Exception as e:
        return f"Error extracting skills: {str(e)}"

def assess_resume_for_job(resume_text, job_description, llm):
    """Assess how well a resume matches a job description.
    
    Args:
        resume_text: Resume text content
        job_description: Job description text
        llm: LangChain language model
        
    Returns:
        str: Assessment or error message
    """
    if not llm:
        return "LangChain LLM not available for resume assessment."
    
    try:
        # Create an assessment chain
        prompt = PromptTemplate.from_template(
            """
            You are a skilled recruiter. Evaluate how well the following resume matches the job description.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Provide an assessment with the following sections:
            1. Match Score (0-100)
            2. Matching Skills & Qualifications
            3. Missing Skills & Qualifications
            4. Overall Assessment
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain
        assessment = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
        return assessment
        
    except Exception as e:
        return f"Error assessing resume: {str(e)}"

def analyze_candidate_profiles(resume_text, job_description, llm):
    """Analyze LinkedIn and GitHub profiles found in the resume.
    
    Args:
        resume_text: Resume text content
        job_description: Job description text
        llm: LangChain language model
        
    Returns:
        str: Profile analysis or error message
    """
    if not llm:
        return "LangChain LLM not available for profile analysis."
    
    try:
        prompt = PromptTemplate.from_template(
            """
            You are an expert tech recruiter. Your task is to extract any URLs for LinkedIn, GitHub, or personal portfolios from the candidate's resume, and provide an analysis of how these profiles add value for the target job description. 
            
            Note: Since you cannot browse the live web, base your "analysis" on the presence of these links, the information explicitly stated in the resume related to them (like open source contributions mentioned, projects hosted on GitHub, or LinkedIn summaries), and provide a hypothetical rating on how critical these profiles are for the given job.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Provide the following sections:
            1. Extracted Links (LinkedIn, GitHub, Portfolio, etc.)
            2. Profile Value Rating (1-10) for the specific job description
            3. Analysis (What we can deduce from the resume about their online presence and why it matters)
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
        
    except Exception as e:
        return f"Error analyzing profiles: {str(e)}"

def generate_interview_questions(resume_text, job_description, llm):
    """Generate tailored interview questions based on the candidate's resume and job description.
    
    Args:
        resume_text: Resume text content
        job_description: Job description text
        llm: LangChain language model
        
    Returns:
        str: Generated interview questions
    """
    if not llm:
        return "LangChain LLM not available for question generation."
    
    try:
        prompt = PromptTemplate.from_template(
            """
            You are an expert technical interviewer. Based on the candidate's resume and the job description, generate tailored interview questions that probe both their stated experience and the requirements of the role.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Please provide:
            1. 3 Technical Questions (testing specific skills mentioned in the resume that overlap with the job)
            2. 2 Behavioral Questions (based on their experience level and project history)
            3. 1 "Gap" Question (exploring an area required by the job but missing or weak on the resume)
            4. Suggested answers or key things to listen for in the candidate's response.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
        
    except Exception as e:
        return f"Error generating interview questions: {str(e)}"


def generate_hiring_scorecard(resume_text, job_description, llm):
    """Generate a structured hiring scorecard and final decision memo.

    Args:
        resume_text: Resume text content
        job_description: Job description text
        llm: LangChain language model

    Returns:
        str: Hiring scorecard memo
    """
    if not llm:
        return "LangChain LLM not available for hiring scorecard generation."

    try:
        prompt = PromptTemplate.from_template(
            """
            You are a senior hiring panel assistant.
            Build an evidence-based hiring scorecard for this candidate using only the resume and the target job description.

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Return the response with these exact sections:
            1. Candidate Snapshot (max 80 words)
            2. Weighted Scorecard
               - Technical Match (Weight 30, Score 0-10, Evidence)
               - Domain/Industry Fit (Weight 20, Score 0-10, Evidence)
               - Project Impact and Ownership (Weight 20, Score 0-10, Evidence)
               - Communication/Leadership Signals (Weight 10, Score 0-10, Evidence)
               - Portfolio and Open-Source Signals (Weight 10, Score 0-10, Evidence)
               - Risk Penalty (Weight -10 to 0, Evidence)
            3. Final Fit Score (0-100) with short calculation
            4. Recommendation (Strong Hire / Hire / Hold / No Hire)
            5. Top 3 Interview Focus Areas
            6. 30-60-90 Day Ramp Plan (if hired)

            Be concise, specific, and evidence-based.
            """
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })

    except Exception as e:
        return f"Error generating hiring scorecard: {str(e)}"