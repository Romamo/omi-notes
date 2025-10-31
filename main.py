from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel as LangChainBaseModel, Field
import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
import requests
app = FastAPI(title="Field Notes Backend")

# Data directory for persistent storage
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# File paths for persistent storage
PROMPTS_FILE = DATA_DIR / "prompts.json"
TOKENS_FILE = DATA_DIR / "tokens.json"
SHEET_IDS_FILE = DATA_DIR / "sheet_ids.json"

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/oauth/callback")

# Data Models
class Segment(BaseModel):
    id: str
    text: str

class TranscriptPayload(BaseModel):
    session_id: str
    segments: List[Segment]
    location: Optional[str] = None
    datetime: Optional[str] = None

class InstructionPayload(BaseModel):
    session_id: str
    prompt: str

class SheetPayload(BaseModel):
    session_id: str
    sheet_url: str  # Google Sheets URL

# Extraction schema for LangChain
class ExtractionResult(LangChainBaseModel):
    """Schema for structured data extraction from transcripts"""
    items: List[Dict[str, Any]] = Field(description="List of extracted items with their properties")

# Utility functions for persistent storage
def load_json_file(file_path: Path) -> Dict:
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_json_file(file_path: Path, data: Dict):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Load persistent data
PROMPTS = load_json_file(PROMPTS_FILE)
TOKENS = load_json_file(TOKENS_FILE)
SHEET_IDS = load_json_file(SHEET_IDS_FILE)

# LangChain setup (lazy initialization)
llm = None

def get_llm():
    global llm
    if llm is None and OPENAI_API_KEY:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
    return llm

extraction_prompt = ChatPromptTemplate.from_template("""
You are a field notes assistant.
You receive user instructions and a text segment.
Return structured JSON with fields mentioned in the instruction.
Do not explain or comment — output valid JSON only.

INSTRUCTION:
{user_prompt}

TRANSCRIPT:
{segment_text}
""")


extraction_chain = None

def get_extraction_chain():
    global extraction_chain
    if extraction_chain is None:
        llm = get_llm()
        if llm:
            extraction_chain = extraction_prompt | llm.with_structured_output(ExtractionResult)
    return extraction_chain

# Google OAuth setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def extract_sheet_id_from_url(url: str) -> str:
    """Extract sheet ID from Google Sheets URL"""
    # Handle various Google Sheets URL formats
    if '/d/' in url:
        parts = url.split('/d/')
        if len(parts) > 1:
            sheet_id = parts[1].split('/')[0]
            return sheet_id
    raise ValueError("Invalid Google Sheets URL")

def get_google_flow():
    """Create Google OAuth flow"""
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=GOOGLE_REDIRECT_URI
    )

def append_to_sheet(sheet_id: str, credentials: Dict[str, Any], rows: List[List[Any]]):
    """Append rows to Google Sheet"""
    creds = google.oauth2.credentials.Credentials(
        token=credentials['token'],
        refresh_token=credentials.get('refresh_token'),
        token_uri=credentials.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES
    )

    # Refresh token if needed
    if creds.expired and creds.refresh_token:
        creds.refresh(google.auth.transport.requests.Request())

    service = build('sheets', 'v4', credentials=creds)

    # Append rows
    body = {'values': rows}
    result = service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        body=body
    ).execute()

    return result

async def process_segments(payload: TranscriptPayload):
    """Background task to process transcript segments"""
    session_id = payload.session_id

    # Get stored data
    prompt = PROMPTS.get(session_id)
    credentials = TOKENS.get(session_id)
    sheet_id = SHEET_IDS.get(session_id)

    if not prompt or not credentials or not sheet_id:
        print(f"Missing data for session {session_id}")
        return

    all_rows = []

    for segment in payload.segments:
        try:
            # Extract structured data using LangChain
            chain = get_extraction_chain()
            if not chain:
                print(f"No LLM configured for session {session_id}")
                continue
            result = chain.invoke({
                "user_prompt": prompt,
                "segment_text": segment.text
            })

            # Process extracted items
            for item in result.items:
                row = []
                if payload.location:
                    row.append(payload.location)
                if payload.datetime:
                    row.append(payload.datetime)

                # Add extracted fields
                for key, value in item.items():
                    row.append(value)

                all_rows.append(row)

        except Exception as e:
            print(f"Error processing segment {segment.id}: {e}")
            continue

    # Append all rows to sheet
    if all_rows:
        try:
            append_to_sheet(sheet_id, credentials, all_rows)
            print(f"Appended {len(all_rows)} rows to sheet {sheet_id}")
        except Exception as e:
            print(f"Error appending to sheet: {e}")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve homepage with welcome message"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Field Notes Backend</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            p {
                color: #666;
                font-size: 18px;
                line-height: 1.6;
            }
            .links {
                margin-top: 30px;
                font-size: 14px;
                color: #888;
            }
            .links a {
                color: #007bff;
                text-decoration: none;
                margin: 0 10px;
            }
            .links a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Field Notes Backend</h1>
            <p>This is the backend API for processing field notes and transcripts. Use the available endpoints to manage your data extraction workflows.</p>
            <div class="links">
                <a href="/privacy">Privacy Policy</a> |
                <a href="/terms">Terms of Service</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# API Endpoints

@app.post("/instructions")
async def store_instructions(payload: InstructionPayload):
    """Store user-defined extraction prompt"""
    PROMPTS[payload.session_id] = payload.prompt
    save_json_file(PROMPTS_FILE, PROMPTS)
    return {"status": "ok"}

@app.get("/oauth/start")
async def oauth_start(uid: str):
    """Start Google OAuth2 flow"""
    flow = get_google_flow()
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=uid  # Store uid in state
    )
    return RedirectResponse(authorization_url)

@app.get("/oauth/callback")
async def oauth_callback(code: str, state: str):
    """Handle OAuth callback"""
    session_id = state
    flow = get_google_flow()
    flow.fetch_token(code=code)

    # Store credentials
    credentials = {
        'token': flow.credentials.token,
        'refresh_token': flow.credentials.refresh_token,
        'token_uri': flow.credentials.token_uri,
        'client_id': flow.credentials.client_id,
        'client_secret': flow.credentials.client_secret,
    }
    TOKENS[session_id] = credentials
    save_json_file(TOKENS_FILE, TOKENS)

    sheet_id = SHEET_IDS.get(session_id)
    return {
        "status": "connected",
        "sheet_id": sheet_id
    }

@app.post("/sheet")
async def store_sheet(payload: SheetPayload):
    """Store Google Sheet ID from URL"""
    try:
        sheet_id = extract_sheet_id_from_url(payload.sheet_url)
        SHEET_IDS[payload.session_id] = sheet_id
        save_json_file(SHEET_IDS_FILE, SHEET_IDS)
        return {"status": "ok", "sheet_id": sheet_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transcript")
async def receive_transcript(payload: TranscriptPayload, background_tasks: BackgroundTasks):
    """Receive transcript segments and process in background"""
    # Validate session has required data
    if payload.session_id not in PROMPTS:
        raise HTTPException(status_code=400, detail="No instructions found for session")
    if payload.session_id not in TOKENS:
        raise HTTPException(status_code=400, detail="No OAuth tokens found for session")
    if payload.session_id not in SHEET_IDS:
        raise HTTPException(status_code=400, detail="No sheet ID found for session")

    # Add background task
    background_tasks.add_task(process_segments, payload)

    return {
        "status": "accepted",
        "segments_received": len(payload.segments)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
