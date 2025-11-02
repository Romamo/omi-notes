import asyncio
import traceback
from datetime import datetime

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials

load_dotenv()

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, create_model
from typing import List, Optional, Dict, Any
import os
import json
import hashlib
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
import argparse

# Global buffers for transcript processing per session
transcript_buffers = {}  # uid -> list of TranscriptPayload
processing_timers = {}  # uid -> asyncio.Task

async def delayed_process(uid: str):
    """Wait 10 seconds then process buffered transcripts for a session"""
    await asyncio.sleep(10)
    await process_buffered_transcripts(uid)

async def process_buffered_transcripts(uid: str):
    """Process all buffered transcripts for a session"""

    if uid not in transcript_buffers:
        return

    buffered_payloads = transcript_buffers[uid]
    if not buffered_payloads:
        return

    logger.info(f"Processing buffered transcripts for session: {uid} count:{len(buffered_payloads)}")

    # Clear the buffer
    del transcript_buffers[uid]

    # Combine all segments from all payloads
    all_segments = []
    location = None
    datetime_val = None

    for payload in buffered_payloads:
        all_segments.extend(payload.segments)
        if payload.location:
            location = payload.location
        if payload.datetime:
            datetime_val = payload.datetime

    # Create a combined payload
    combined_payload = TranscriptPayload(
        session_id=uid,
        segments=all_segments,
        location=location,
        datetime=datetime_val
    )

    # Process the combined payload
    await process_segments(combined_payload)

app = FastAPI(title="Field Notes Backend")

# Templates
templates = Jinja2Templates(directory="templates")

# Data directory for persistent storage
PATH_DATA = os.getenv("PATH_DATA", "data")
DATA_DIR = Path(PATH_DATA)
DATA_DIR.mkdir(exist_ok=True)

# Utility functions for user-specific storage
def get_user_data_dir(uid: str) -> Path:
    """Get user-specific data directory"""
    user_dir = DATA_DIR / uid
    user_dir.mkdir(exist_ok=True)
    return user_dir

def get_user_file_path(uid: str, filename: str) -> Path:
    """Get path to user-specific file"""
    return get_user_data_dir(uid) / filename

# Legacy file paths for backward compatibility (will be migrated)
# PROMPTS_FILE = DATA_DIR / "prompts.json"
# TOKENS_FILE = DATA_DIR / "tokens.json"
# SHEET_IDS_FILE = DATA_DIR / "sheet_ids.json"

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

# Function to generate and cache Pydantic models using LLM
def generate_pydantic_model_with_llm(prompt: str, uid: str):
    """Generate a Pydantic model using LLM based on the user's extraction prompt"""
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    cache_file = get_user_file_path(uid, f"model_{prompt_hash}.py")

    # Check if we already have a cached model
    if cache_file.exists():
        try:
            # Import the cached model
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"user_model_{prompt_hash}", cache_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.OutputResponse
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")

    # Generate new model using LLM
    llm = get_llm()
    if not llm:
        # Fallback to basic model
        return create_fallback_model(prompt_hash)

    schema_prompt = f"""
    Based on this extraction instruction: "{prompt}"

    Generate a Python Pydantic model class that represents the structured data to extract.
    The model should be called "OutputResponse" and inherit from BaseModel.

    The OutputResponse should have:
    - items: List[DynamicExtractionResult] - where DynamicExtractionResult is a nested model with fields for the extracted data
    - explain: str - explanation of extraction decisions

    Return ONLY the Python class definition, no imports or other code.

    Example format:
    class DynamicExtractionResult(BaseModel):
        name: Optional[str] = Field(default=None, description="Extracted name")
        date: Optional[str] = Field(default=None, description="Extracted date")

    class OutputResponse(BaseModel):
        items: List[DynamicExtractionResult] = Field(default_factory=list, description="List of extracted items")
        explain: str = Field(description="Explanation of extraction decisions")

    Make the DynamicExtractionResult model appropriate for the extraction task described in the instruction.
    Focus on the specific fields mentioned or implied in the instruction.
    """

    try:
        response = llm.invoke(schema_prompt)
        model_code = str(response.content).strip()

        # Clean up the response (remove markdown code blocks if present)
        if "```python" in model_code:
            model_code = model_code.split("```python")[1].split("```")[0].strip()
        elif "```" in model_code:
            model_code = model_code.split("```")[1].split("```")[0].strip()

        # Add necessary imports
        full_code = f"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

{model_code}
"""

        # Save to cache file
        with open(cache_file, 'w') as f:
            f.write(full_code)

        # Execute and return the model
        exec_globals = {
            'BaseModel': BaseModel,
            'Field': Field,
            'List': List,
            'Dict': Dict,
            'Any': Any,
            'Optional': Optional
        }
        logger.info(f"GENERATED RESPONSE DATA MODEL: {full_code}")
        exec(full_code, exec_globals)
        return exec_globals['OutputResponse']

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error generating model with LLM: {e}")
        return create_fallback_model(prompt_hash)

def create_fallback_model(prompt_hash: str):
    """Create a fallback model when LLM generation fails"""
    FallbackModel = create_model(
        f'ExtractionResult_{prompt_hash}',
        items=(List[Dict[str, Any]], Field(default_factory=list, description="List of extracted items")),
        __base__=LangChainBaseModel
    )
    return FallbackModel
    confidence: str = Field(description="Reason why data extracted or not", default="")

# Utility functions for persistent storage
def load_json_file(file_path: Path) -> Dict:
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_json_file(file_path: Path, data: Dict):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Load persistent data (legacy - will be migrated to user-specific storage)
# PROMPTS = load_json_file(PROMPTS_FILE)
# TOKENS = load_json_file(TOKENS_FILE)
# SHEET_IDS = load_json_file(SHEET_IDS_FILE)

# User-specific data loading functions
def load_user_prompts(uid: str) -> str:
    """Load user-specific prompt (single prompt per user for now)"""
    file_path = get_user_file_path(uid, "prompt.txt")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if content:  # Only return if there's actual content
                return content

    # If no user prompt exists, load and set default prompt
    default_prompt_path = Path("templates/default_prompt.txt")
    if default_prompt_path.exists():
        with open(default_prompt_path, 'r') as f:
            default_content = f.read().strip()
            if default_content:
                # Save the default prompt for the user
                save_user_prompts(uid, default_content)
                return default_content

    return ""

def save_user_prompts(uid: str, prompt: str):
    """Save user-specific prompt (single prompt per user for now)"""
    file_path = get_user_file_path(uid, "prompt.txt")
    with open(file_path, 'w') as f:
        f.write(prompt)

def load_user_tokens(uid: str) -> Credentials | None:
    """Load user-specific tokens"""
    file_path = get_user_file_path(uid, "tokens.json")
    if not file_path.exists():
        return None
    # Load credentials from JSON using from_authorized_user_info
    return google.oauth2.credentials.Credentials.from_authorized_user_file(str(file_path))

def save_user_tokens(uid: str, credentials: Credentials):
    """Save user-specific tokens"""
    file_path = get_user_file_path(uid, "tokens.json")
    # Store credentials as JSON using to_json()
    with open(file_path, 'w') as f:
        f.write(credentials.to_json())

def load_user_sheet_ids(uid: str) -> str:
    """Load user-specific sheet ID (single sheet per user for now)"""
    file_path = get_user_file_path(uid, "sheet_id.txt")
    if file_path.exists():
        with open(file_path, 'r') as f:
            return f.read().strip()
    return ""

def save_user_sheet_ids(uid: str, sheet_id: str):
    """Save user-specific sheet ID (single sheet per user for now)"""
    file_path = get_user_file_path(uid, "sheet_id.txt")
    with open(file_path, 'w') as f:
        f.write(sheet_id)

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


extraction_chains = {}

def get_extraction_chain(uid: str):
    """Get or create extraction chain for user based on their prompt"""
    global extraction_chains

    # Get user's prompt
    prompt = load_user_prompts(uid)
    if not prompt:
        return None

    # Create cache key based on prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    cache_key = f"{uid}_{prompt_hash}"

    if cache_key not in extraction_chains:
        llm = get_llm()
        if llm:
            # Generate model using LLM
            DynamicExtractionResult = generate_pydantic_model_with_llm(prompt, uid)
            extraction_chains[cache_key] = extraction_prompt | llm.with_structured_output(DynamicExtractionResult, method="function_calling")

    return extraction_chains.get(cache_key)

# Google OAuth setup
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive'
]

# Warning('Scope has changed from "https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/drive" to
# "https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive".')

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

def append_to_sheet(uid, sheet_id: str, credentials: Credentials, rows: List[List[Any]], columns: Optional[List[str]] = None):
    credentials_original = credentials._make_copy()

    service = build('sheets', 'v4', credentials=credentials)

    # If columns are provided and this is a new sheet (no existing data), add headers
    if columns:
        # Check if sheet is empty by trying to read A1
        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=sheet_id,
                range="Sheet1!A1"
            ).execute()
            existing_data = result.get('values', [])
        except:
            existing_data = []

        # If sheet is empty, add column headers
        if not existing_data:
            header_body = {'values': [columns]}
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range="Sheet1!A1",
                valueInputOption="RAW",
                body=header_body
            ).execute()

    # Append rows
    body = {'values': rows}
    result = service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="Sheet1!A2",  # Start from A2 to avoid overwriting headers
        valueInputOption="RAW",
        body=body
    ).execute()

    # Check if the token was refreshed internally to write to a file
    if credentials_original.token != credentials.token:
        save_user_tokens(uid, credentials)

    return result

def list_user_sheets(credentials: Credentials) -> List[Dict[str, Any]]:
    # print(credentials.token_state)
    # Refresh token if needed
    # if credentials.token_state == 'EXPIRED' and creds.refresh_token:
    #     creds.refresh(google.auth.transport.requests.Request())

    service = build('drive', 'v3', credentials=credentials)

    # List spreadsheets owned by user (simpler approach)
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.spreadsheet'",
        fields="files(id,name,modifiedTime,owners)",
        corpora="user"
    ).execute()

    files = results.get('files', [])

    # Filter to only include sheets where user is owner or has edit permission
    filtered_files = []
    for file in files:
        # Check if user is owner
        if file.get('owners', []):
            user_email = None
            try:
                # Get user info to check ownership
                user_info = service.about().get(fields="user").execute()
                user_email = user_info.get('user', {}).get('emailAddress')
            except:
                pass

            # Include if user is owner or if we can't determine (fallback to showing all)
            if user_email and any(owner.get('emailAddress') == user_email for owner in file.get('owners', [])):
                filtered_files.append(file)
            elif not user_email:  # If we can't get user email, include the file
                filtered_files.append(file)

    return filtered_files

def create_google_sheet(credentials: Credentials, title: str) -> str:

    service = build('sheets', 'v4', credentials=credentials)

    # Create new spreadsheet
    spreadsheet = {
        'properties': {
            'title': title
        }
    }

    result = service.spreadsheets().create(body=spreadsheet).execute()
    return result.get('spreadsheetId')

async def process_segments(payload: TranscriptPayload):
    """Background task to process transcript segments"""
    session_id = payload.session_id
    uid = session_id  # For now, session_id serves as uid

    logger.info(f"Processing segments for session {session_id}")

    # Get stored data from user-specific storage
    prompt = load_user_prompts(uid)
    credentials = load_user_tokens(uid)
    sheet_id = load_user_sheet_ids(uid)

    if not prompt or not credentials or not sheet_id:
        logger.error(f"Missing data for session {session_id}")
        return

    all_rows = []
    columns = ['Location', 'Date']

    user_segments = [segment.text for segment in payload.segments]
    user_text = ' '.join(user_segments)
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"USER: {user_text}")

    # Extract structured data using LangChain
    chain = get_extraction_chain(uid)
    if not chain:
        raise (f"No LLM configured for user {uid}")

    result = chain.invoke({
        "user_prompt": prompt,
        "segment_text": user_text
    })

    # Process extracted items
    for item in result.items:
        row = []
        row.append(payload.location or 'unknown')
        row.append(datetime.now().strftime('%Y-%m-%d %H:%M'))

        if len(columns) == 2:
            columns.extend(item.model_fields.keys())

        # Add extracted fields
        dump = item.model_dump()
        for key in item.model_fields.keys():
            row.append(dump.get(key))

        all_rows.append(row)

    logger.info(f"Data extracted: {all_rows}")

    # Append all rows to sheet
    if all_rows:
        try:
            append_to_sheet(uid, sheet_id, credentials, all_rows, columns)
            logger.info(f"Appended {len(all_rows)} rows to sheet {sheet_id}")
        except Exception as e:
            print(f"Error appending to sheet: {e}")

# API Endpoints

@app.get("/sheets")
async def sheets_page(request: Request, uid: str):
    """Serve Google Sheets selection/creation page using template"""
    return templates.TemplateResponse("sheets.html", {"request": request, "uid": uid})

# API Endpoints

@app.get("/api/sheets")
async def list_user_sheets_api(uid: str):
    """API endpoint to list user's Google Sheets"""
    credentials = load_user_tokens(uid)

    if not credentials:
        return {"error": "No OAuth credentials found"}

    try:
        sheets = list_user_sheets(credentials)
        return {"sheets": sheets}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def homepage():
    """Serve homepage with welcome message"""
    return templates.TemplateResponse("index.html", {"request": {}})

# API Endpoints

@app.post("/instructions")
async def store_instructions(request: Request):
    """Store user-defined extraction prompt"""
    data = await request.json()
    uid = data.get('uid')
    prompt = data.get('prompt')

    if not uid or not prompt:
        raise HTTPException(status_code=400, detail="uid and prompt are required")

    # Use user-specific storage
    save_user_prompts(uid, prompt)
    return {"status": "ok"}

@app.get("/api/instructions")
async def get_instructions(uid: str):
    """Retrieve user-defined extraction prompt"""
    prompt = load_user_prompts(uid)
    return {"prompt": prompt}
@app.get("/api/current-sheet")
async def get_current_sheet(uid: str):
    """Retrieve user's current sheet information"""
    sheet_id = load_user_sheet_ids(uid)
    if sheet_id:
        return {"sheet_id": sheet_id}
    else:
        return {"sheet_id": None}

@app.get("/oauth/start")
async def oauth_start(uid: str, force: bool = True):
    """Start Google OAuth2 flow"""

    # https://github.com/googleads/google-ads-python/blob/main/examples/authentication/generate_user_credentials.py
    credentials = load_user_tokens(uid)
    if not credentials:
        flow = get_google_flow()
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=uid,  # Store uid in state
            approval_prompt='force' if force else None
        )
        return RedirectResponse(authorization_url)
    return RedirectResponse(f"/sheets?uid={uid}")

@app.get("/oauth/callback")
async def oauth_callback(code: str, state: str):
    """Handle OAuth callback"""
    uid = state  # state contains the uid
    flow = get_google_flow()

    try:
        flow.fetch_token(code=code)
    except Exception as e:
        print(e)

    if not flow.credentials.refresh_token:
        return RedirectResponse("/oauth/start?uid={}&force=true".format(uid))

    # Store credentials directly
    save_user_tokens(uid, flow.credentials)

    # Redirect to sheet selector page
    return RedirectResponse(f"/sheets?uid={uid}")

@app.post("/api/sheet")
async def store_sheet_api(payload: SheetPayload):
    """Store Google Sheet ID from URL"""
    try:
        sheet_id = extract_sheet_id_from_url(payload.sheet_url)
        # Use user-specific storage
        uid = payload.session_id  # For now, session_id serves as uid
        save_user_sheet_ids(uid, sheet_id)
        return {"status": "ok", "sheet_id": sheet_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transcript")
async def receive_transcript(payload: TranscriptPayload):
    """Receive transcript segments and buffer them for delayed processing"""
    uid = payload.session_id  # For now, session_id serves as uid

    logger.info(f"Received transcript for session {uid}")

    # Validate session has required data in user-specific storage
    prompt = load_user_prompts(uid)
    credentials = load_user_tokens(uid)
    sheet_id = load_user_sheet_ids(uid)

    if not prompt:
        raise HTTPException(status_code=400, detail="No instructions found for session")
    if not credentials:
        raise HTTPException(status_code=400, detail="No OAuth tokens found for session")
    if not sheet_id:
        raise HTTPException(status_code=400, detail="No sheet ID found for session")

    # Buffer the payload
    if uid not in transcript_buffers:
        transcript_buffers[uid] = []
    transcript_buffers[uid].append(payload)

    # Start a new 10-second timer if not already running
    if uid not in processing_timers or processing_timers[uid].done():
        processing_timers[uid] = asyncio.create_task(delayed_process(uid))

    return {
        "status": "accepted",
        "segments_received": len(payload.segments)
        }

@app.get("/privacy")
async def privacy_policy():
    """Serve privacy policy page"""
    return templates.TemplateResponse("privacy.html", {"request": {}})

@app.get("/terms")
async def terms_of_service():
    """Serve terms of service page"""
    return templates.TemplateResponse("terms.html", {"request": {}})
