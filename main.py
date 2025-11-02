import traceback

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials

load_dotenv()
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
app = FastAPI(title="Field Notes Backend")

# Templates
templates = Jinja2Templates(directory="templates")

# Data directory for persistent storage
DATA_DIR = Path("data")
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
            print(f"Error loading cached model: {e}")

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
        print(f"GENERATED RESPONSE DATA MODEL: {full_code}")
        exec(full_code, exec_globals)
        return exec_globals['OutputResponse']

    except Exception as e:
        traceback.print_exc()
        print(f"Error generating model with LLM: {e}")
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
PROMPTS = load_json_file(PROMPTS_FILE)
TOKENS = load_json_file(TOKENS_FILE)
SHEET_IDS = load_json_file(SHEET_IDS_FILE)

# User-specific data loading functions
def load_user_prompts(uid: str) -> str:
    """Load user-specific prompt (single prompt per user for now)"""
    file_path = get_user_file_path(uid, "prompt.txt")
    if file_path.exists():
        with open(file_path, 'r') as f:
            return f.read().strip()
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
    # """Append rows to Google Sheet"""
    # creds = google.oauth2.credentials.Credentials(
    #     token=credentials['token'],
    #     refresh_token=credentials.get('refresh_token'),
    #     token_uri=credentials.get('token_uri', 'https://oauth2.googleapis.com/token'),
    #     client_id=GOOGLE_CLIENT_ID,
    #     client_secret=GOOGLE_CLIENT_SECRET,
    #     scopes=SCOPES
    # )

    # # Refresh token if needed
    # if creds.expired and creds.refresh_token:
    #     creds.refresh(google.auth.transport.requests.Request())
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
    print(credentials.token_state)
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

def create_google_sheet(credentials: Dict[str, Any], title: str) -> str:
    """Create a new Google Sheet and return its ID"""
    # Ensure we have all required fields for token refresh
    creds_dict = {
        'token': credentials['token'],
        'refresh_token': credentials.get('refresh_token'),
        'token_uri': credentials.get('token_uri', 'https://oauth2.googleapis.com/token'),
        'client_id': credentials.get('client_id', GOOGLE_CLIENT_ID),
        'client_secret': credentials.get('client_secret', GOOGLE_CLIENT_SECRET),
        'scopes': SCOPES
    }

    # Check if refresh_token is None and try to get it from the credentials
    if creds_dict['refresh_token'] is None:
        # For Google OAuth, if refresh_token is None, we might need to re-authenticate
        # But for now, let's try to create the credentials without refresh_token
        creds = google.oauth2.credentials.Credentials(
            token=creds_dict['token'],
            refresh_token=None,
            token_uri=creds_dict['token_uri'],
            client_id=creds_dict['client_id'],
            client_secret=creds_dict['client_secret'],
            scopes=creds_dict['scopes']
        )
    else:
        creds = google.oauth2.credentials.Credentials(**creds_dict)

    # Only refresh if we have a refresh token and token is expired
    if creds.expired and creds.refresh_token:
        creds.refresh(google.auth.transport.requests.Request())

    service = build('sheets', 'v4', credentials=creds)

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

    # Get stored data from user-specific storage
    prompt = load_user_prompts(uid)
    credentials = load_user_tokens(uid)
    sheet_id = load_user_sheet_ids(uid)

    if not prompt or not credentials or not sheet_id:
        print(f"Missing data for session {session_id}")
        return

    all_rows = []
    columns = ['Location', 'Date']

    user_segments = [segment.text for segment in payload.segments]
    user_text = ' '.join(user_segments)
    print(f"PROMPT: {prompt}")
    print(f"USER: {user_text}")

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
        row.append(payload.datetime or payload.datetime.format("YYYY-MM-DD HH:mm"))

        if len(columns) == 2:
            columns.extend(item.model_fields.keys())

        # Add extracted fields
        dump = item.model_dump()
        for key in item.model_fields.keys():
            row.append(dump.get(key))

        all_rows.append(row)

    print(f"Data extracted: {all_rows}")

    # Append all rows to sheet
    if all_rows:
        try:
            append_to_sheet(uid, sheet_id, credentials, all_rows, columns)
            print(f"Appended {len(all_rows)} rows to sheet {sheet_id}")
        except Exception as e:
            print(f"Error appending to sheet: {e}")

# API Endpoints

@app.get("/sheets", response_class=HTMLResponse)
async def sheets_page(uid: str):
    """Serve Google Sheets selection/creation page"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Google Sheets - Field Notes Backend</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }}
            h1 {{
                color: #333;
                margin-bottom: 30px;
            }}
            .option {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .option:hover {{
                background-color: #e9ecef;
            }}
            .option h3 {{
                margin: 0 0 10px 0;
                color: #495057;
            }}
            .option p {{
                margin: 0;
                color: #6c757d;
                font-size: 14px;
            }}
            .form-group {{
                margin: 20px 0;
                text-align: left;
            }}
            .form-group label {{
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #495057;
            }}
            .form-group input {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 16px;
            }}
            .btn {{
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
                text-decoration: none;
                display: inline-block;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
            .btn-secondary {{
                background-color: #6c757d;
            }}
            .btn-secondary:hover {{
                background-color: #545b62;
            }}
            .back-link {{
                margin-top: 40px;
            }}
            .back-link a {{
                color: #007bff;
                text-decoration: none;
            }}
            #create-form {{
                display: none;
            }}
            .sheets-list {{
                max-height: 300px;
                overflow-y: auto;
                margin: 20px 0;
            }}
            .sheet-item {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                margin: 10px 0;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .sheet-item:hover {{
                background-color: #e9ecef;
            }}
            .sheet-item h4 {{
                margin: 0 0 5px 0;
                color: #495057;
            }}
            .sheet-item p {{
                margin: 0;
                color: #6c757d;
                font-size: 12px;
            }}
            .loader {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .loading-text {{
                display: inline;
                vertical-align: middle;
            }}
        </style>
        <script>
            function showCreateForm() {{
                document.getElementById('create-form').style.display = 'block';
                document.getElementById('options').style.display = 'none';
            }}

            function showOptions() {{
                document.getElementById('create-form').style.display = 'none';
                document.getElementById('user-sheets').style.display = 'none';
                document.getElementById('options').style.display = 'block';
            }}

            async function loadUserSheets() {{
                console.log('loadUserSheets called');
                const optionsDiv = document.getElementById('options');
                const userSheetsDiv = document.getElementById('user-sheets');
                console.log('optionsDiv:', optionsDiv);
                console.log('userSheetsDiv:', userSheetsDiv);

                optionsDiv.style.display = 'none';
                userSheetsDiv.style.display = 'block';
                console.log('Set display styles');

                const sheetsList = document.getElementById('sheets-list');
                console.log('sheetsList element:', sheetsList);
                sheetsList.innerHTML = '<div class="loader"></div><span class="loading-text">Loading your sheets...</span>';
                console.log('Set loading HTML');
                console.log('Current sheetsList.innerHTML:', sheetsList.innerHTML);

                try {{
                    console.log('Making API call...');
                    const response = await fetch(`/api/sheets?uid={uid}`);
                    console.log('API response:', response);
                    const data = await response.json();
                    console.log('API data:', data);

                    if (data.error) {{
                        sheetsList.innerHTML = `<p>Error: ${{data.error}}</p>`;
                        return;
                    }}

                    if (data.sheets.length === 0) {{
                        sheetsList.innerHTML = '<p>No Google Sheets found. Create a new one or use an existing URL.</p>';
                        return;
                    }}

                    const sheetsHtml = data.sheets.slice(0, 5).map(sheet => `
                        <div class="sheet-item" onclick="selectSheet('${{sheet.id}}', '${{sheet.name.replace(/'/g, "\\'")}}')">
                            <h4>${{sheet.name}}</h4>
                            <p>Last modified: ${{new Date(sheet.modifiedTime).toLocaleDateString()}}</p>
                        </div>
                    `).join('');

                    sheetsList.innerHTML = sheetsHtml;
                    console.log('Set sheets HTML');
                    console.log('Sheets HTML content:', sheetsHtml);
                    console.log('Final sheetsList.innerHTML:', sheetsList.innerHTML);
                }} catch (error) {{
                    console.error('Error loading sheets:', error);
                    sheetsList.innerHTML = '<p>Error loading sheets. Please try again.</p>';
                }}
            }}

            async function selectSheet(sheetId, sheetName) {{
                if (confirm(`Use "${{sheetName}}" for storing field notes?`)) {{
                    try {{
                        const response = await fetch('/api/sheet', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                            }},
                            body: JSON.stringify({{
                                session_id: '{uid}',
                                sheet_url: `https://docs.google.com/spreadsheets/d/${{sheetId}}`
                            }})
                        }});

                        if (response.ok) {{
                            const data = await response.json();
                            alert('Sheet selected successfully! You can now use it for storing field notes.');
                            window.location.href = '/';  // Redirect to home
                        }} else {{
                            const error = await response.json();
                            alert('Error selecting sheet: ' + (error.detail || 'Unknown error'));
                        }}
                    }} catch (error) {{
                        alert('Error selecting sheet. Please try again.');
                    }}
                }}
            }}

            async function createSheet() {{
                const sheetName = document.getElementById('sheet-name').value.trim();
                if (!sheetName) {{
                    alert('Please enter a sheet name');
                    return;
                }}

                try {{
                    const response = await fetch(`/api/sheets/create?uid={uid}&title=${{encodeURIComponent(sheetName)}}`, {{
                        method: 'POST'
                    }});

                    const data = await response.json();

                    if (data.error) {{
                        alert('Error creating sheet: ' + data.error);
                        return;
                    }}

                    if (data.success) {{
                        alert('Sheet created successfully! You can now use it for storing field notes.');
                        // Submit JSON to store the newly created sheet
                        const storeResponse = await fetch('/api/sheet', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                            }},
                            body: JSON.stringify({{
                                session_id: '{uid}',
                                sheet_url: data.url
                            }})
                        }});

                        if (storeResponse.ok) {{
                            window.location.href = '/';  // Redirect to home
                        }} else {{
                            alert('Sheet created but failed to store. Please try selecting it manually.');
                        }}
                    }} else {{
                        alert('Error creating sheet: ' + (data.error || 'Unknown error'));
                    }}
                }} catch (error) {{
                    alert('Error creating sheet. Please try again.');
                }}

                // Reset form
                document.getElementById('sheet-name').value = '';
                showOptions();
            }}

            async function useExistingSheet() {{
                const sheetUrl = document.getElementById('sheet-url').value.trim();
                if (!sheetUrl) {{
                    alert('Please enter a Google Sheets URL');
                    return;
                }}

                try {{
                    const response = await fetch('/api/sheet', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            session_id: '{uid}',
                            sheet_url: sheetUrl
                        }})
                    }});

                    if (response.ok) {{
                        const data = await response.json();
                        alert('Sheet stored successfully! You can now use it for storing field notes.');
                        window.location.href = '/';  // Redirect to home
                    }} else {{
                        const error = await response.json();
                        alert('Error storing sheet: ' + (error.detail || 'Unknown error'));
                    }}
                }} catch (error) {{
                    alert('Error storing sheet. Please try again.');
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Google Sheets Setup</h1>

            <div id="options">
                <div class="option" onclick="showCreateForm()">
                    <h3>📄 Create New Sheet</h3>
                    <p>Create a new Google Sheet for storing your field notes data</p>
                </div>

                <div class="option" onclick="loadUserSheets()">
                    <h3>📋 Select from My Sheets</h3>
                    <p>Choose from your existing Google Sheets</p>
                </div>

                <div class="option" onclick="document.getElementById('existing-form').style.display='block'; this.style.display='none'">
                    <h3>🔗 Use Existing Sheet</h3>
                    <p>Use an existing Google Sheet by providing its URL</p>
                </div>

                <div id="existing-form" style="display: none;">
                    <div class="form-group">
                        <label for="sheet-url">Google Sheets URL:</label>
                        <input type="url" id="sheet-url" placeholder="https://docs.google.com/spreadsheets/d/...">
                    </div>
                    <button type="button" class="btn" onclick="useExistingSheet()">Use This Sheet</button>
                    <button type="button" class="btn btn-secondary" onclick="document.getElementById('existing-form').style.display='none'; document.querySelector('.option[onclick*=\"existing-form\"]').style.display='block'">Cancel</button>
                </div>
            </div>

            <div id="user-sheets" style="display: none;">
                <h3>Your Google Sheets</h3>
                <div id="sheets-list" class="sheets-list">
                    <!-- Content will be loaded here by JavaScript -->
                </div>
                <button type="button" class="btn btn-secondary" onclick="showOptions()">Back</button>
            </div>

            <div id="create-form">
                <h3>Create New Google Sheet</h3>
                <div class="form-group">
                    <label for="sheet-name">Sheet Name:</label>
                    <input type="text" id="sheet-name" placeholder="My Field Notes">
                </div>
                <button type="button" class="btn" onclick="createSheet()">Create Sheet</button>
                <button type="button" class="btn btn-secondary" onclick="showOptions()">Back</button>
            </div>

            <div class="back-link">
                <a href="/">&lt; Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

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
async def store_instructions(uid: str, prompt: str):
    """Store user-defined extraction prompt"""
    # Use user-specific storage
    save_user_prompts(uid, prompt)
    return {"status": "ok"}

@app.get("/oauth/start")
async def oauth_start(uid: str):
    """Start Google OAuth2 flow"""

    # https://github.com/googleads/google-ads-python/blob/main/examples/authentication/generate_user_credentials.py

    flow = get_google_flow()
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=uid,  # Store uid in state
        approval_prompt='force'
    )
    return RedirectResponse(authorization_url)

@app.get("/oauth/callback")
async def oauth_callback(code: str, state: str):
    """Handle OAuth callback"""
    uid = state  # state contains the uid
    flow = get_google_flow()

    try:
        flow.fetch_token(code=code)
    except Exception as e:
        print(e)

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
async def receive_transcript(payload: TranscriptPayload, background_tasks: BackgroundTasks):
    """Receive transcript segments and process in background"""
    uid = payload.session_id  # For now, session_id serves as uid

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

    # Add background task
    background_tasks.add_task(process_segments, payload)

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
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
