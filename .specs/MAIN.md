# 📋 Field Notes — Backend Spec for Kiro Code Agent

## Overview
Create a lightweight backend for **Field Notes**, a voice-driven structured data capture tool for Omi wearables.  
Users dictate notes or measurements; the backend extracts structured data and writes it to Google Sheets.

---

## Tech Stack

| Component | Requirement |
|------------|--------------|
| Language | Python 3.13 |
| Framework | FastAPI |
| AI Framework | LangChain v1 |
| LLM | OpenAI (JSON mode or function calling) |
| Google Integration | Google Sheets API with OAuth2 |
| Task Runner | FastAPI BackgroundTasks |
| Server | uvicorn |
| No | Queues, API keys, or Redis |
| OAuth Redirect | `http://localhost:8000/oauth/callback` |

---

## Core Concept

Each user (identified by `session_id`) provides:
1. **Custom instructions** (the extraction prompt).
2. **Realtime transcripts** (streamed speech-to-text segments).
3. **Google Sheets integration** (via OAuth).

System combines those to append structured rows to Google Sheets.

---

## API Endpoints

### `POST /instructions`
Store user-defined extraction prompt.

**Request Body**
```json
{
  "session_id": "SzDL5D2J1VMxn4JfIpDkxBo56EB3",
  "prompt": "Write down all my measurements: item, width, height"
}
Behavior

Save the prompt in an in-memory dictionary.

Use session_id as key.

No database required.

Response

{ "status": "ok" }
GET /oauth/start

Starts Google OAuth2 flow for Sheets API.

Behavior

Redirects to Google consent screen using environment variables:

GOOGLE_CLIENT_ID

GOOGLE_REDIRECT_URI

Scope: https://www.googleapis.com/auth/spreadsheets

GET /oauth/callback

Handles redirect from Google after user authorizes.

Behavior

Exchanges authorization code for tokens.

Stores tokens in memory under the user’s session_id.

Returns confirmation.

Response
{
  "status": "connected",
  "sheet_id": "<sheet_id>"
}

POST /transcript

Receives transcript segments for a session.

Request Example
{
  "session_id": "SzDL5D2J1VMxn4JfIpDkxBo56EB3",
  "segments": [
    {
      "id": "seg1",
      "text": "Living room window: width 1.8 meters, height 1.2 meters."
    },
    {
      "id": "seg2",
      "text": "Kitchen door: width 0.9 meters, height 2.0 meters."
    }
  ],
  "location": "Pine Street 27",
  "datetime": "2025-10-31T09:33:00Z"
}

Behavior

Looks up stored prompt for session_id.

For each segment:

Send prompt + transcript text to OpenAI model using LangChain agent.

Expect JSON response with structured data.

Flatten results into rows.

Append to Google Sheets:

Columns: location, datetime, and extracted fields.

Run sheet append in background task (BackgroundTasks).

Response
{ "status": "accepted", "segments_received": 2 }

LLM Agent

Prompt Template
You are a field notes assistant.
You receive user instructions and a text segment.
Return structured JSON with fields mentioned in the instruction.
Do not explain or comment — output valid JSON only.

INSTRUCTION:
{user_prompt}

TRANSCRIPT:
{segment_text}

Google Sheets Logic

Each user connects their Sheet via OAuth.
System stores:

Access tokens

Target Sheet ID

Use spreadsheets.values.append to write rows:
service.spreadsheets().values().append(
    spreadsheetId=sheet_id,
    range="Sheet1!A1",
    valueInputOption="RAW",
    body={"values": rows}
).execute()

Each row format:
[location, datetime, field1, field2, field3, ...]

Data Models (Pydantic)
from pydantic import BaseModel
from typing import List, Optional

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

Internal Memory Stores

Use simple Python dicts:
PROMPTS = {}       # session_id → user prompt
TOKENS = {}        # session_id → Google credentials
SHEET_IDS = {}     # session_id → target sheet ID

No persistence or duplicate handling required.
Background Task

For each /transcript request:

Schedule a background job to call process_segments(payload).

Job should:

Load user prompt.

Loop through each transcript segment.

Call OpenAI model for extraction.

Append structured data to Google Sheets.

Example Flow

User: sets prompt
POST /instructions

User: connects Sheets
GET /oauth/start → GET /oauth/callback

Omi sends transcript
POST /transcript

FastAPI spawns background task
→ LangChain + OpenAI extracts structured data
→ Append rows to Sheet

User checks their Google Sheet for live updates.

Example Output Row
location	datetime	item	width	height
Pine Street 27	2025-10-31T09:33:00Z	Living room window	1.8	1.2
Pine Street 27	2025-10-31T09:33:00Z	Kitchen door	0.9	2.0
Environment Variables
OPENAI_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/oauth/callback

Success Criteria

✅ Functional FastAPI app with /instructions, /oauth/*, /transcript
✅ Background processing using BackgroundTasks
✅ Uses LangChain agent + OpenAI for structured extraction
✅ Writes rows to Google Sheets via OAuth2
✅ No auth or deduplication
✅ Demo-ready for hackathon use
