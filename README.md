# Omi Notes

Field Notes backend for voice-driven structured data capture. Turns your Omi wearable into a voice-driven reporting assistant for technicians, real-estate agents, contractors, and field engineers.

Instead of juggling clipboards, phones, or spreadsheets, just speak your observations, and Omi instantly structures and logs them into Google Sheets — ready for reports, invoices, or client updates.

## Features

- **Voice-Driven Data Capture**: Process transcript segments from Omi wearable devices
- **AI-Powered Extraction**: Use OpenAI GPT models to extract structured data from natural language
- **Google Sheets Integration**: Automatically append extracted data to Google Sheets
- **User-Specific Storage**: Isolated data storage per user with OAuth authentication
- **Dynamic Schema Generation**: LLM-generated Pydantic models based on user prompts
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Background Processing**: Asynchronous transcript processing for scalability
- **CLI Tools**: Command-line utilities for token management
- **Standalone Demo**: Test the application with a provided OpenAI key on the main page

## Recent Updates

### November 4, 2025
- Added standalone demo on main page with OpenAI key provided by user.
  - How to run standalone demo:
    1. Open with any uid: http://127.0.0.1:8000/oauth/start?uid=123
    2. Submit the form with the same uid: http://127.0.0.1:8000/

## Installation


### Use Omi app

https://h.omi.me/apps/01K8WEY210BJFDD7X2TZ70MVA1

### Prerequisites

- Python 3.13+
- uv package manager
- Google Cloud Console project with OAuth 2.0 credentials
- OpenAI API key

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd omi-notes
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your credentials:
   ```env
   # OpenAI API Configuration
   OPENAI_API_KEY=your_openai_api_key_here

   # Google OAuth Configuration
   GOOGLE_CLIENT_ID=your_google_client_id_here
   GOOGLE_CLIENT_SECRET=your_google_client_secret_here
   GOOGLE_REDIRECT_URI=http://localhost:8000/oauth/callback

   # OMI Key (if needed)
   OMI_KEY=your_omi_key_here
   ```

## Usage

### Running the Application

**Development:**
```bash
uv run uvicorn main:app --reload
```

**Production:**
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Workflow

1. **Setup Google OAuth**: Visit `http://localhost:8000/oauth/start?uid=<user_id>` to authenticate
2. **Configure Extraction Prompt**: Use the web interface or API to set your data extraction instructions
3. **Link Google Sheet**: Provide a Google Sheets URL to store extracted data
4. **Send Transcripts**: POST transcript segments to `/transcript` endpoint
5. **View Results**: Data automatically appears in your linked Google Sheet

### API Endpoints

#### Core Endpoints

- `GET /` - Homepage with setup instructions
- `POST /transcript` - Receive and process transcript segments
- `POST /instructions` - Store user extraction prompts
- `GET /api/instructions` - Retrieve user extraction prompts
- `POST /api/sheet` - Store Google Sheet ID from URL
- `GET /api/sheets` - List user's Google Sheets

#### OAuth Endpoints

- `GET /oauth/start` - Initiate Google OAuth flow
- `GET /oauth/callback` - Handle OAuth callback

#### Web Interface

- `GET /sheets` - Google Sheets selection/creation page
- `GET /privacy` - Privacy policy
- `GET /terms` - Terms of service

### API Usage Examples

#### Store Extraction Instructions
```bash
curl -X POST "http://localhost:8000/instructions" \
  -H "Content-Type: application/json" \
  -d '{
    "uid": "user123",
    "prompt": "Extract item name, quantity, and condition from the transcript."
  }'
```

#### Send Transcript for Processing
```bash
curl -X POST "http://localhost:8000/transcript" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "segments": [
      {"id": "seg1", "text": "Found 5 damaged cables in the server room"},
      {"id": "seg2", "text": "Location is building A, floor 3"}
    ],
    "location": "Building A, Floor 3",
    "datetime": "2024-01-15T10:30:00Z"
  }'
```

#### Link Google Sheet
```bash
curl -X POST "http://localhost:8000/api/sheet" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "sheet_url": "https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit"
  }'
```

### CLI Tools

#### Refresh OAuth Token
```bash
uv run python cli.py refresh-token user123
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM processing | Yes |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | Yes |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | Yes |
| `GOOGLE_REDIRECT_URI` | OAuth callback URL | No (defaults to localhost:8000) |
| `OMI_KEY` | Omi device API key | Optional |

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Sheets API and Google Drive API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs (include your callback URL)
6. Set appropriate scopes in the application

### Data Storage

- User data is stored in the `data/` directory
- Each user gets their own subdirectory: `data/<user_id>/`
- Files stored per user:
  - `prompt.txt` - Extraction instructions
  - `tokens.json` - OAuth credentials
  - `sheet_id.txt` - Linked Google Sheet ID
  - `model_*.py` - Cached Pydantic models

## Deployment

### Using systemd

The project includes systemd service files for production deployment:

1. **Copy service files:**
   ```bash
   sudo cp systemd/ominotes.service /etc/systemd/system/
   sudo cp systemd/ominotes.ini /etc/systemd/system/
   ```

2. **Create user and directories:**
   ```bash
   sudo useradd -r -s /bin/false ominotes
   sudo mkdir -p /home/ominotes/main
   sudo mkdir -p /var/log/ominotes
   sudo mkdir -p /run/ominotes
   sudo chown -R ominotes:ominotes /home/ominotes /var/log/ominotes /run/ominotes
   ```

3. **Copy application and configure environment:**
   ```bash
   sudo cp -r . /home/ominotes/main/
   sudo cp .env /home/ominotes/
   sudo chown ominotes:ominotes /home/ominotes/.env
   ```

4. **Enable and start service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ominotes
   sudo systemctl start ominotes
   ```

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .
RUN pip install uv && uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Architecture

### Components

- **FastAPI Application** (`main.py`): Core web server and API endpoints
- **Data Processing**: Background tasks for transcript processing
- **LLM Integration**: OpenAI GPT models for data extraction
- **Google APIs**: Sheets API for data storage, Drive API for sheet management
- **User Management**: Session-based user isolation with persistent storage
- **CLI Tools** (`cli.py`): Administrative utilities

### Data Flow

1. **Transcript Reception**: API receives transcript segments with metadata
2. **User Context Loading**: Retrieve user's prompt, OAuth tokens, and sheet configuration
3. **LLM Processing**: Generate or load cached Pydantic model based on user prompt
4. **Data Extraction**: Apply LLM chain to extract structured data from transcript
5. **Sheet Integration**: Append extracted data to configured Google Sheet
6. **Token Management**: Refresh OAuth tokens as needed

### Security Considerations

- User data isolation in separate directories
- OAuth 2.0 for Google API access
- Environment variable configuration
- No sensitive data logging
- Background processing for scalability

## Development

### Project Structure

```
omi-notes/
├── main.py                 # FastAPI application
├── cli.py                  # CLI tools
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
├── .env.example          # Environment template
├── data/                 # User data storage
├── templates/            # Jinja2 templates
│   ├── index.html
│   ├── sheets.html
│   ├── privacy.html
│   ├── terms.html
│   └── default_prompt.txt
├── systemd/              # Systemd service files
└── README.md
```

### Adding New Features

1. Define API endpoints in `main.py`
2. Add Pydantic models for request/response data
3. Implement business logic with proper error handling
4. Update user storage functions if needed
5. Add CLI commands in `cli.py` if required
6. Update documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here]