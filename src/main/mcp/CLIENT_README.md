# Simple MCP Client

A simple Python client for invoking the MCP Document Processing Server.

## Files

- `client.py` - Main client script
- `config.env` - Configuration file for image path and server URL
- `server.py` - MCP server (already exists)

## Usage

1. **Start the MCP Server** (if not already running):
   ```bash
   ./bin/mcp_start.sh
   ```

2. **Configure the image to process**:
   Edit `config.env` and update the `IMAGE_PATH` to point to your desired image:
   ```
   IMAGE_PATH=/path/to/your/image.png
   ```

3. **Run the client**:
   ```bash
   python src/main/mcp/client.py
   ```

## Configuration Options

Edit `config.env` to change:

- `IMAGE_PATH` - Path to the image you want to process
- `SERVER_URL` - MCP server URL (default: http://localhost:3000)

## Available Sample Images

### ID Cards
- `samples/input/idcard_classic/idcard_specimen_validation.png`
- `samples/input/idcard_classic/idcard_classic_Balea.png`
- `samples/input/idcard_classic/idcard_classic_Traian.png`
- `samples/input/idcard_classic/idcard_classic_Superman.jpg`
- `samples/input/idcard_classic/idcard_classic_specimen2.jpeg`

### Driver Licenses
- `samples/input/driverlicence/driverlicence_front_ilie.jpg`
- `samples/input/driverlicence/driverlicence_front_Ionut.jpg`
- `samples/input/driverlicence/driverlicence_front_specimen.png`
- `samples/input/driverlicence/driverlicence_front_Viorel.jpeg`

### Passports
- `samples/input/passport/pasaport_Mihai.jpg`
- `samples/input/passport/pasaport_specimen.jpg`
- `samples/input/passport/passport.png`

## Client Operations

The client performs 4 operations in sequence:

1. **Health Check** - Verifies server is running
2. **Document Classification** - Identifies document type
3. **Field Extraction** - Extracts structured fields
4. **Full Analysis** - Performs OCR and comprehensive analysis

## Example Output

```
============================================================
🚀 Simple MCP Document Processing Client
============================================================
📁 Image: idcard_specimen_validation.png
🌐 Server: http://localhost:3000

🏥 Health Check...
✅ Server is healthy!

🔍 Document Classification...
✅ Document Type: unknown
📊 Confidence: 0.30

📋 Field Extraction...
✅ Extracted 1 field(s):
   • extracted_text: ROMANIA, 2 KARO) 22) yy
     Confidence: 0.50

🔬 Full Document Analysis...
✅ Analysis Results:
   📝 Text Length: 23 characters
   📊 OCR Confidence: 41.80
   📍 Text Regions: 5
   🖼️  Base64 Image: Yes
   📖 Text Preview: ROMANIA, 2 KARO) 22) yy

============================================================
✅ Processing completed!
============================================================
```

## Requirements

- Python 3.7+
- requests library
- Running MCP server on localhost:3000
