# 🚀 RAG Web Application - Setup Guide

## 📦 What You Downloaded

You should have these files:

```
rag-web-app/
├── simple_server.py           # ⭐ Main web server (USE THIS!)
├── frontend/
│   └── index.html             # Web interface
├── requirements_web.txt       # Python dependencies
├── README_SETUP.txt           # This file
└── diagnose.py               # Diagnostic tool
```

**IMPORTANT:** You also need your existing RAG backend files:
- `fast_rag_universal.py`
- `universal_extractor.py`
- `chunking_structure_aware.py`
- `advanced_retrieval.py`

---

## ⚡ Quick Start (3 Steps!)

### Step 1: Put All Files Together

Put ALL files in ONE folder:

```
my-rag-folder/
├── simple_server.py           # ← From download
├── frontend/
│   └── index.html             # ← From download
├── requirements_web.txt       # ← From download
├── fast_rag_universal.py      # ← Your existing file
├── universal_extractor.py     # ← Your existing file
├── chunking_structure_aware.py # ← Your existing file
├── advanced_retrieval.py      # ← Your existing file
└── (your document files or chroma_universal/ database)
```

### Step 2: Install Dependencies

```bash
pip install flask flask-cors
```

Or use the requirements file:

```bash
pip install -r requirements_web.txt
```

### Step 3: Run It!

```bash
# Go to your folder
cd my-rag-folder

# Set your API key
export GROQ_API_KEY="your-groq-api-key-here"

# Run the server
python3 simple_server.py
```

Then open your browser: **http://localhost:5000**

🎉 Done! You should see the beautiful purple interface!

---

## 📋 Detailed Steps

### Step 1: Check Python Version

```bash
python3 --version
```

Need Python 3.8 or higher.

### Step 2: Install Flask

```bash
pip install flask flask-cors
```

### Step 3: Organize Files

Create a new folder and put everything there:

```bash
# Create folder
mkdir ~/rag-web-app
cd ~/rag-web-app

# Copy downloaded files here
# Copy your existing RAG files here
```

Your folder should look like:

```
~/rag-web-app/
├── simple_server.py
├── frontend/
│   └── index.html
├── fast_rag_universal.py
├── universal_extractor.py
├── chunking_structure_aware.py
├── advanced_retrieval.py
└── chroma_universal/ (if you have existing database)
```

### Step 4: Set API Key

```bash
export GROQ_API_KEY="your-actual-groq-api-key"
```

**Or** add to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export GROQ_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Run Diagnostic (Optional)

```bash
python3 diagnose.py
```

This checks if everything is ready!

### Step 6: Start Server

```bash
python3 simple_server.py
```

You'll see:

```
Initializing RAG system...
✅ Ready! 2257 documents loaded

============================================================
🚀 RAG Web Server Running!
============================================================

   Open: http://localhost:5000

   Press Ctrl+C to stop
```

### Step 7: Open Browser

Open: **http://localhost:5000**

You should see the purple gradient interface! 🎨

---

## 🎯 If You Have Existing Database

If you already have a `chroma_universal` folder with your documents:

```bash
# Just make sure it's in the same folder
ls chroma_universal

# Should show database files
```

The server will automatically use it!

---

## 🎯 If You Need to Add Documents

If you DON'T have a database yet:

**Option 1: Edit simple_server.py**

Find line 20 and add:

```python
# After initializing RAG, add this:
if rag.collection.count() == 0:
    print("Loading documents...")
    rag.add_folder("/path/to/your/documents", recursive=True)
```

**Option 2: Load documents separately first**

Create a script `load_docs.py`:

```python
from fast_rag_universal import FastRAGSystemUniversal
import os

rag = FastRAGSystemUniversal(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    collection_name="universal_rag_production",
    persist_directory="./chroma_universal"
)

# Add your documents
rag.add_folder("/path/to/your/documents", recursive=True)
print(f"✅ Loaded {rag.collection.count()} chunks")
```

Run it once:

```bash
python3 load_docs.py
```

Then start the web server!

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install flask flask-cors
```

### Problem: "ModuleNotFoundError: No module named 'fast_rag_universal'"

**Solution:**
Make sure `fast_rag_universal.py` is in the same folder!

```bash
ls fast_rag_universal.py
# Should show the file
```

### Problem: "GROQ_API_KEY not set"

**Solution:**
```bash
export GROQ_API_KEY="your-key"
```

### Problem: "Port 5000 already in use"

**Solution:**
```bash
# Find and kill the process
lsof -i :5000
kill -9 <PID>

# Or use different port
# Edit simple_server.py, change last line to:
app.run(host='0.0.0.0', port=5001, debug=False)
```

### Problem: "frontend/index.html not found"

**Solution:**
Make sure folder structure is correct:

```bash
ls frontend/index.html
# Should show the file
```

If missing:

```bash
mkdir frontend
# Copy index.html into frontend/ folder
```

### Problem: Browser shows error or blank page

**Solution:**

1. Check server is running (terminal should show "Running!")
2. Check URL is correct: `http://localhost:5000` (not `/api/`)
3. Open browser DevTools (F12), check Console for errors
4. Try: `curl http://localhost:5000` - should return HTML

---

## ✅ Verification Checklist

Before starting server:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Flask installed (`python3 -c "import flask"`)
- [ ] All RAG files in same folder
- [ ] frontend/index.html exists
- [ ] GROQ_API_KEY is set (`echo $GROQ_API_KEY`)
- [ ] In the correct directory (`pwd`)

After starting server:

- [ ] No errors in terminal
- [ ] Shows "Ready! X documents loaded"
- [ ] `curl http://localhost:5000/api/health` works
- [ ] Browser shows purple interface at http://localhost:5000

---

## 🎨 What You Should See

### In Terminal:
```
Initializing RAG system...
✅ Ready! 2257 documents loaded

============================================================
🚀 RAG Web Server Running!
============================================================

   Open: http://localhost:5000

   Press Ctrl+C to stop
```

### In Browser:
```
╔════════════════════════════════════════════╗
║  📚 Texas Dealer RAG System                ║
║  Ask questions about Texas motor vehicle   ║
║                                            ║
║  [Documents: 2257] [Queries: 0] [Time: -]  ║
║                                            ║
║  ┌──────────────────────────────────────┐ ║
║  │ Type your question here...           │ ║
║  │                                      │ ║
║  └──────────────────────────────────────┘ ║
║                                            ║
║  [🔍 Search]  [🗑️ Clear]                  ║
╚════════════════════════════════════════════╝
```

---

## 🚀 Usage

1. **Type a question** or **click an example**
2. Click **Search** (or press Enter)
3. Wait 2-3 seconds
4. See **answer** with **7 sources**
5. Check **relevance scores**
6. Ask more questions!

---

## 📝 Files Included

### simple_server.py
The main web server. This is what you run!

**What it does:**
- Starts Flask web server
- Loads your RAG system
- Serves the web interface
- Handles API requests

**How to use:**
```bash
python3 simple_server.py
```

### frontend/index.html
The web interface (purple gradient design).

**Features:**
- Modern responsive design
- Real-time statistics
- Example questions
- Source highlighting
- Mobile-friendly

**Don't edit unless:** you want to customize colors/text

### requirements_web.txt
Python packages needed.

**Install with:**
```bash
pip install -r requirements_web.txt
```

### diagnose.py
Checks if everything is set up correctly.

**Run with:**
```bash
python3 diagnose.py
```

---

## 🔧 Customization

### Change Port

Edit `simple_server.py`, last line:

```python
# Change 5000 to any port you want
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Change Title

Edit `frontend/index.html`, find:

```html
<h1>
    <span>📚</span>
    Texas Dealer RAG System  <!-- Change this -->
</h1>
```

### Change Colors

Edit `frontend/index.html`, find:

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Replace with:
- Blue: `linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)`
- Green: `linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)`
- Orange: `linear-gradient(135deg, #fa709a 0%, #fee140 100%)`

---

## 🆘 Getting Help

If stuck:

1. **Run diagnostic:**
   ```bash
   python3 diagnose.py
   ```

2. **Check terminal output** for error messages

3. **Check browser console** (F12 → Console tab)

4. **Verify files:**
   ```bash
   ls -la
   ls -la frontend/
   ```

5. **Test API:**
   ```bash
   curl http://localhost:5000/api/health
   ```

---

## 📖 Summary

**To run your RAG web app:**

```bash
# 1. Go to your folder
cd ~/rag-web-app

# 2. Make sure all files are there
ls

# 3. Set API key
export GROQ_API_KEY="your-key"

# 4. Start server
python3 simple_server.py

# 5. Open browser
# http://localhost:5000
```

That's it! 🎉

---

## ✨ Features

Your web app has:

- ✅ Beautiful modern UI (purple gradient)
- ✅ Real-time statistics dashboard
- ✅ Example questions (click to use)
- ✅ Source highlighting with relevance scores
- ✅ Mobile responsive design
- ✅ Fast 2-3 second responses
- ✅ 7 sources per query
- ✅ PDF + TXT file support
- ✅ Hybrid search (semantic + keyword)

Enjoy! 🚀
