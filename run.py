from pathlib import Path
import subprocess, sys
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-m','streamlit','run',str(root/'minesafe'/'app.py')],cwd=str(root),check=True)
