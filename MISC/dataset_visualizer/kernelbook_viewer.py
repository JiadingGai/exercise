from flask import Flask, render_template_string, redirect, url_for, request
from datasets import load_dataset
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

app = Flask(__name__)

# === Load KernelBook dataset ===
print("🔄 Loading GPUMODE/KernelBook...")
ds = load_dataset("GPUMODE/KernelBook")
all_entries = list(ds["train"])
print(f"✅ Loaded {len(all_entries)} total entries.")

# === Filter for valid entries ===
def is_valid(entry):
    return all(
        isinstance(entry.get(f), str) and entry[f].strip()
        for f in ["python_code", "triton_code", "repo_name"]
    )

valid_entries = [e for e in all_entries if is_valid(e)]
total = len(valid_entries)
print(f"✅ Found {total} valid entries with usable code.")

# === Syntax highlighting ===
def highlight_code(code):
    return highlight(code, PythonLexer(), HtmlFormatter(nowrap=True))

formatter_styles = HtmlFormatter().get_style_defs('.highlight')

# === HTML Template with meta formatting and coloring ===
HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>KernelBook Viewer - Entry {{ index + 1 }}</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        {{ formatter_styles }}
        .entry { margin-bottom: 4em; border-bottom: 2px solid #ccc; padding-bottom: 2em; }
        .meta {
            font-size: 0.9em;
            color: #444;
            margin-bottom: 1em;
        }
        .key { font-weight: bold; color: #005a9c; }
        .value { color: #222; }
        .nav {
            margin: 1em 0;
        }
        a {
            text-decoration: none;
            margin: 1em;
            font-weight: bold;
        }
        form {
            display: inline-block;
            margin-left: 1em;
        }
        input[type="number"] {
            width: 60px;
            padding: 0.2em;
            font-size: 0.9em;
        }
        input[type="submit"] {
            font-size: 0.9em;
            padding: 0.3em 0.6em;
        }
        pre {
            background: #f4f4f4;
            padding: 1em;
            border-radius: 8px;
            font-size: 0.9em;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>KernelBook Viewer</h1>
    <p>Entry {{ index + 1 }} of {{ total }}</p>

    <div class="nav">
        {% if index > 0 %}
            <a href="{{ url_for('show_entry', index=index-1) }}">&laquo; Prev</a>
        {% endif %}
        {% if index < total - 1 %}
            <a href="{{ url_for('show_entry', index=index+1) }}">Next &raquo;</a>
        {% endif %}
        <form method="get" action="{{ url_for('jump_to') }}">
            <label for="target">Jump to:</label>
            <input type="number" name="target" min="1" max="{{ total }}" required>
            <input type="submit" value="Go">
        </form>
    </div>

    <div class="entry">
        <p class="meta">
            <strong>Meta:</strong>
            <span class="key">repo_name:</span> <span class="value">{{ entry.get("repo_name", "N/A") }}</span> |
            <span class="key">EntryPoint:</span> <span class="value">{{ entry.get("module_name", "N/A") }}</span> |
            <span class="key">Synthetic:</span> <span class="value">{{ entry.get("synthetic", "N/A") }}</span> |
            <span class="key">UUID:</span> <span class="value">{{ entry.get("uuid", "N/A") }}</span> |
            <span class="key">License(s):</span> <span class="value">{{ entry.get("licenses", []) }}</span> |
            <span class="key">⭐</span> <span class="value">{{ entry.get("stars", 0) }}</span> |
            <span class="key">SHA:</span> <span class="value">{{ entry.get("sha", "")[:7] }}</span> |
            <a href="{{ entry.get('repo_link', '#') }}" target="_blank">Repo Link</a>
        </p>

        <p><strong>PyTorch Code:</strong></p>
        <pre class="highlight">{{ highlighted_python|safe }}</pre>

        <p><strong>Original Triton Code (Torch-Inductor):</strong></p>
        <pre class="highlight">{{ highlighted_original_triton|safe }}</pre>

        <p><strong>LLM-Rewritten Triton Code:</strong></p>
        <pre class="highlight">{{ highlighted_triton|safe }}</pre>
    </div>

    <div class="nav">
        {% if index > 0 %}
            <a href="{{ url_for('show_entry', index=index-1) }}">&laquo; Prev</a>
        {% endif %}
        {% if index < total - 1 %}
            <a href="{{ url_for('show_entry', index=index+1) }}">Next &raquo;</a>
        {% endif %}
        <form method="get" action="{{ url_for('jump_to') }}">
            <label for="target">Jump to:</label>
            <input type="number" name="target" min="1" max="{{ total }}" required>
            <input type="submit" value="Go">
        </form>
    </div>
</body>
</html>
'''

@app.route("/")
def home():
    return redirect(url_for("show_entry", index=0))

@app.route("/entry/<int:index>")
def show_entry(index):
    if index < 0 or index >= total:
        return f"Invalid index {index}", 404
    entry = valid_entries[index]
    return render_template_string(
        HTML_TEMPLATE,
        entry=entry,
        index=index,
        total=total,
        highlighted_python=highlight_code(entry["python_code"]),
        highlighted_triton=highlight_code(entry["triton_code"]),
        highlighted_original_triton=highlight_code(entry.get("original_triton_code", "")),
        formatter_styles=formatter_styles
    )

@app.route("/jump")
def jump_to():
    try:
        target = int(request.args.get("target", "1")) - 1
    except ValueError:
        return redirect(url_for("show_entry", index=0))
    if 0 <= target < total:
        return redirect(url_for("show_entry", index=target))
    return f"Invalid entry number {target + 1}", 400

if __name__ == "__main__":
    app.run(debug=True)

