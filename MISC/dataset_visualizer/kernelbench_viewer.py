from flask import Flask, render_template_string
from datasets import load_dataset

app = Flask(__name__)

# Load dataset once at startup
print("🔄 Loading KernelBench dataset...")
dataset = load_dataset("ScalingIntelligence/KernelBench")
print("✅ Dataset loaded!\n")

# === Print dataset metadata to terminal ===
print("📊 KernelBench Dataset Info:")
for split_name, split_data in dataset.items():
    print(f"  - Split: {split_name}")
    print(f"    - Num samples: {len(split_data)}")
    print(f"    - Fields: {split_data.column_names}")
print()

# === Flask templates ===

INDEX_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>KernelBench Viewer</title>
</head>
<body>
    <h1>KernelBench Dataset</h1>
    <ul>
        {% for level in levels %}
        <li><a href="/level/{{ level }}">{{ level }}</a></li>
        {% endfor %}
    </ul>
</body>
</html>
'''

LEVEL_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>KernelBench - {{ level_name }}</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 2em;
        }
        pre {
            background: #f4f4f4;
            padding: 1em;
            border-radius: 8px;
            white-space: pre-wrap;
        }
        .entry {
            margin-bottom: 2em;
            border-bottom: 1px solid #ccc;
            padding-bottom: 1em;
        }
    </style>
</head>
<body>
    <h1>Level: {{ level_name }}</h1>
    <a href="/">← Back to Home</a>
    <div>
        {% for entry in entries %}
        <div class="entry">
            <h2>{{ entry['name'] }}</h2>
            <p><strong>Problem ID:</strong> {{ entry['problem_id'] }}</p>
            <p><strong>Level:</strong> {{ entry['level'] }}</p>
            <p><strong>Code:</strong></p>
            <pre>{{ entry['code'] }}</pre>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE, levels=dataset.keys())

@app.route("/level/<level_name>")
def show_level(level_name):
    entries = sorted(dataset[level_name], key=lambda x: x["problem_id"])
    return render_template_string(LEVEL_TEMPLATE, level_name=level_name, entries=entries)

if __name__ == "__main__":
    app.run(debug=True)

