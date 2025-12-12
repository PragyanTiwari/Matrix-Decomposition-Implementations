import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def discover_marimo_files(directory):
    """Discover all .py marimo files in a directory"""
    files = []
    dir_path = Path(directory)
    if dir_path.exists():
        for file in dir_path.glob("*.py"):
            files.append(file.stem)  # filename without extension
    return files

def create_card_data(filename, file_type):
    """Create card data for a marimo file"""
    # Convert filename to title (e.g., "gram_schmidt_process" -> "Gram Schmidt Process")
    title = filename.replace("_", " ").title()
    
    card = {
        "title": title,
        "description": f"Interactive {file_type} for {title.lower()}",
        "image_url": "/placeholder.svg?height=160&width=320",
        "tags": [],
        "source_link": f"notebooks/{filename}.html",
        "date": "2025-08-15",
        "type": file_type
    }
    
    # Check if there's a corresponding app
    app_path = Path(f"apps/{filename}.py")
    if app_path.exists():
        card["app_link"] = f"apps/{filename}.html"
    
    return card

# Discover source notebooks and apps
notebook_files = discover_marimo_files("notebooks")
app_files = discover_marimo_files("apps")

print(f"📁 Found {len(notebook_files)} source notebooks in notebooks/")
print(f"📁 Found {len(app_files)} source apps in apps/")

# Load existing cards.json for metadata (descriptions, tags, etc.)
cards_metadata = {}
try:
    with open("templates/cards.json", "r", encoding="utf-8") as f:
        cards_data = json.load(f)
        for card in cards_data:
            # Extract filename from source_link
            source_link = card.get("source_link", "")
            if source_link:
                filename = source_link.split("/")[-1].replace(".html", "")
                cards_metadata[filename] = card
except FileNotFoundError:
    print("Warning: templates/cards.json not found, using default metadata")

# Create card data for all notebooks
notebooks = []
for filename in notebook_files:
    card = create_card_data(filename, "notebook")
    
    # Merge with metadata from cards.json if available
    if filename in cards_metadata:
        metadata = cards_metadata[filename]
        card.update({
            "title": metadata.get("title", card["title"]),
            "description": metadata.get("description", card["description"]),
            "tags": metadata.get("tags", []),
            "date": metadata.get("date", card["date"]),
        })
    
    notebooks.append(card)

# Create card data for standalone apps (apps without corresponding notebooks)
apps = []
for filename in app_files:
    if filename not in notebook_files:  # Only add if not already in notebooks
        card = create_card_data(filename, "app")
        
        # Merge with metadata from cards.json if available
        if filename in cards_metadata:
            metadata = cards_metadata[filename]
            card.update({
                "title": metadata.get("title", card["title"]),
                "description": metadata.get("description", card["description"]),
                "tags": metadata.get("tags", []),
                "date": metadata.get("date", card["date"]),
            })
        
        apps.append(card)

# Prepare template data
template_data = {
    "notebooks": notebooks,
    "apps": apps
}

# Render template
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("index.html.j2")

html_output = template.render(**template_data)

with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("\n✅ Rendered file saved as output.html")
print(f"  - {len(notebooks)} notebook(s)")
print(f"  - {len(apps)} app(s)")
print("\nDiscovered files:")
for nb in notebooks:
    print(f"  📓 {nb['title']}")
    if nb.get('app_link'):
        print(f"     ↳ Has app version")
        
print("\n💡 Open output.html in your browser to preview the changes!")


