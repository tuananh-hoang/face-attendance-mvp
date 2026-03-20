import os

def create_structure():
    base_path = "."

    folders = [
        "core",
        "database",
        "scripts",
        "data/employees",
        "data/embeddings",
        "models",
        "logs"
    ]

    files = [
        "main.py",
        "requirements.txt",
        "README.md",
        "core/__init__.py",
        "core/camera.py",
        "core/face_engine.py",
        "core/matcher.py",
        "core/attendance.py",
        "database/__init__.py",
        "database/db.py",
        "database/schema.sql",
        "scripts/enroll.py",
        "logs/attendance.db"
    ]

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Create files
    for file in files:
        dir_name = os.path.dirname(file)
        if dir_name:  # ⭐ FIX CHÍNH Ở ĐÂY
            os.makedirs(dir_name, exist_ok=True)

        with open(file, "w", encoding="utf-8") as f:
            pass

    print("✅ MVP folder structure created successfully!")

if __name__ == "__main__":
    create_structure()
