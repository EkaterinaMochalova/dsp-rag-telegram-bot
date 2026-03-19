import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PDF_DIR = Path("kb_pdfs")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Create .env in project root and set OPENAI_API_KEY=...")

    client = OpenAI(api_key=api_key)

    if not PDF_DIR.exists():
        raise RuntimeError(f"Folder not found: {PDF_DIR.resolve()}")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}/")
    for p in pdfs:
        print(" -", p.name)

    if not pdfs:
        raise RuntimeError("No PDFs found in kb_pdfs/. Put your PDFs there and rerun.")

    print("Creating vector store...")
    vs = client.vector_stores.create(name="DSP Client KB")
    vector_store_id = vs.id
    print("VECTOR_STORE_ID =", vector_store_id)

    file_ids = []
    for path in pdfs:
        print("Uploading:", path.name)
        with path.open("rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")
        file_ids.append(file_obj.id)
        print("  file_id:", file_obj.id)

    print("Creating file batch (indexing)...")
    client.vector_stores.file_batches.create(
        vector_store_id=vector_store_id,
        file_ids=file_ids,
    )
    print("Done ✅ Files submitted for indexing.")
    print("Put this into .env:")
    print(f"VECTOR_STORE_ID={vector_store_id}")

if __name__ == "__main__":
    main()
