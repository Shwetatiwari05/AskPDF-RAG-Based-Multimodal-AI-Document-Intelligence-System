#!/usr/bin/env python3
"""
Backfill missing file_size for documents stored in Supabase.

For each document row where file_size IS NULL:
1. Extract user_id, pdf_id, pdf_name, storage_path from the row.
2. If storage_path is missing, skip.
3. Call Supabase Storage list on the user's folder and match the object
   corresponding to this document (by name/filename in storage_path).
4. Read the object's size metadata (e.g. object.size or contentLength).
5. Update the documents table row for that (user_id, pdf_id) with file_size.

Usage:
  python scripts/backfill_file_size.py [--dry-run] [--limit N] [--bucket pdfs]

Environment:
  Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from environment
  (uses core.supabase_client.supabase).
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.supabase_client import supabase


BUCKET = os.getenv("SUPABASE_PDFS_BUCKET", "pdfs")


def get_filename_from_path(storage_path: str | None) -> str | None:
    if not storage_path:
        return None
    return str(storage_path).split("/")[-1]


def backfill(dry_run: bool = True, limit: int | None = None, bucket: str = BUCKET):
    print(f"Backfill started | dry_run={dry_run} | bucket={bucket} | limit={limit}")
    # Fetch documents with null file_size
    query = (
        supabase.table("documents")
        .select("user_id,pdf_id,pdf_name,storage_path,file_size")
        .is_("file_size", "null")
        .order("created_at", desc=True)
    )
    if limit is not None:
        query = query.limit(limit)
    res = query.execute()
    rows = res.data or []
    if not rows:
        print("No documents found with file_size IS NULL")
        return
    print(f"Found {len(rows)} documents with missing file_size")
    updated = 0
    skipped = 0
    for i, row in enumerate(rows, 1):
        user_id = row.get("user_id")
        pdf_id = row.get("pdf_id")
        storage_path = row.get("storage_path")
        if not user_id or not pdf_id or not storage_path:
            skipped += 1
            print(f"[{i}] SKIP missing keys: {row}")
            continue
        folder = str(user_id)
        try:
            # List objects in user's folder
            listed = supabase.storage.from_(bucket).list(folder)
        except Exception as e:
            skipped += 1
            print(f"[{i}] SKIP list failed for {folder}/{storage_path}: {e}")
            continue
        items = listed if isinstance(listed, list) else (listed.data or [])
        # Find matching object
        fname = get_filename_from_path(storage_path)
        match = None
        for obj in items:
            name = obj.get("name") or obj.get("file_name") or obj.get("path")
            if not name:
                continue
            if fname and name == fname:
                match = obj
                break
            if name == str(storage_path).split("/")[-1]:
                match = obj
                break
            if str(storage_path).endswith(name):
                match = obj
                break
        if match is None:
            skipped += 1
            print(f"[{i}] SKIP no storage match for {storage_path} (items={len(items)})")
            continue
        size = match.get("size") or match.get("contentLength") or match.get("metadata", {}).get("size")
        if size is None:
            skipped += 1
            print(f"[{i}] SKIP size missing in match for {storage_path}: {match}")
            continue
        try:
            size_int = int(size)
        except Exception:
            skipped += 1
            print(f"[{i}] SKIP non-numeric size {size} for {storage_path}")
            continue
        if dry_run:
            print(f"[{i}] DRY-RUN would set file_size={size_int} for pdf_id={pdf_id} (user={user_id})")
            continue
        try:
            supabase.table("documents").update({"file_size": size_int}).eq("user_id", user_id).eq("pdf_id", pdf_id).execute()
            updated += 1
            print(f"[{i}] UPDATED file_size={size_int} for pdf_id={pdf_id}")
        except Exception as e:
            skipped += 1
            print(f"[{i}] SKIP update failed for pdf_id={pdf_id}: {e}")
    print(f"Done | updated={updated} | skipped={skipped} | total={len(rows)}")


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args or "-n" in args
    limit = None
    for a in args:
        if a.startswith("--limit="):
            try:
                limit = int(a.split("=", 1)[1])
            except Exception:
                pass
    backfill(dry_run=dry_run, limit=limit)


if __name__ == "__main__":
    main()
