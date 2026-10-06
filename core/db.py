from core.supabase_client import supabase

def insert_document(metadata, user_id):
    return (
        supabase.table("documents")
        .insert({
            "user_id": user_id,
            "pdf_id": metadata["pdf_id"],
            "pdf_name": metadata["pdf_name"],
            "page_count": metadata["page_count"],
            "chunk_count": metadata["chunk_count"],
            "word_count": metadata["word_count"],
            "force_ocr": metadata.get("force_ocr", False),
            "source_path": None,
            "storage_path": metadata.get("storage_path"),
            "used_for": metadata.get("used_for", "chat"),
        })
        .execute()
    )


def get_documents(user_id):
    result = (
        supabase.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data


def get_document(pdf_id, user_id):
    print("SUPABASE =", supabase)
    print("PDF ID =", pdf_id)
    print("USER ID =", user_id)

    try:
        result = (
            supabase.table("documents")
            .select("*")
            .eq("pdf_id", pdf_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        print("RESULT =", result)
        print("RESULT TYPE =", type(result))

        if result is None:
            return None

        return result.data

    except Exception as e:
        print("DB EXCEPTION =", repr(e))
        raise


def delete_document_db(pdf_id, user_id):
    return (
        supabase.table("documents")
        .delete()
        .eq("pdf_id", pdf_id)
        .eq("user_id", user_id)
        .execute()
    )

def append_history(user_id, pdf_id, history_type, content):
    return (
        supabase.table("document_history")
        .insert({
            "user_id": user_id,
            "pdf_id": pdf_id,
            "history_type": history_type,
            "content": content,
        })
        .execute()
    )


def load_history(user_id, pdf_id, history_type):
    result = (
        supabase.table("document_history")
        .select("content")
        .eq("user_id", user_id)
        .eq("pdf_id", pdf_id)
        .eq("history_type", history_type)
        .order("created_at")
        .execute()
    )

    return [row["content"] for row in result.data]


def clear_history(user_id, pdf_id, history_type):
    return (
        supabase.table("document_history")
        .delete()
        .eq("user_id", user_id)
        .eq("pdf_id", pdf_id)
        .eq("history_type", history_type)
        .execute()
    )

def delete_history(pdf_id, user_id):
    return (
        supabase.table("document_history")
        .delete()
        .eq("pdf_id", pdf_id)
        .eq("user_id", user_id)
        .execute()
    )