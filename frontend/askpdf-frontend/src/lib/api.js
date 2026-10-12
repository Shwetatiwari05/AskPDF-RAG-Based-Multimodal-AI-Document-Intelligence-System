// src/lib/api.js
// Thin wrapper around the FastAPI backend (main.py).
// Centralizing fetch calls here means every page talks to the
// backend the same way, and the base URL only lives in one place.
import { supabase } from "./supabase";
const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function getPdfBlobUrl(pdfId) {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const res = await fetch(`${BASE_URL}/documents/${pdfId}/file`, {
    headers: {
      Authorization: `Bearer ${session.access_token}`,
    },
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

async function getThumbnailBlobUrl(pdfId) {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const res = await fetch(`${BASE_URL}/documents/${pdfId}/thumbnail`, {
    headers: {
      Authorization: `Bearer ${session.access_token}`,
    },
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

async function request(path, options = {}) {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const token = session?.access_token;

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      Authorization: token ? `Bearer ${token}` : "",
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    // FastAPI errors are JSON ({"detail": "..."}); surface that message so the
    // UI shows the server's reason instead of a raw JSON blob.
    const body = await res.text();
    let message = body;
    try {
      const parsed = JSON.parse(body);
      message = parsed.detail || parsed.message || body;
    } catch {
      /* not JSON, keep the raw text */
    }
    throw new Error(message || `Request failed (${res.status})`);
  }

  return res.json();
}

export const api = {
  getThumbnailBlobUrl,
  getPdfBlobUrl,
  // ── Documents ────────────────────────────────────────
  uploadPdf: (file, usedFor = "chat", forceOcr = false) => {
    const form = new FormData();
    form.append("file", file);
    form.append("force_ocr", String(forceOcr));
    form.append("used_for", usedFor);
    return request("/upload", { method: "POST", body: form });
  },

  listDocuments: () => request("/documents"),

  deleteDocument: (pdfId) =>
    request(`/documents/${pdfId}`, { method: "DELETE" }),

  thumbnailUrl: (pdfId) => `${BASE_URL}/documents/${pdfId}/thumbnail`,
  fileUrl: (pdfId) => `${BASE_URL}/documents/${pdfId}/file`,

  // ── Chat ─────────────────────────────────────────────
  sendChatMessage: (pdfId, query, sessionId) =>
    request("/chat", {
      method: "POST",
      body: JSON.stringify({ pdf_id: pdfId, query, session_id: sessionId }),
    }),

  getChatHistory: (pdfId) => request(`/chat/history/${pdfId}`),

  clearChatMemory: (sessionId) =>
    request("/chat/clear", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),

  // ── Summarize ────────────────────────────────────────
  summarizeDocument: (pdfId) =>
    request("/summarize", {
      method: "POST",
      body: JSON.stringify({ pdf_id: pdfId }),
    }),

  summarizeText: (text) =>
    request("/summarize", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  getSummarizeHistory: (pdfId) => request(`/summarize/history/${pdfId}`),

  // ── Redact ───────────────────────────────────────────
  redactDocument: (pdfId, redactTypes, customTerms) =>
    request("/redact", {
      method: "POST",
      body: JSON.stringify({
        pdf_id: pdfId,
        redact_types: redactTypes,
        custom_terms: customTerms,
      }),
    }),

  getRedactHistory: (pdfId) => request(`/redact/history/${pdfId}`),

  downloadRedactedUrl: (filename) =>
    `${BASE_URL}/redact/download/${filename}`,

  // ── Voice ────────────────────────────────────────────
  transcribeAudio: (audioBlob, filename = "recording.webm") => {
    const form = new FormData();
    form.append("file", audioBlob, filename);
    return request("/voice/transcribe", { method: "POST", body: form });
  },
};

export default api;