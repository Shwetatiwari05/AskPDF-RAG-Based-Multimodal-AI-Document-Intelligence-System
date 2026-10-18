// src/pages/Dashboard.jsx
import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  FileText,
  Search,
  MoreVertical,
  Trash2,
  Loader2,
  ArrowRight,
} from "lucide-react";
import AppShell from "../components/layout/AppShell";
import { useDocuments } from "../context/DocumentContext";
import api from "../lib/api";

const ICON_COLORS = [
  { bg: "bg-accent-100", fg: "text-accent-600" },
  { bg: "bg-coral-bg", fg: "text-coral" },
  { bg: "bg-teal-bg", fg: "text-teal" },
];

// Formats a byte count into a human-readable size (KB/MB), based on
// the real file size returned by the backend — never hardcoded.
function formatFileSize(bytes) {
  if (bytes === null || bytes === undefined) return "Unknown size";
  let val = bytes;
  if (typeof val === "string") {
    val = parseInt(val, 10);
  }
  if (typeof val !== "number" || isNaN(val)) return "Unknown size";
  if (val === 0) return "0 B";
  if (val < 1024) return `${val} B`;
  if (val < 1024 * 1024) return `${(val / 1024).toFixed(0)} KB`;
  return `${(val / (1024 * 1024)).toFixed(1)} MB`;
}

// Turns the backend's "ingested_at" timestamp (e.g. "2026-06-24 06:21")
// into a relative/readable upload date — computed from the real
// timestamp each render, never a fixed string.
function formatUploadDate(ingestedAt) {
  if (!ingestedAt) return "Unknown date";
  const date = new Date(ingestedAt.replace(" ", "T"));
  if (isNaN(date.getTime())) return "Unknown date";

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfDoc = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const diffDays = Math.round((startOfToday - startOfDoc) / 86400000);

  if (diffDays === 0) return "Uploaded today";
  if (diffDays === 1) return "Uploaded yesterday";
  if (diffDays > 1 && diffDays < 7) return `Uploaded ${diffDays} days ago`;
  return `Uploaded ${date.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  })}`;
}

// Maps the backend's "used_for" field to a small badge. Deliberately
// avoids the word "Chat" for the default case (it'd be confusing next to
// the sidebar's "Chat" nav item) — "Asked" reads as the parallel verb to
// "Summarized" / "Redacted".
const USED_FOR_TAGS = {
  summarize: { label: "Summarized", bg: "bg-coral-bg", fg: "text-coral" },
  redact: { label: "Redacted", bg: "bg-teal-bg", fg: "text-teal" },
  chat: { label: "Q&A", bg: "bg-accent-100", fg: "text-accent-700" },
};

function getUsedForTag(usedFor) {
  return USED_FOR_TAGS[usedFor] || USED_FOR_TAGS.chat;
}

export default function Dashboard() {
  const { documents, loadingDocs, refreshDocuments, selectDocument } =
    useDocuments();
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [search, setSearch] = useState("");
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [menuOpenFor, setMenuOpenFor] = useState(null);
  const [thumbFailed, setThumbFailed] = useState({});
  const [thumbUrls, setThumbUrls] = useState({});

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  async function handleFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files are supported.");
      return;
    }
    setUploading(true);
    try {
      await api.uploadPdf(file);
      await refreshDocuments();
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  async function handleDelete(pdfId) {
    setMenuOpenFor(null);
    await api.deleteDocument(pdfId);
    refreshDocuments();
  }

  function openInWorkspace(doc, destination) {
    selectDocument(doc);
    navigate(`/${destination}`);
  }

  useEffect(() => {
  let mounted = true;

  async function loadThumbnails() {
    const urls = {};

    for (const doc of documents) {
      try {
        urls[doc.pdf_id] = await api.getThumbnailBlobUrl(doc.pdf_id);
      } catch {
        // ignore
      }
    }

    if (mounted) {
      setThumbUrls(urls);
    }
  }

  if (documents.length) {
    loadThumbnails();
  }

  return () => {
    mounted = false;
  };
}, [documents]);

  const filtered = documents.filter((d) =>
    d.pdf_name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <AppShell>
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-border-soft bg-surface px-7 py-4">
        <h1 className="flex items-center gap-2.5 font-display text-[17px] font-medium text-text-primary">
          Your documents
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-100 text-accent-600">
            <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </span>
        </h1>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="flex items-center gap-1.5 rounded-[var(--radius-md)] bg-accent-600 px-4 py-2 text-[13px] font-medium text-white transition-colors hover:bg-accent-700 disabled:opacity-60"
        >
          {uploading ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Upload className="h-4 w-4" aria-hidden="true" />
          )}
          Upload PDF
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <div className="p-7">
        {/* Search */}
        <div className="mb-4 flex items-center gap-2.5 rounded-[var(--radius-md)] border border-border-soft bg-surface px-3.5 py-2.5">
          <Search className="h-4 w-4 text-text-secondary" aria-hidden="true" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search your documents..."
            className="w-full bg-transparent text-[13px] text-text-primary placeholder:text-text-secondary focus:outline-none"
          />
        </div>

        {/* Document grid */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className="grid grid-cols-2 gap-3 lg:grid-cols-3"
        >
          {loadingDocs ? (
            <p className="col-span-full py-10 text-center text-[13px] text-text-secondary">
              Loading your documents...
            </p>
          ) : (
            filtered.map((doc, i) => {
              const color = ICON_COLORS[i % ICON_COLORS.length];
              const tag = getUsedForTag(doc.used_for);
              return (
                <div
                  key={doc.pdf_id}
                  onClick={() => openInWorkspace(doc, doc.used_for || "chat")}
                  className="group relative cursor-pointer rounded-[var(--radius-lg)] border border-border-soft bg-surface p-4 transition-shadow hover:shadow-md"
                >
                  <div className="flex items-start justify-between">
                    <span
                      className={`inline-flex items-center rounded-[var(--radius-sm)] px-2 py-0.5 text-[10.5px] font-medium ${tag.bg} ${tag.fg}`}
                    >
                      {tag.label}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenFor(
                          menuOpenFor === doc.pdf_id ? null : doc.pdf_id
                        );
                      }}
                      className="rounded-[var(--radius-sm)] p-1 text-text-secondary opacity-0 transition-opacity hover:bg-surface-sunken group-hover:opacity-100"
                    >
                      <MoreVertical className="h-4 w-4" aria-hidden="true" />
                    </button>

                    {menuOpenFor === doc.pdf_id && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="absolute right-3 top-9 z-10 rounded-[var(--radius-md)] border border-border-soft bg-surface py-1 shadow-md"
                      >
                        <button
                          onClick={() => handleDelete(doc.pdf_id)}
                          className="flex w-full items-center gap-2 px-3 py-2 text-[12px] text-danger hover:bg-danger-bg"
                        >
                          <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                          Delete
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="mt-2.5 h-28 overflow-hidden rounded-[var(--radius-md)] bg-surface-sunken">
                    {thumbFailed[doc.pdf_id] ? (
                      <div className={`flex h-full items-center justify-center ${color.bg}`}>
                        <FileText className={`h-6 w-6 ${color.fg}`} aria-hidden="true" />
                      </div>
                    ) : (
                        <img
                            src={thumbUrls[doc.pdf_id]}
                            alt=""
                            loading="lazy"
                            className="h-full w-full object-cover object-top"
                            onError={() =>
                              setThumbFailed((prev) => ({ ...prev, [doc.pdf_id]: true }))
                            }
                          />
                    )}
                  </div>

                  <p className="mt-2.5 truncate text-[13px] font-medium text-text-primary">
                    {doc.pdf_name}
                  </p>
                  <p className="mt-0.5 text-[11px] text-text-secondary">
                    {formatUploadDate(doc.ingested_at)} &middot;{" "}
                    {formatFileSize(doc.file_size)}
                  </p>
                </div>
              );
            })
          )}

          {/* Drop zone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`flex min-h-[112px] cursor-pointer flex-col items-center justify-center gap-2 rounded-[var(--radius-lg)] border-2 border-dashed p-4 text-center transition-colors ${
              isDragging
                ? "border-accent-600 bg-accent-100"
                : "border-accent-600 bg-accent-50 hover:border-accent-700 hover:bg-accent-100"
            }`}
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-md)] bg-accent-600">
              <Upload className="h-[18px] w-[18px] text-white" aria-hidden="true" />
            </div>
            <div>
              <p className="text-[13px] font-medium text-text-primary">
                Drop a PDF here
              </p>
              <p className="mt-0.5 text-[11px] text-text-secondary">
                or click to browse
              </p>
            </div>
          </div>
        </div>

        {!loadingDocs && filtered.length === 0 && documents.length > 0 && (
          <p className="mt-10 text-center text-[13px] text-text-secondary">
            No documents match "{search}".
          </p>
        )}
      </div>
    </AppShell>
  );
}