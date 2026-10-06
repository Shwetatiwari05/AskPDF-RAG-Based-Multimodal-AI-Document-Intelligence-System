// src/pages/Chat.jsx
import { useState, useRef, useEffect } from "react";
import { Send, Mic, Square, FileText, RotateCcw, Loader2, Eye, X } from "lucide-react";
import AppShell from "../components/layout/AppShell";
import DocumentPicker from "../components/layout/DocumentPicker";
import { useDocuments } from "../context/DocumentContext";
import api from "../lib/api";
import MarkdownMessage from "../components/MarkdownMessage";

export default function Chat() {
  const { selectedDoc, selectDocument } = useDocuments();
  const [messages, setMessages] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [pdfBlobUrl, setPdfBlobUrl] = useState("");

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const scrollRef = useRef(null);

  // The document's own pdf_id IS the session id — this scopes both the
  // backend's short-term conversation memory and the persisted chat log
  // to this exact document, so switching documents never leaks context
  // from one PDF's conversation into another's.
  const sessionId = selectedDoc?.pdf_id;

  // Load this document's saved chat history every time a different
  // document is opened, so reopening a PDF shows its past conversation
  // instead of starting blank.
  useEffect(() => {
    if (!selectedDoc) return;
    let cancelled = false;
    setLoadingHistory(true);
    api
      .getChatHistory(selectedDoc.pdf_id)
      .then((res) => {
        if (!cancelled) setMessages(res.messages || []);
      })
      .catch(() => {
        if (!cancelled) setMessages([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDoc?.pdf_id]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
  if (!showPdfModal || !selectedDoc) return;

  api
    .getPdfBlobUrl(selectedDoc.pdf_id)
    .then((url) => setPdfBlobUrl(url))
    .catch(console.error);

  return () => {
    if (pdfBlobUrl) {
      URL.revokeObjectURL(pdfBlobUrl);
    }
  };
}, [showPdfModal, selectedDoc]);

  if (!selectedDoc) {
    return (
      <AppShell>
        <DocumentPicker onPick={selectDocument} />
      </AppShell>
    );
  }

  async function handleSend(text) {
    const query = text ?? input;
    if (!query.trim()) return;

    setMessages((m) => [...m, { role: "user", content: query }]);
    setInput("");
    setSending(true);

    try {
      const res = await api.sendChatMessage(
        selectedDoc.pdf_id,
        query,
        sessionId
      );
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Something went wrong: ${err.message}`, error: true },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleClearMemory() {
    await api.clearChatMemory(sessionId);
    setMessages([]);
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      setTranscribing(true);
      try {
        const res = await api.transcribeAudio(blob);
        if (res.text) handleSend(res.text);
      } catch (err) {
        console.error("Transcription failed", err);
      } finally {
        setTranscribing(false);
      }
    };

    recorder.start();
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  }

  return (
    <AppShell>
      <div className="flex h-screen flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border-soft bg-surface px-6 py-3.5">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-accent-600" aria-hidden="true" />
            <span className="text-[13px] font-medium text-text-primary">
              {selectedDoc.pdf_name}
            </span>
            <span className="text-[12px] text-text-secondary">
              &middot; {selectedDoc.page_count} pages
            </span>
            <button
              onClick={() => setShowPdfModal(true)}
              className="ml-1 flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-border-soft px-2.5 py-1 text-[11.5px] font-medium text-text-secondary transition-colors hover:border-accent-500 hover:text-accent-700"
            >
              <Eye className="h-3 w-3" aria-hidden="true" />
              View PDF
            </button>
          </div>
          <button
            onClick={handleClearMemory}
            className="flex items-center gap-1.5 text-[12px] text-text-secondary hover:text-text-primary"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Clear conversation
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {loadingHistory ? (
            <div className="flex items-center justify-center gap-2 py-10 text-[13px] text-text-secondary">
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              Loading conversation...
            </div>
          ) : messages.length === 0 ? (
            <p className="py-10 text-center text-[13px] text-text-secondary">
              Ask anything about {selectedDoc.pdf_name}.
            </p>
          ) : null}

          <div className="flex flex-col gap-4">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
            {sending && (
              <div className="self-start rounded-[var(--radius-lg)] border border-border-soft bg-surface px-4 py-2.5 text-[13px] text-text-secondary">
                Thinking...
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </div>

        {/* Input bar */}
        <div className="flex items-center gap-2.5 border-t border-border-soft bg-surface px-6 py-3.5">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={
              transcribing ? "Transcribing..." : "Ask a question about this document..."
            }
            disabled={transcribing}
            className="flex-1 rounded-[var(--radius-lg)] bg-surface-sunken px-4 py-2.5 text-[13px] text-text-primary placeholder:text-text-secondary focus:outline-none"
          />
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={transcribing}
            aria-label={isRecording ? "Stop recording" : "Record a voice question"}
            className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full transition-colors ${
              isRecording
                ? "bg-danger text-white"
                : "bg-accent-100 text-accent-600 hover:bg-accent-100/70"
            }`}
          >
            {isRecording ? (
              <Square className="h-4 w-4" aria-hidden="true" />
            ) : (
              <Mic className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || sending}
            aria-label="Send message"
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-accent-600 text-white transition-colors hover:bg-accent-700 disabled:opacity-50"
          >
            <Send className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* PDF preview modal */}
      {showPdfModal && (
        <div
          onClick={() => setShowPdfModal(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-6"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="flex h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border-soft bg-surface shadow-md"
          >
            <div className="flex items-center justify-between border-b border-border-soft px-4 py-2.5">
              <span className="truncate text-[13px] font-medium text-text-primary">
                {selectedDoc.pdf_name}
              </span>
              <div className="flex items-center gap-4">
                <a
                  href={pdfBlobUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[12px] font-medium text-accent-600 hover:underline"
                >
                  Open in new tab
                </a>
                <button
                  onClick={() => setShowPdfModal(false)}
                  aria-label="Close"
                  className="text-text-secondary hover:text-text-primary"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            </div>
            <iframe
            src={pdfBlobUrl}
            title={selectedDoc.pdf_name}
            className="flex-1"
          />
          </div>
        </div>
      )}
    </AppShell>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[75%] rounded-[var(--radius-lg)] px-4 py-2.5 text-[13px] leading-[1.6] ${
          isUser
            ? "bg-accent-600 text-white"
            : message.error
            ? "bg-danger-bg text-danger"
            : "border border-border-soft bg-surface text-text-primary"
        }`}
      >
        {isUser ? message.content : <MarkdownMessage content={message.content} />}
      </div>

      {message.sources?.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {message.sources
            .filter((s) => s.page_num || s.page_range)
            .slice(0, 4)
            .map((s, i) => (
              <span
                key={i}
                className="rounded-[var(--radius-sm)] bg-accent-100 px-2 py-0.5 text-[11px] font-medium text-accent-700"
              >
                Page {s.page_range || s.page_num}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}