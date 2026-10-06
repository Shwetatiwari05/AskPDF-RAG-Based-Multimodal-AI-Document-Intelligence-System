// MarkdownMessage.jsx
// Renders LLM answers (markdown) as clean, professional UI.
// Uses: react-markdown, remark-gfm, lucide-react

import { Children, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";

const ACCENT = "#534AB7";
const BORDER = "#E4E0F7";
const SOFT = "#F5F3FF";

function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // clipboard can be blocked on non-https origins; fail silently
    }
  };

  return (
    <div
      className="my-3 overflow-hidden rounded-xl border bg-white"
      style={{ borderColor: BORDER }}
    >
      <div
        className="flex items-center justify-between px-3 py-1.5 text-xs"
        style={{ backgroundColor: SOFT, color: ACCENT }}
      >
        <span className="font-medium uppercase tracking-wide">
          {language || "code"}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 rounded-md px-2 py-1 font-medium transition-colors hover:bg-white"
          aria-label="Copy code"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="m-0 overflow-x-auto px-4 py-3 text-[13px] leading-relaxed text-slate-800">
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  );
}

const components = {
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,

  strong: ({ children }) => (
    <strong className="font-semibold text-slate-900">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,

  h1: ({ children }) => (
    <h1 className="mb-2 mt-4 text-lg font-semibold text-slate-900 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-4 text-base font-semibold text-slate-900 first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3
      className="mb-1.5 mt-3 text-[15px] font-semibold first:mt-0"
      style={{ color: ACCENT }}
    >
      {children}
    </h3>
  ),

  ul: ({ children }) => (
    <ul className="my-2 list-disc space-y-1 pl-5 marker:text-[#534AB7]">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 list-decimal space-y-1 pl-5 marker:font-medium marker:text-[#534AB7]">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,

  blockquote: ({ children }) => (
    <blockquote
      className="my-3 rounded-r-lg border-l-4 px-4 py-2 text-slate-700"
      style={{ borderColor: ACCENT, backgroundColor: SOFT }}
    >
      {children}
    </blockquote>
  ),

  hr: () => <hr className="my-4" style={{ borderColor: BORDER }} />,

  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-2"
      style={{ color: ACCENT }}
    >
      {children}
    </a>
  ),

  table: ({ children }) => (
    <div
      className="my-3 overflow-x-auto rounded-xl border"
      style={{ borderColor: BORDER }}
    >
      <table className="w-full border-collapse text-left text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead style={{ backgroundColor: SOFT, color: ACCENT }}>{children}</thead>
  ),
  th: ({ children }) => (
    <th
      className="border-b px-3 py-2 font-semibold"
      style={{ borderColor: BORDER }}
    >
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td
      className="border-b px-3 py-2 align-top last:border-b-0"
      style={{ borderColor: BORDER }}
    >
      {children}
    </td>
  ),

  // Inline code only. Fenced blocks are handled in `pre` below.
  code: ({ children }) => (
    <code
      className="rounded-md px-1.5 py-0.5 font-mono text-[13px]"
      style={{ backgroundColor: "#F1EFFC", color: ACCENT }}
    >
      {children}
    </code>
  ),

  pre: ({ children }) => {
    const child = Children.toArray(children)[0];
    const className = child?.props?.className || "";
    const language = /language-([\w-]+)/.exec(className)?.[1];
    const code = String(child?.props?.children ?? "").replace(/\n$/, "");
    return <CodeBlock language={language} code={code} />;
  },
};

export default function MarkdownMessage({ content }) {
  return (
    <div className="break-words text-[15px] leading-relaxed text-slate-800">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}
