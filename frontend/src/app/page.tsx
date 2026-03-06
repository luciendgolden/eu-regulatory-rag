'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Send,
  Download,
  Trash2,
  History,
  X,
  Scale,
  AlertCircle,
} from 'lucide-react'
import { ChatMessage, QueryHistoryEntry, RegulationFilter } from '@/types'
import { streamQuery } from '@/lib/api'
import { generateId } from '@/lib/utils'
import { ChatMessageComponent } from '@/components/ChatMessage'
import { RegulationFilterSelector } from '@/components/RegulationFilter'
import { QueryHistory } from '@/components/QueryHistory'
import { cn } from '@/lib/utils'

const EXAMPLE_QUESTIONS = [
  'What are the key ICT risk management requirements under DORA?',
  'How does NIS2 define essential entities?',
  'What are the incident reporting obligations under DORA?',
  'Which sectors are covered by NIS2?',
]

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [regulation, setRegulation] = useState<RegulationFilter>('all')
  const [isLoading, setIsLoading] = useState(false)
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const [history, setHistory] = useState<QueryHistoryEntry[]>([])
  const [apiKey, setApiKey] = useState('')

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('rag-history')
      if (stored) setHistory(JSON.parse(stored).map((e: QueryHistoryEntry) => ({
        ...e,
        timestamp: new Date(e.timestamp),
      })))
    } catch { /* ignore */ }
  }, [])

  const saveHistory = useCallback((entries: QueryHistoryEntry[]) => {
    setHistory(entries)
    localStorage.setItem('rag-history', JSON.stringify(entries))
  }, [])

  const handleSubmit = async (question: string = input.trim()) => {
    if (!question || isLoading) return
    setInput('')
    setError(null)

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    }

    const assistantId = generateId()
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setIsLoading(true)
    setStreamingId(assistantId)

    let fullContent = ''
    let finalSources: ChatMessage['sources'] = []
    let queryTimeMs = 0

    try {
      for await (const event of streamQuery(question, regulation, 5, apiKey || undefined)) {
        if (event.type === 'token' && event.content) {
          fullContent += event.content
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: fullContent } : m,
            ),
          )
        } else if (event.type === 'sources') {
          finalSources = event.sources as ChatMessage['sources']
        } else if (event.type === 'done') {
          queryTimeMs = (event.query_time_ms as number) ?? 0
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(msg)
      setMessages((prev) => prev.filter((m) => m.id !== assistantId))
      setIsLoading(false)
      setStreamingId(null)
      return
    }

    // Finalize assistant message
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId
          ? { ...m, content: fullContent, sources: finalSources, query_time_ms: queryTimeMs }
          : m,
      ),
    )
    setIsLoading(false)
    setStreamingId(null)

    // Save to history
    const entry: QueryHistoryEntry = {
      id: generateId(),
      question,
      answer: fullContent,
      sources: finalSources ?? [],
      regulation,
      timestamp: new Date(),
      query_time_ms: queryTimeMs,
    }
    saveHistory([...history, entry])
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleExportPDF = async () => {
    const { exportConversationToPDF } = await import('@/lib/pdf')
    await exportConversationToPDF(messages, regulation)
  }

  const handleClearChat = () => {
    setMessages([])
    setError(null)
  }

  const handleHistorySelect = (entry: QueryHistoryEntry) => {
    setShowHistory(false)
    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: entry.question,
      timestamp: entry.timestamp,
    }
    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: entry.answer,
      sources: entry.sources,
      timestamp: entry.timestamp,
      query_time_ms: entry.query_time_ms,
    }
    setMessages([userMsg, assistantMsg])
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="flex-shrink-0 bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-blue-600" />
            <h1 className="font-semibold text-gray-900 text-sm sm:text-base">
              EU Regulatory RAG
            </h1>
            <span className="hidden sm:block text-xs text-gray-400">· DORA & NIS2 Compliance Assistant</span>
          </div>

          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <>
                <button
                  onClick={handleExportPDF}
                  title="Export to PDF"
                  className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                >
                  <Download className="w-4 h-4" />
                </button>
                <button
                  onClick={handleClearChat}
                  title="Clear conversation"
                  className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </>
            )}
            <button
              onClick={() => setShowHistory((s) => !s)}
              title="Query history"
              className={cn(
                'p-2 rounded-lg transition-colors',
                showHistory
                  ? 'bg-blue-100 text-blue-600'
                  : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700',
              )}
            >
              <History className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex flex-1 min-h-0 max-w-4xl mx-auto w-full">
        {/* Main chat area */}
        <div className="flex flex-col flex-1 min-h-0">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6 chat-scroll">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
                <div>
                  <Scale className="w-10 h-10 text-blue-200 mx-auto mb-3" />
                  <h2 className="text-xl font-semibold text-gray-700 mb-1">
                    Ask about EU Regulations
                  </h2>
                  <p className="text-gray-400 text-sm max-w-sm">
                    Get grounded answers from DORA and NIS2 with inline article citations linking to EUR-Lex.
                  </p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleSubmit(q)}
                      className="px-4 py-3 rounded-xl border border-gray-200 bg-white text-sm text-gray-600 hover:border-blue-300 hover:text-blue-600 transition-colors text-left leading-snug"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                {messages.map((msg) => (
                  <ChatMessageComponent
                    key={msg.id}
                    message={msg}
                    isStreaming={msg.id === streamingId}
                  />
                ))}
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input area */}
          <div className="flex-shrink-0 px-4 pb-4">
            <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
              {/* Filter bar */}
              <div className="flex items-center gap-3 px-4 pt-3 pb-2 border-b border-gray-100">
                <span className="text-xs text-gray-400 font-medium">Filter:</span>
                <RegulationFilterSelector value={regulation} onChange={setRegulation} />
                <div className="flex-1" />
                <input
                  type="password"
                  placeholder="API Key (optional)"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="text-xs border border-gray-200 rounded-lg px-2 py-1 w-32 focus:outline-none focus:border-blue-300"
                />
              </div>

              {/* Text input */}
              <div className="flex items-end gap-2 px-4 py-3">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about DORA, NIS2, ICT risk management, incident reporting…"
                  rows={1}
                  className="flex-1 resize-none focus:outline-none text-sm text-gray-800 placeholder-gray-400 max-h-32 leading-relaxed"
                  style={{ minHeight: '24px' }}
                  disabled={isLoading}
                />
                <button
                  onClick={() => handleSubmit()}
                  disabled={!input.trim() || isLoading}
                  className={cn(
                    'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors',
                    input.trim() && !isLoading
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-100 text-gray-400',
                  )}
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <p className="text-center text-xs text-gray-400 mt-2">
              Answers grounded in official EUR-Lex sources · Press Enter to send
            </p>
          </div>
        </div>

        {/* History sidebar */}
        {showHistory && (
          <div className="flex-shrink-0 w-72 border-l border-gray-200 bg-white flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h3 className="text-sm font-medium text-gray-700">Query History</h3>
              <button
                onClick={() => setShowHistory(false)}
                className="p-1 rounded hover:bg-gray-100 text-gray-400"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              <QueryHistory history={history} onSelect={handleHistorySelect} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
