'use client'

import { User, Bot, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import { ChatMessage as ChatMessageType } from '@/types'
import { SourceCitationCard } from './SourceCitationCard'
import { cn, formatDate } from '@/lib/utils'

interface ChatMessageProps {
  message: ChatMessageType
  isStreaming?: boolean
}

export function ChatMessageComponent({ message, isStreaming }: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-blue-600' : 'bg-gray-100 border border-gray-200',
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {/* Bubble */}
      <div className={cn('flex flex-col gap-1 max-w-[80%]', isUser ? 'items-end' : 'items-start')}>
        <div
          className={cn(
            'px-4 py-3 rounded-2xl text-sm leading-relaxed',
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm'
              : 'bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm',
            isStreaming && !isUser && 'streaming-cursor',
          )}
        >
          {message.content || <span className="text-gray-400 italic">Thinking…</span>}
        </div>

        {/* Meta: timestamp + query time */}
        <div className={cn('flex items-center gap-2 text-xs text-gray-400', isUser ? 'flex-row-reverse' : 'flex-row')}>
          <span>{formatDate(message.timestamp)}</span>
          {message.query_time_ms !== undefined && (
            <span>· {(message.query_time_ms / 1000).toFixed(1)}s</span>
          )}
        </div>

        {/* Sources toggle */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setShowSources((s) => !s)}
              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 font-medium"
            >
              {showSources ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
            </button>
            {showSources && (
              <div className="mt-2 flex flex-col gap-1.5">
                {message.sources.map((src, i) => (
                  <SourceCitationCard key={i} source={src} index={i} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
