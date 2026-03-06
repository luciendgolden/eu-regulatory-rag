'use client'

import { Clock, ChevronRight } from 'lucide-react'
import { QueryHistoryEntry, RegulationFilter } from '@/types'
import { RegulationBadge } from './RegulationBadge'
import { formatDate } from '@/lib/utils'
import { cn } from '@/lib/utils'

interface QueryHistoryProps {
  history: QueryHistoryEntry[]
  onSelect: (entry: QueryHistoryEntry) => void
  className?: string
}

export function QueryHistory({ history, onSelect, className }: QueryHistoryProps) {
  if (history.length === 0) {
    return (
      <div className={cn('text-center py-8 text-gray-400', className)}>
        <Clock className="w-6 h-6 mx-auto mb-2 opacity-40" />
        <p className="text-sm">No history yet</p>
      </div>
    )
  }

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      {history.slice().reverse().map((entry) => (
        <button
          key={entry.id}
          onClick={() => onSelect(entry)}
          className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 text-left group transition-colors border border-transparent hover:border-gray-200"
        >
          <Clock className="w-3.5 h-3.5 text-gray-400 mt-0.5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              {entry.regulation !== 'all' && (
                <RegulationBadge regulation={entry.regulation} />
              )}
              <span className="text-xs text-gray-400">{formatDate(entry.timestamp)}</span>
            </div>
            <p className="text-sm text-gray-700 truncate font-medium">{entry.question}</p>
            <p className="text-xs text-gray-500 truncate mt-0.5">{entry.answer.slice(0, 100)}…</p>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-gray-300 group-hover:text-gray-500 flex-shrink-0 mt-0.5 transition-colors" />
        </button>
      ))}
    </div>
  )
}
