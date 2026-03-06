'use client'

import { ExternalLink } from 'lucide-react'
import { SourceCitation } from '@/types'
import { getEurLexUrl } from '@/lib/api'
import { RegulationBadge } from './RegulationBadge'
import { cn } from '@/lib/utils'

interface SourceCitationCardProps {
  source: SourceCitation
  index: number
  className?: string
}

export function SourceCitationCard({ source, index, className }: SourceCitationCardProps) {
  const url = getEurLexUrl(source.regulation, source.section_number)

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        'flex items-start gap-3 p-3 rounded-lg border bg-white hover:bg-gray-50',
        'transition-colors group cursor-pointer text-left',
        className,
      )}
    >
      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gray-100 text-gray-500 text-xs flex items-center justify-center font-medium">
        {index + 1}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <RegulationBadge regulation={source.regulation} />
          <span className="text-xs text-gray-500 capitalize">{source.section_type}</span>
          {source.section_number && (
            <span className="text-xs font-medium text-gray-700">#{source.section_number}</span>
          )}
        </div>
        {source.section_title && (
          <p className="text-xs text-gray-600 mt-1 truncate">{source.section_title}</p>
        )}
        <p className="text-xs text-gray-400 mt-0.5">
          Relevance: {(source.score * 100).toFixed(0)}%
        </p>
      </div>
      <ExternalLink className="w-3.5 h-3.5 text-gray-300 group-hover:text-blue-500 flex-shrink-0 mt-0.5 transition-colors" />
    </a>
  )
}
