'use client'

import { cn } from '@/lib/utils'

interface RegulationBadgeProps {
  regulation: string
  className?: string
}

const COLORS: Record<string, string> = {
  DORA: 'bg-blue-100 text-blue-800 border-blue-200',
  NIS2: 'bg-purple-100 text-purple-800 border-purple-200',
}

export function RegulationBadge({ regulation, className }: RegulationBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border',
        COLORS[regulation] ?? 'bg-gray-100 text-gray-800 border-gray-200',
        className,
      )}
    >
      {regulation}
    </span>
  )
}
