'use client'

import { RegulationFilter } from '@/types'
import { cn } from '@/lib/utils'

interface RegulationFilterProps {
  value: RegulationFilter
  onChange: (v: RegulationFilter) => void
  className?: string
}

const OPTIONS: { label: string; value: RegulationFilter; desc: string }[] = [
  { label: 'All Regulations', value: 'all', desc: 'DORA + NIS2' },
  { label: 'DORA', value: 'DORA', desc: 'Digital Operational Resilience' },
  { label: 'NIS2', value: 'NIS2', desc: 'Network & Information Security' },
]

export function RegulationFilterSelector({ value, onChange, className }: RegulationFilterProps) {
  return (
    <div className={cn('flex gap-2', className)}>
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors border',
            value === opt.value
              ? 'bg-blue-600 text-white border-blue-600'
              : 'bg-white text-gray-600 border-gray-200 hover:border-blue-300 hover:text-blue-600',
          )}
          title={opt.desc}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
