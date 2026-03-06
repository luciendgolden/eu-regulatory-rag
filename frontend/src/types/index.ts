export interface SourceCitation {
  regulation: string
  section_type: string
  section_number?: string
  section_title?: string
  score: number
}

export interface QueryResponse {
  answer: string
  sources: SourceCitation[]
  query_time_ms: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceCitation[]
  timestamp: Date
  query_time_ms?: number
}

export interface Regulation {
  regulation: string
  article_count: number
  recital_count: number
}

export type RegulationFilter = 'all' | 'DORA' | 'NIS2'

export interface QueryHistoryEntry {
  id: string
  question: string
  answer: string
  sources: SourceCitation[]
  regulation: RegulationFilter
  timestamp: Date
  query_time_ms: number
}
