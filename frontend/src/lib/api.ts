import { QueryResponse, Regulation, RegulationFilter } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function fetchRegulations(): Promise<Regulation[]> {
  const res = await fetch(`${API_URL}/regulations`)
  if (!res.ok) throw new Error('Failed to fetch regulations')
  return res.json()
}

export async function queryRegulations(
  question: string,
  regulation: RegulationFilter = 'all',
  topK: number = 5,
  apiKey?: string,
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    },
    body: JSON.stringify({
      question,
      regulation: regulation === 'all' ? null : regulation,
      top_k: topK,
      stream: false,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Query failed (${res.status})`)
  }
  return res.json()
}

export async function* streamQuery(
  question: string,
  regulation: RegulationFilter = 'all',
  topK: number = 5,
  apiKey?: string,
): AsyncGenerator<{ type: string; content?: string; sources?: unknown; query_time_ms?: number }> {
  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    },
    body: JSON.stringify({
      question,
      regulation: regulation === 'all' ? null : regulation,
      top_k: topK,
      stream: true,
    }),
  })

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Stream failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6))
        } catch {
          // skip malformed
        }
      }
    }
  }
}

export function getEurLexUrl(regulation: string, articleNumber?: string): string {
  const uris: Record<string, string> = {
    DORA: 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554',
    NIS2: 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022L2555',
  }
  return uris[regulation] ?? 'https://eur-lex.europa.eu'
}
