import { ChatMessage, RegulationFilter } from '@/types'
import { formatDate } from './utils'

export async function exportConversationToPDF(
  messages: ChatMessage[],
  regulation: RegulationFilter,
): Promise<void> {
  // Dynamic import to avoid SSR issues
  const { default: jsPDF } = await import('jspdf')

  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
  const pageWidth = doc.internal.pageSize.getWidth()
  const margin = 15
  const contentWidth = pageWidth - margin * 2
  let y = margin

  // Header
  doc.setFontSize(18)
  doc.setFont('helvetica', 'bold')
  doc.text('EU Regulatory RAG — Conversation Export', margin, y)
  y += 8

  doc.setFontSize(10)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(100)
  doc.text(
    `Regulation filter: ${regulation} | Exported: ${formatDate(new Date())}`,
    margin,
    y,
  )
  y += 8

  doc.setDrawColor(200)
  doc.line(margin, y, pageWidth - margin, y)
  y += 6

  doc.setTextColor(0)

  for (const msg of messages) {
    if (y > 260) {
      doc.addPage()
      y = margin
    }

    // Role label
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(msg.role === 'user' ? 30 : 0)
    doc.text(
      `${msg.role === 'user' ? 'You' : 'Assistant'} — ${formatDate(msg.timestamp)}`,
      margin,
      y,
    )
    y += 5

    // Message body
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(40)
    const lines = doc.splitTextToSize(msg.content, contentWidth)
    for (const line of lines) {
      if (y > 270) { doc.addPage(); y = margin }
      doc.text(line, margin, y)
      y += 5
    }

    // Sources
    if (msg.sources && msg.sources.length > 0) {
      y += 2
      doc.setFontSize(8)
      doc.setFont('helvetica', 'italic')
      doc.setTextColor(80)
      doc.text('Sources:', margin, y)
      y += 4
      for (const src of msg.sources) {
        if (y > 270) { doc.addPage(); y = margin }
        const label = `• ${src.regulation} ${src.section_type}${src.section_number ? ` ${src.section_number}` : ''}${src.section_title ? ` — ${src.section_title}` : ''} (score: ${src.score.toFixed(2)})`
        const srcLines = doc.splitTextToSize(label, contentWidth - 4)
        for (const sl of srcLines) {
          if (y > 270) { doc.addPage(); y = margin }
          doc.text(sl, margin + 2, y)
          y += 4
        }
      }
    }

    y += 6
  }

  doc.save(`eu-regulatory-rag-${new Date().toISOString().slice(0, 10)}.pdf`)
}
