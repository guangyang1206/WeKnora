package chatpipeline

import (
	"context"
	"strings"

	"github.com/Tencent/WeKnora/internal/types"
)

// populateFAQAnswers populates FAQ answers for the search results
func (p *PluginMerge) populateFAQAnswers(
	ctx context.Context,
	chatManage *types.ChatManage,
	results []*types.SearchResult,
) []*types.SearchResult {
	if len(results) == 0 || p.chunkRepo == nil {
		return results
	}

	tenantID, _ := types.TenantIDFromContext(ctx)
	if tenantID == 0 && chatManage != nil {
		tenantID = chatManage.TenantID
	}
	if tenantID == 0 {
		pipelineWarn(ctx, "Merge", "faq_enrich_skip", map[string]interface{}{
			"reason": "missing_tenant",
		})
		return results
	}

	chunkResultMap := make(map[string][]*types.SearchResult)
	chunkIDSet := make(map[string]struct{})
	for _, r := range results {
		if r == nil || r.ID == "" {
			continue
		}
		if r.ChunkType != string(types.ChunkTypeFAQ) {
			continue
		}
		chunkResultMap[r.ID] = append(chunkResultMap[r.ID], r)
		if _, exists := chunkIDSet[r.ID]; !exists {
			chunkIDSet[r.ID] = struct{}{}
		}
	}

	if len(chunkIDSet) == 0 {
		return results
	}

	chunkIDs := make([]string, 0, len(chunkIDSet))
	for id := range chunkIDSet {
		chunkIDs = append(chunkIDs, id)
	}

	chunks, err := p.chunkRepo.ListChunksByID(ctx, tenantID, chunkIDs)
	if err != nil {
		pipelineWarn(ctx, "Merge", "faq_chunk_fetch_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return results
	}

	updated := 0
	// Build a set of disabled chunk IDs so we can remove them from results.
	// Fix: previously disabled FAQ entries were still returned because
	// ListChunksByID does NOT filter by is_enabled.
	disabledChunkIDs := make(map[string]struct{})
	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}
		if !chunk.IsEnabled {
			disabledChunkIDs[chunk.ID] = struct{}{}
		}
	}

	// Remove results whose chunks are disabled, so they don't appear
	// in the final answer.
	if len(disabledChunkIDs) > 0 {
		pipelineInfo(ctx, "Merge", "faq_disabled_filtered", map[string]interface{}{
			"count": len(disabledChunkIDs),
		})
	}

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}
		meta, err := chunk.FAQMetadata()
		if err != nil || meta == nil {
			if err != nil {
				pipelineWarn(ctx, "Merge", "faq_metadata_parse_failed", map[string]interface{}{
					"chunk_id": chunk.ID,
					"error":    err.Error(),
				})
			}
			continue
		}
		// Skip disabled FAQ entries — they should not be shown.
		if _, disabled := disabledChunkIDs[chunk.ID]; disabled {
			continue
		}

		content := buildFAQAnswerContent(meta)
		if content == "" {
			continue
		}
		for _, r := range chunkResultMap[chunk.ID] {
			if r == nil {
				continue
			}
			r.Content = content
			updated++
		}
	}

	if updated > 0 {
		pipelineInfo(ctx, "Merge", "faq_content_enriched", map[string]interface{}{
			"chunk_cnt": updated,
		})
	}
	return results
}

// buildFAQAnswerContent builds the content of a FAQ answer
func buildFAQAnswerContent(meta *types.FAQChunkMetadata) string {
	if meta == nil {
		return ""
	}

	question := strings.TrimSpace(meta.StandardQuestion)
	answers := make([]string, 0, len(meta.Answers))
	for _, ans := range meta.Answers {
		if trimmed := strings.TrimSpace(ans); trimmed != "" {
			answers = append(answers, trimmed)
		}
	}

	if question == "" && len(answers) == 0 {
		return ""
	}

	var builder strings.Builder
	if question != "" {
		builder.WriteString("Q: ")
		builder.WriteString(question)
		builder.WriteString("\n")
	}

	if len(answers) > 0 {
		builder.WriteString("Answer:\n")
		for _, ans := range answers {
			builder.WriteString("- ")
			builder.WriteString(ans)
			builder.WriteString("\n")
		}
	}

	return strings.TrimSpace(builder.String())
}
