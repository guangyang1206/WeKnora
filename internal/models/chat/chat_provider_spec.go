package chat

import (
	"context"
	"net/http"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/models/provider"
	modelutils "github.com/Tencent/WeKnora/internal/models/utils"
	"github.com/google/uuid"
	"github.com/sashabaranov/go-openai"
)

// ProviderSpec describes provider-specific behavior for chat completions.
// Each spec is registered with a ProviderName and optionally a model matcher.
type ProviderSpec struct {
	Provider provider.ProviderName
	// ModelMatcher: if non-nil, this spec only applies when the model name matches.
	// Used for sub-provider routing (e.g. Qwen3 within Aliyun).
	ModelMatcher func(modelName string) bool
	// RequestCustomizer: provider-specific request modification.
	RequestCustomizer func(req *openai.ChatCompletionRequest, opts *ChatOptions, isStream bool) (any, bool)
	// EndpointCustomizer: provider-specific endpoint URL override.
	EndpointCustomizer func(baseURL string, modelID string, isStream bool) string
	// HeaderCustomizer: provider-specific raw HTTP header customization.
	HeaderCustomizer func(chat *RemoteAPIChat, req *http.Request, body []byte) error
}

// chatProviderSpecs is the ordered list of provider specs.
// Order matters: more specific specs (with ModelMatcher) should come before generic ones.
var chatProviderSpecs = []ProviderSpec{
	// WeKnoraCloud
	{
		Provider:          provider.ProviderWeKnoraCloud,
		RequestCustomizer: weKnoraCloudRequestCustomizer,
		EndpointCustomizer: func(baseURL string, _ string, _ bool) string {
			return strings.TrimRight(baseURL, "/") + "/api/v1/chat/completions"
		},
		HeaderCustomizer: weKnoraCloudHeaderCustomizer,
	},
	// Aliyun Qwen Thinking Models (must be before generic Aliyun)
	{
		Provider:          provider.ProviderAliyun,
		ModelMatcher:      func(name string) bool { return provider.IsQwenThinkingModel(name) },
		RequestCustomizer: qwenThinkingRequestCustomizer,
	},
	// LKEAP
	{
		Provider:          provider.ProviderLKEAP,
		RequestCustomizer: lkeapRequestCustomizer,
	},
	// DeepSeek
	{
		Provider:          provider.ProviderDeepSeek,
		RequestCustomizer: deepseekRequestCustomizer,
	},
	// Generic (vLLM)
	{
		Provider:          provider.ProviderGeneric,
		RequestCustomizer: genericRequestCustomizer,
	},
	// Volcengine (火山引擎 Ark)
	{
		Provider:          provider.ProviderVolcengine,
		RequestCustomizer: volcengineRequestCustomizer,
	},
	// NVIDIA
	{
		Provider:          provider.ProviderNvidia,
		RequestCustomizer: genericRequestCustomizer,
	},
}

// findProviderSpec finds the matching spec for the given provider and model name.
func findProviderSpec(providerName provider.ProviderName, modelName string) *ProviderSpec {
	for i := range chatProviderSpecs {
		spec := &chatProviderSpecs[i]
		if spec.Provider != providerName {
			continue
		}
		if spec.ModelMatcher != nil && !spec.ModelMatcher(modelName) {
			continue
		}
		return spec
	}
	return nil
}

// --- Type definitions (moved from wrapper files) ---

// QwenChatCompletionRequest Qwen 模型的自定义请求结构体
type QwenChatCompletionRequest struct {
	openai.ChatCompletionRequest
	EnableThinking *bool `json:"enable_thinking,omitempty"`
}

// ThinkingConfig 思维链配置（LKEAP / Volcengine 等通用格式）
type ThinkingConfig struct {
	Type string `json:"type"` // "enabled" 或 "disabled"
}

// ThinkingChatCompletionRequest 带 thinking 字段的自定义请求结构体
// 适用于 LKEAP、Volcengine 等使用 { "thinking": { "type": "enabled" } } 格式的 provider
type ThinkingChatCompletionRequest struct {
	openai.ChatCompletionRequest
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// --- Customizer functions ---

// weKnoraCloudRequestCustomizer 构造 WeKnoraCloud 请求。
// WeKnoraCloud 走 OpenAI 兼容格式，除了 MultiContent 需要降级为纯文本 Content 之外，
// 其他字段（tools / tool_choice / parallel_tool_calls / response_format / stream_options 等）直接透传，
// 以保证 function calling 等能力可用。
func weKnoraCloudRequestCustomizer(req *openai.ChatCompletionRequest, _ *ChatOptions, isStream bool) (any, bool) {
	cloudReq := *req
	cloudReq.Stream = isStream
	cloudReq.Messages = convertToWeKnoraCloudMessagesFromOpenAI(req.Messages)
	return cloudReq, true
}

func weKnoraCloudHeaderCustomizer(chat *RemoteAPIChat, req *http.Request, body []byte) error {
	requestID := uuid.NewString()
	headers := modelutils.Sign(chat.appID, chat.appSecret, requestID, string(body))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return nil
}

// convertToWeKnoraCloudMessagesFromOpenAI 将 MultiContent 降级为纯文本，
// 其它字段（tool_calls / tool_call_id / name 等）完全保留，保证 tool 协议正常。
func convertToWeKnoraCloudMessagesFromOpenAI(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, 0, len(messages))
	for _, m := range messages {
		msg := m
		if msg.Content == "" && len(msg.MultiContent) > 0 {
			var textParts []string
			for _, part := range msg.MultiContent {
				if part.Type == openai.ChatMessagePartTypeText && part.Text != "" {
					textParts = append(textParts, part.Text)
				}
			}
			msg.Content = strings.Join(textParts, "\n")
			msg.MultiContent = nil
		}
		result = append(result, msg)
	}
	return result
}

// qwenThinkingRequestCustomizer 自定义 Qwen 系列（阿里云）模型的思考请求
func qwenThinkingRequestCustomizer(
	req *openai.ChatCompletionRequest, opts *ChatOptions, isStream bool,
) (any, bool) {
	if !isStream {
		// Qwen3 模型在非流式请求时需要显式禁用 thinking
		qwenReq := QwenChatCompletionRequest{
			ChatCompletionRequest: *req,
		}
		enableThinking := false
		qwenReq.EnableThinking = &enableThinking
		return qwenReq, true
	}

	// 流式请求：根据 opts.Thinking 启用思考
	qwenReq := QwenChatCompletionRequest{
		ChatCompletionRequest: *req,
	}
	thinking := false
	if opts != nil && opts.Thinking != nil {
		thinking = *opts.Thinking
	}
	qwenReq.EnableThinking = &thinking

	// 必须返回 true 以使用 raw HTTP，否则 SDK 会过滤掉 enable_thinking 字段
	return qwenReq, true
}

// lkeapRequestCustomizer 自定义 LKEAP 请求
// 仅对 DeepSeek V3.x 系列模型设置 thinking 参数；R1 系列默认开启思维链
// 参考：https://cloud.tencent.com/document/product/1772/115963
func lkeapRequestCustomizer(
	req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool,
) (any, bool) {
	modelName := req.Model
	if !strings.Contains(strings.ToLower(modelName), "deepseek-v3") || opts == nil || opts.Thinking == nil {
		return nil, false
	}

	lkeapReq := ThinkingChatCompletionRequest{
		ChatCompletionRequest: *req,
	}

	thinkingType := "disabled"
	if *opts.Thinking {
		thinkingType = "enabled"
	}
	lkeapReq.Thinking = &ThinkingConfig{Type: thinkingType}

	return lkeapReq, true
}

// deepseekRequestCustomizer 自定义 DeepSeek 请求
// DeepSeek 模型不支持 tool_choice 参数，需要清除
func deepseekRequestCustomizer(
	req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool,
) (any, bool) {
	if opts != nil && opts.ToolChoice != "" {
		logger.Infof(context.Background(), "deepseek model, skip tool_choice")
		req.ToolChoice = nil
	}
	return nil, false
}

// genericRequestCustomizer 自定义 Generic 请求
// Generic provider（如 vLLM）使用 ChatTemplateKwargs 传递 thinking 参数
func genericRequestCustomizer(
	req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool,
) (any, bool) {
	thinking := false
	if opts != nil && opts.Thinking != nil {
		thinking = *opts.Thinking
	}
	req.ChatTemplateKwargs = map[string]interface{}{
		"enable_thinking": thinking,
	}
	return req, true
}

// volcengineRequestCustomizer 自定义火山引擎请求
// 火山引擎使用 thinking 参数控制深度思考，格式同 LKEAP: { "type": "enabled"/"disabled" }
func volcengineRequestCustomizer(
	req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool,
) (any, bool) {
	if opts == nil || opts.Thinking == nil {
		return nil, false
	}

	vcReq := ThinkingChatCompletionRequest{
		ChatCompletionRequest: *req,
	}

	thinkingType := "disabled"
	if *opts.Thinking {
		thinkingType = "enabled"
	}
	vcReq.Thinking = &ThinkingConfig{Type: thinkingType}

	return vcReq, true
}

// --- Local LLM RequestCustomizer ---

// localLLMRequestCustomizer 自定义本地 LLM 请求（vLLM、Xinference、Ollama 等）
// 修复两个与本地推理引擎 Jinja2 模板的兼容性问题：
//   1. `content: null` 或 missing content 导致 Jinja2 TypeError
//      → 确保 content 字段始终存在（nil → JSON null，空字符串 → ""）
//   2. `tool_calls[].function.arguments` 在 OpenAI wire format 中是 JSON 字符串，
//      但 vLLM/Xinference 的 Jinja2 模板用 `.items()` 遍历，期望是 dict
//      → 将 arguments 从 JSON 字符串解析后作为 JSON 对象重新序列化
func localLLMRequestCustomizer(
	req *openai.ChatCompletionRequest, _ *ChatOptions, isStream bool,
) (any, bool) {
	type customToolCallFunction struct {
		Name      string          `json:"name,omitempty"`
		Arguments json.RawMessage `json:"arguments"` // serialized as JSON object, not string
	}
	type customToolCall struct {
		ID       string                   `json:"id,omitempty"`
		Type     string                   `json:"type"`
		Function customToolCallFunction `json:"function"`
	}
	type customMessage struct {
		Role             string            `json:"role"`
		Content          *string           `json:"content"` // pointer: nil → null, non-nil → string
		Name             string            `json:"name,omitempty"`
		ToolCalls        []customToolCall `json:"tool_calls,omitempty"`
		ToolCallID       string            `json:"tool_call_id,omitempty"`
		ReasoningContent string            `json:"reasoning_content,omitempty"`
	}
	type customRequest struct {
		Model             string            `json:"model"`
		Messages          []customMessage   `json:"messages"`
		Stream            bool              `json:"stream,omitempty"`
		Temperature        *float32         `json:"temperature,omitempty"`
		TopP              *float32         `json:"top_p,omitempty"`
		MaxTokens         int               `json:"max_tokens,omitempty"`
		MaxCompletionTokens int               `json:"max_completion_tokens,omitempty"`
		FrequencyPenalty   *float32         `json:"frequency_penalty,omitempty"`
		PresencePenalty    *float32         `json:"presence_penalty,omitempty"`
		Tools             []openai.Tool    `json:"tools,omitempty"`
		ToolChoice        any               `json:"tool_choice,omitempty"`
		ParallelToolCalls  any               `json:"parallel_tool_calls,omitempty"`
	}

	// Convert messages
	customMsgs := make([]customMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		cmsg := customMessage{
			Role:             m.Role,
			Name:             m.Name,
			ToolCallID:       m.ToolCallID,
			ReasoningContent: m.ReasoningContent,
		}

		// Fix content: pointer semantics
		// - assistant with tool_calls + no content → nil (JSON null)
		// - otherwise → point to content string (may be "")
		if m.Content != "" {
			cmsg.Content = &m.Content
		} else if len(m.ToolCalls) > 0 {
			cmsg.Content = nil // JSON null
		} else {
			empty := ""
			cmsg.Content = &empty
		}

		// Fix tool_calls: arguments JSON string → JSON object
		if len(m.ToolCalls) > 0 {
			cmsg.ToolCalls = make([]customToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				var argsObj json.RawMessage
				if tc.Function.Arguments != "" {
					var parsed interface{}
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err == nil {
						argsObj, _ = json.Marshal(parsed)
					} else {
						// not valid JSON: keep as string wrapped in quotes
						argsObj = json.RawMessage(`"` + tc.Function.Arguments + `"`)
					}
				} else {
					argsObj = json.RawMessage("{}")
				}
				cmsg.ToolCalls = append(cmsg.ToolCalls, customToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: customToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: argsObj,
					},
				})
			}
		}

		customMsgs = append(customMsgs, cmsg)
	}

	customReq := customRequest{
		Model:    req.Model,
		Messages: customMsgs,
		Stream:   isStream,
	}
	if req.Temperature != 0 {
		t := req.Temperature
		customReq.Temperature = &t
	}
	if req.TopP != 0 {
		t := req.TopP
		customReq.TopP = &t
	}
	if req.MaxTokens != 0 {
		customReq.MaxTokens = req.MaxTokens
	}
	if req.MaxCompletionTokens != 0 {
		customReq.MaxCompletionTokens = req.MaxCompletionTokens
	}
	if len(req.Tools) > 0 {
		customReq.Tools = req.Tools
	}
	if req.ToolChoice != nil {
		customReq.ToolChoice = req.ToolChoice
	}
	if req.ParallelToolCalls != nil {
		customReq.ParallelToolCalls = req.ParallelToolCalls
	}

	return customReq, true // useRawHTTP = true
}
