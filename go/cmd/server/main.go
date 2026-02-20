package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"microgpt-go/pkg/model"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
	TopP        float64       `json:"top_p"`
	Stream      bool          `json:"stream"`
}

type ChatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Message      ChatMessage `json:"message"`
		Index        int         `json:"index"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

var (
	gpt       func(tokenID, posID int, keys, values [][][]*model.Value) []*model.Value
	tokenizer model.TokenizerRuntime
	config    model.TrainingCheckpointConfig
	state     map[string][][]*model.Value
)

func initModel() {
	ckptPath := os.Getenv("MODEL_PATH")
	if ckptPath == "" {
		ckptPath = "models/latest_checkpoint.json"
	}
	log.Printf("Loading model from %s...", ckptPath)
	ckpt, err := model.LoadCheckpoint(ckptPath)
	if err != nil {
		log.Fatalf("Failed to load checkpoint: %v", err)
	}

	tokenizer, err = model.TokenizerFromCheckpoint(ckpt)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	state = model.ImportState(ckpt.State)
	config = ckpt.Config
	gpt = model.BuildGPT(state, config.NLayer, config.NEmbd, config.NHead)
	log.Println("Model loaded successfully.")
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Temperature <= 0 {
		req.Temperature = 0.5
	}
	if req.TopP <= 0 {
		req.TopP = 0.9
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}

	// Simple prompt construction from messages
	var promptBuilder strings.Builder
	for _, msg := range req.Messages {
		role := "User"
		if msg.Role == "assistant" {
			role = "Assistant"
		}
		fmt.Fprintf(&promptBuilder, "%s: %s\n", role, msg.Content)
	}
	promptBuilder.WriteString("Assistant: ")
	promptText := promptBuilder.String()

	promptTokens := tokenizer.EncodeDoc(promptText)
	if len(promptTokens) > config.BlockSize-1 {
		promptTokens = promptTokens[len(promptTokens)-(config.BlockSize-1):]
	}

	keys := make([][][]*model.Value, config.NLayer)
	values := make([][][]*model.Value, config.NLayer)
	tokenID := tokenizer.BosID
	pos := 0

	// Process prompt tokens (pre-fill KV cache)
	for _, nextID := range promptTokens {
		if pos >= config.BlockSize {
			break
		}
		_ = gpt(tokenID, pos, keys, values)
		tokenID = nextID
		pos++
	}

	// Generate response
	completionTokens := 0
	outTokens := make([]int, 0, req.MaxTokens)
	recent := make([]int, 0, 64)
	stopSeqs := []string{"\nUser:", "\nAssistant:"}

	for pos < config.BlockSize && completionTokens < req.MaxTokens {
		logits := gpt(tokenID, pos, keys, values)
		recentSet := map[int]bool{}
		for _, id := range recent {
			recentSet[id] = true
		}
		weights := model.NextTokenWeights(logits, req.Temperature, 40, req.TopP, recentSet, 1.1)
		tokenID = model.SampleWeighted(weights)

		if tokenID == tokenizer.BosID {
			break
		}

		outTokens = append(outTokens, tokenID)
		recent = append(recent, tokenID)
		if len(recent) > 64 {
			recent = recent[len(recent)-64:]
		}
		completionTokens++
		pos++

		// Check for stop sequences in decoded text
		fullText := tokenizer.DecodeTokens(outTokens)
		stopFound := false
		for _, stop := range stopSeqs {
			if strings.Contains(fullText, stop) {
				stopFound = true
				break
			}
		}
		if stopFound {
			break
		}
	}

	responseText := strings.TrimSpace(tokenizer.DecodeTokens(outTokens))
	// Clean up any trailing stop sequence markers
	for _, stop := range stopSeqs {
		if idx := strings.Index(responseText, strings.TrimSpace(stop)); idx >= 0 {
			responseText = responseText[:idx]
		}
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "microgpt",
		Choices: []struct {
			Message      ChatMessage `json:"message"`
			Index        int         `json:"index"`
			FinishReason string      `json:"finish_reason"`
		}{
			{
				Message: ChatMessage{
					Role:    "assistant",
					Content: strings.TrimSpace(responseText),
				},
				Index:        0,
				FinishReason: "stop",
			},
		},
	}
	resp.Usage.PromptTokens = len(promptTokens)
	resp.Usage.CompletionTokens = completionTokens
	resp.Usage.TotalTokens = resp.Usage.PromptTokens + resp.Usage.CompletionTokens

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	resp := struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}{
		Object: "list",
		Data: []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		}{
			{
				ID:      "microgpt",
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "microgpt",
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "MicroGPT API is running.\n\nEndpoints:\n- POST /v1/chat/completions\n- GET /v1/models\n")
}

func main() {
	initModel()

	http.HandleFunc("/", handleRoot)
	http.HandleFunc("/v1/chat/completions", handleChat)
	http.HandleFunc("/v1/models", handleModels)

	port := os.Getenv("PORT")
	if port == "" {
		port = "7860" // Standard port for HF Spaces
	}

	log.Printf("Starting OpenAI-compatible server on port %s...", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
