//go:build ignore

// Commented reference for main.go (not compiled).
//
// Quick explanation:
//  1. Read assistant_dataset_train.jsonl
//  2. Validate each record by record_type
//     and ensure each record has a unique id
//  3. Convert records into trainable text docs
//  4. Train tiny next-character model
//  5. Generate sample outputs
//
// Analysis:
// - Uses strict hybrid schema (knowledge, memory, qa, chat, trajectory, preference)
// - Validation happens before training text extraction
// - `go run . validate-dataset` checks schema quality without training
package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DatasetRecord is one JSONL row.
// id and record_type are required.
type DatasetRecord struct {
	RecordType string   `json:"record_type"`
	Text       string   `json:"text,omitempty"`
	Question   string   `json:"question,omitempty"`
	Answer     string   `json:"answer,omitempty"`
	Input      string   `json:"input,omitempty"`
	Output     string   `json:"output,omitempty"`
	Task       string   `json:"task,omitempty"`
	Actions    []string `json:"actions,omitempty"`
	Result     string   `json:"result,omitempty"`
	Prompt     string   `json:"prompt,omitempty"`
	Chosen     string   `json:"chosen,omitempty"`
	Rejected   string   `json:"rejected,omitempty"`
	ID         string   `json:"id,omitempty"`
}

func normalize(s string) string { return strings.TrimSpace(s) }

func joinNonEmpty(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = normalize(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, "\n")
}

// validateAndExtractDoc enforces required fields by record_type and returns
// the text representation used for training.
func validateAndExtractDoc(rec DatasetRecord) (string, error) {
	if normalize(rec.ID) == "" {
		return "", fmt.Errorf("missing required field: id")
	}
	rt := normalize(rec.RecordType)
	if rt == "" {
		return "", fmt.Errorf("missing required field: record_type")
	}
	switch rt {
	case "knowledge", "memory":
		if normalize(rec.Text) == "" {
			return "", fmt.Errorf("%s requires non-empty text", rt)
		}
		return normalize(rec.Text), nil
	case "qa":
		if normalize(rec.Question) == "" || normalize(rec.Answer) == "" {
			return "", fmt.Errorf("qa requires non-empty question and answer")
		}
		return joinNonEmpty("Question: "+normalize(rec.Question), "Answer: "+normalize(rec.Answer)), nil
	case "chat":
		if normalize(rec.Input) == "" || normalize(rec.Output) == "" {
			return "", fmt.Errorf("chat requires non-empty input and output")
		}
		return joinNonEmpty("User: "+normalize(rec.Input), "Assistant: "+normalize(rec.Output)), nil
	case "trajectory":
		if normalize(rec.Task) == "" || normalize(rec.Result) == "" {
			return "", fmt.Errorf("trajectory requires non-empty task and result")
		}
		a := ""
		if len(rec.Actions) > 0 {
			a = "Actions: " + strings.Join(rec.Actions, " | ")
		}
		return joinNonEmpty("Task: "+normalize(rec.Task), a, "Result: "+normalize(rec.Result)), nil
	case "preference":
		if normalize(rec.Prompt) == "" || normalize(rec.Chosen) == "" || normalize(rec.Rejected) == "" {
			return "", fmt.Errorf("preference requires non-empty prompt, chosen, and rejected")
		}
		return joinNonEmpty("Prompt: "+normalize(rec.Prompt), "Preferred: "+normalize(rec.Chosen)), nil
	default:
		return "", fmt.Errorf("unsupported record_type: %s", rt)
	}
}

// parseDatasetLine parses and validates one JSONL line.
func parseDatasetLine(line string, lineNo int) (DatasetRecord, string, error) {
	var rec DatasetRecord
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		return DatasetRecord{}, "", fmt.Errorf("invalid JSON at line %d: %w", lineNo, err)
	}
	doc, err := validateAndExtractDoc(rec)
	if err != nil {
		if rec.ID != "" {
			return DatasetRecord{}, "", fmt.Errorf("line %d (id=%s): %w", lineNo, rec.ID, err)
		}
		return DatasetRecord{}, "", fmt.Errorf("line %d: %w", lineNo, err)
	}
	return rec, doc, nil
}
