package model

import (
	"fmt"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

type TokenizerRuntime struct {
	Mode        string
	CharToLocal map[rune]int
	LocalToChar []rune
	BpeEncoding string
	Bpe         *tiktoken.Tiktoken
	BpeToLocal  map[int]int
	LocalToBPE  []int
	UnkID       int
	BosID       int
}

func (t TokenizerRuntime) VocabSize() int {
	if t.Mode == "bpe_cl100k" {
		return len(t.LocalToBPE) + 2
	}
	return len(t.LocalToChar) + 1
}

func (t TokenizerRuntime) EncodeDoc(doc string) []int {
	if t.Mode == "bpe_cl100k" {
		raw := t.Bpe.EncodeOrdinary(doc)
		out := make([]int, 0, len(raw))
		for _, id := range raw {
			if local, ok := t.BpeToLocal[id]; ok {
				out = append(out, local)
			} else {
				out = append(out, t.UnkID)
			}
		}
		return out
	}
	out := make([]int, 0, len(doc))
	for _, r := range doc {
		if id, ok := t.CharToLocal[r]; ok {
			out = append(out, id)
		}
	}
	return out
}

func (t TokenizerRuntime) DecodeTokens(tokens []int) string {
	if t.Mode == "bpe_cl100k" {
		raw := make([]int, 0, len(tokens))
		for _, local := range tokens {
			if local >= 0 && local < len(t.LocalToBPE) {
				raw = append(raw, t.LocalToBPE[local])
			}
		}
		return t.Bpe.Decode(raw)
	}
	out := make([]rune, 0, len(tokens))
	for _, id := range tokens {
		if id >= 0 && id < len(t.LocalToChar) {
			out = append(out, t.LocalToChar[id])
		}
	}
	return string(out)
}

func TokenizerFromCheckpoint(ckpt TrainingCheckpoint) (TokenizerRuntime, error) {
	if ckpt.Tokenization == "bpe_cl100k" || len(ckpt.BPETokenIDs) > 0 {
		encName := strings.TrimSpace(ckpt.BPEEncoding)
		if encName == "" {
			encName = "cl100k_base"
		}
		enc, err := tiktoken.GetEncoding(encName)
		if err != nil {
			return TokenizerRuntime{}, err
		}
		localToBPE := append([]int(nil), ckpt.BPETokenIDs...)
		bpeToLocal := make(map[int]int, len(localToBPE))
		for i, id := range localToBPE {
			bpeToLocal[id] = i
		}
		return TokenizerRuntime{
			Mode:        "bpe_cl100k",
			BpeEncoding: encName,
			Bpe:         enc,
			BpeToLocal:  bpeToLocal,
			LocalToBPE:  localToBPE,
			UnkID:       len(localToBPE),
			BosID:       len(localToBPE) + 1,
		}, nil
	}
	uchars, err := stringsToRunes(ckpt.Vocab)
	if err != nil {
		return TokenizerRuntime{}, err
	}
	if len(uchars) == 0 {
		return TokenizerRuntime{}, fmt.Errorf("checkpoint has empty character vocab")
	}
	charToLocal := make(map[rune]int, len(uchars))
	for i, r := range uchars {
		charToLocal[r] = i
	}
	return TokenizerRuntime{
		Mode:        "char",
		CharToLocal: charToLocal,
		LocalToChar: uchars,
		BosID:       len(uchars),
	}, nil
}

func stringsToRunes(ss []string) ([]rune, error) {
	out := make([]rune, 0, len(ss))
	for _, s := range ss {
		r := []rune(s)
		if len(r) != 1 {
			return nil, fmt.Errorf("invalid vocab token %q: expected one rune", s)
		}
		out = append(out, r[0])
	}
	return out, nil
}
