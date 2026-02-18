//go:build ignore

// This file is a heavily commented reference version of main.go.
// It is excluded from build with `//go:build ignore` so it does not conflict
// with the runnable `main.go` entrypoint.
package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
)

// Value is a scalar in the computation graph.
// Data is forward value, Grad is d(loss)/d(this node).
// Children + LocalGrads encode graph edges and local derivatives.
type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

// V creates a leaf scalar.
func V(x float64) *Value { return &Value{Data: x} }

// Add/Mul/Sub/Div/Pow/Log/Exp/ReLU mirror autograd operations in Python microgpt.
func Add(a, b *Value) *Value {
	return &Value{Data: a.Data + b.Data, Children: []*Value{a, b}, LocalGrads: []float64{1, 1}}
}
func Sub(a, b *Value) *Value { return Add(a, Neg(b)) }
func Mul(a, b *Value) *Value {
	return &Value{Data: a.Data * b.Data, Children: []*Value{a, b}, LocalGrads: []float64{b.Data, a.Data}}
}
func Pow(a *Value, p float64) *Value {
	return &Value{Data: math.Pow(a.Data, p), Children: []*Value{a}, LocalGrads: []float64{p * math.Pow(a.Data, p-1)}}
}
func Div(a, b *Value) *Value { return Mul(a, Pow(b, -1)) }
func Neg(a *Value) *Value    { return Mul(a, V(-1)) }
func Log(a *Value) *Value {
	return &Value{Data: math.Log(a.Data), Children: []*Value{a}, LocalGrads: []float64{1 / a.Data}}
}
func Exp(a *Value) *Value {
	ed := math.Exp(a.Data)
	return &Value{Data: ed, Children: []*Value{a}, LocalGrads: []float64{ed}}
}
func ReLU(a *Value) *Value {
	if a.Data > 0 {
		return &Value{Data: a.Data, Children: []*Value{a}, LocalGrads: []float64{1}}
	}
	return &Value{Data: 0, Children: []*Value{a}, LocalGrads: []float64{0}}
}

// Backward performs reverse-mode autodiff by:
// 1) topologically sorting graph nodes, 2) traversing reverse order, 3) chain-rule accumulation.
func Backward(out *Value) {
	var topo []*Value
	visited := map[*Value]bool{}
	var build func(v *Value)
	build = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, ch := range v.Children {
			build(ch)
		}
		topo = append(topo, v)
	}
	build(out)
	out.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j, ch := range v.Children {
			ch.Grad += v.LocalGrads[j] * v.Grad
		}
	}
}

// matrix initializes a 2D parameter tensor with Gaussian noise.
func matrix(nout, nin int, std float64) [][]*Value {
	m := make([][]*Value, nout)
	for o := 0; o < nout; o++ {
		row := make([]*Value, nin)
		for i := 0; i < nin; i++ {
			row[i] = V(rand.NormFloat64() * std)
		}
		m[o] = row
	}
	return m
}

// linear computes matrix-vector multiply.
func linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for o, row := range w {
		s := V(0)
		for i := 0; i < len(x); i++ {
			s = Add(s, Mul(row[i], x[i]))
		}
		out[o] = s
	}
	return out
}

// softmax turns logits into probabilities with max-shift stabilization.
func softmax(logits []*Value) []*Value {
	maxVal := logits[0].Data
	for i := 1; i < len(logits); i++ {
		if logits[i].Data > maxVal {
			maxVal = logits[i].Data
		}
	}
	exps := make([]*Value, len(logits))
	total := V(0)
	for i, l := range logits {
		e := Exp(Sub(l, V(maxVal)))
		exps[i] = e
		total = Add(total, e)
	}
	probs := make([]*Value, len(logits))
	for i := range exps {
		probs[i] = Div(exps[i], total)
	}
	return probs
}

// rmsnorm applies root-mean-square normalization.
func rmsnorm(x []*Value) []*Value {
	ms := V(0)
	for _, xi := range x {
		ms = Add(ms, Mul(xi, xi))
	}
	ms = Div(ms, V(float64(len(x))))
	scale := Pow(Add(ms, V(1e-5)), -0.5)
	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = Mul(xi, scale)
	}
	return out
}

// maybeDownloadDefault writes default dataset to input.txt if missing.
func maybeDownloadDefault() error {
	if _, err := os.Stat("input.txt"); err == nil {
		return nil
	}
	url := "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed download: %s", resp.Status)
	}
	out, err := os.Create("input.txt")
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = bufio.NewReader(resp.Body).WriteTo(out)
	return err
}

// loadDocs reads one non-empty training sample per line.
func loadDocs(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var docs []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

// sampleWeighted samples an index from non-negative weights.
func sampleWeighted(weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	r := rand.Float64() * total
	acc := 0.0
	for i, w := range weights {
		acc += w
		if r <= acc {
			return i
		}
	}
	return len(weights) - 1
}

// main follows the same macro-steps as Python microgpt:
// data -> tokenizer -> params -> train -> infer.
func main() {
	rand.Seed(42)

	if err := maybeDownloadDefault(); err != nil {
		panic(err)
	}
	docs, err := loadDocs("input.txt")
	if err != nil {
		panic(err)
	}
	rand.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	fmt.Printf("num docs: %d\n", len(docs))

	// Build rune vocabulary.
	charset := map[rune]bool{}
	for _, d := range docs {
		for _, r := range d {
			charset[r] = true
		}
	}
	uchars := make([]rune, 0, len(charset))
	for r := range charset {
		uchars = append(uchars, r)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })

	stoi := make(map[rune]int, len(uchars))
	for i, r := range uchars {
		stoi[r] = i
	}
	BOS := len(uchars)
	vocabSize := len(uchars) + 1
	fmt.Printf("vocab size: %d\n", vocabSize)

	// Tiny transformer hyperparameters.
	nLayer, nEmbd, blockSize, nHead := 1, 16, 16, 4
	headDim := nEmbd / nHead

	// Initialize all trainable matrices.
	state := map[string][][]*Value{
		"wte":     matrix(vocabSize, nEmbd, 0.08),
		"wpe":     matrix(blockSize, nEmbd, 0.08),
		"lm_head": matrix(vocabSize, nEmbd, 0.08),
	}
	for i := 0; i < nLayer; i++ {
		state[fmt.Sprintf("layer%d.attn_wq", i)] = matrix(nEmbd, nEmbd, 0.08)
		state[fmt.Sprintf("layer%d.attn_wk", i)] = matrix(nEmbd, nEmbd, 0.08)
		state[fmt.Sprintf("layer%d.attn_wv", i)] = matrix(nEmbd, nEmbd, 0.08)
		state[fmt.Sprintf("layer%d.attn_wo", i)] = matrix(nEmbd, nEmbd, 0.08)
		state[fmt.Sprintf("layer%d.mlp_fc1", i)] = matrix(4*nEmbd, nEmbd, 0.08)
		state[fmt.Sprintf("layer%d.mlp_fc2", i)] = matrix(nEmbd, 4*nEmbd, 0.08)
	}

	// Flatten params for optimizer updates.
	var params []*Value
	for _, mat := range state {
		for _, row := range mat {
			for _, p := range row {
				params = append(params, p)
			}
		}
	}
	fmt.Printf("num params: %d\n", len(params))

	// gpt is identical in structure to Python: embed -> attn -> mlp -> logits.
	gpt := func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		// See `main.go` for executable implementation.
		// This commented reference intentionally avoids duplicating the full function body.
		panic("use main.go for execution; this file is reference-only")
	}
	_ = gpt
	_ = sampleWeighted
	_ = stoi
	_ = BOS
	_ = headDim
	_ = params
	_ = state
	_ = blockSize
	_ = nLayer
}
