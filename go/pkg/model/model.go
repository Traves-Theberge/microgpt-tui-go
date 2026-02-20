package model

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
)

// Value represents a scalar for autograd
type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

func V(x float64) *Value {
	return &Value{Data: x}
}

func Add(a, b *Value) *Value {
	out := &Value{Data: a.Data + b.Data, Children: []*Value{a, b}, LocalGrads: []float64{1, 1}}
	return out
}

func Sub(a, b *Value) *Value {
	out := &Value{Data: a.Data - b.Data, Children: []*Value{a, b}, LocalGrads: []float64{1, -1}}
	return out
}

func Mul(a, b *Value) *Value {
	out := &Value{Data: a.Data * b.Data, Children: []*Value{a, b}, LocalGrads: []float64{b.Data, a.Data}}
	return out
}

func Pow(a *Value, p float64) *Value {
	out := &Value{Data: math.Pow(a.Data, p), Children: []*Value{a}, LocalGrads: []float64{p * math.Pow(a.Data, p-1)}}
	return out
}

func Div(a, b *Value) *Value {
	return Mul(a, Pow(b, -1))
}

func Neg(a *Value) *Value {
	return Mul(a, V(-1))
}

func Log(a *Value) *Value {
	out := &Value{Data: math.Log(a.Data), Children: []*Value{a}, LocalGrads: []float64{1 / a.Data}}
	return out
}

func Exp(a *Value) *Value {
	out := &Value{Data: math.Exp(a.Data), Children: []*Value{a}, LocalGrads: []float64{math.Exp(a.Data)}}
	return out
}

func ReLU(a *Value) *Value {
	val := 0.0
	grad := 0.0
	if a.Data > 0 {
		val = a.Data
		grad = 1
	}
	out := &Value{Data: val, Children: []*Value{a}, LocalGrads: []float64{grad}}
	return out
}

func Backward(out *Value) {
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	var buildTopo func(*Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.Children {
				buildTopo(child)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(out)

	for _, v := range topo {
		v.Grad = 0
	}
	out.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j, child := range v.Children {
			child.Grad += v.LocalGrads[j] * v.Grad
		}
	}
}

func linear(x []*Value, w [][]*Value) []*Value {
	nout := len(w)
	nin := len(x)
	out := make([]*Value, nout)
	for i := 0; i < nout; i++ {
		s := V(0)
		for j := 0; j < nin; j++ {
			s = Add(s, Mul(x[j], w[i][j]))
		}
		out[i] = s
	}
	return out
}

func softmax(logits []*Value) []*Value {
	maxVal := -math.MaxFloat64
	for _, l := range logits {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}
	exps := make([]*Value, len(logits))
	sumExp := V(0)
	for i, l := range logits {
		exps[i] = Exp(Sub(l, V(maxVal)))
		sumExp = Add(sumExp, exps[i])
	}
	out := make([]*Value, len(logits))
	invSum := Div(V(1), sumExp)
	for i := range exps {
		out[i] = Mul(exps[i], invSum)
	}
	return out
}

func rmsnorm(x []*Value) []*Value {
	meanSq := V(0)
	for _, v := range x {
		meanSq = Add(meanSq, Pow(v, 2))
	}
	meanSq = Mul(V(1/float64(len(x))), meanSq)
	invStd := Div(V(1), Pow(Add(meanSq, V(1e-6)), 0.5))
	out := make([]*Value, len(x))
	for i, v := range x {
		out[i] = Mul(v, invStd)
	}
	return out
}

// TrainingCheckpoint structs
type TrainingCheckpoint struct {
	Version      int                      `json:"version"`
	CreatedAt    string                   `json:"created_at"`
	Config       TrainingCheckpointConfig `json:"config"`
	Tokenization string                   `json:"tokenization,omitempty"`
	BPEEncoding  string                   `json:"bpe_encoding,omitempty"`
	BPETokenIDs  []int                    `json:"bpe_token_ids,omitempty"`
	Vocab        []string                 `json:"vocab,omitempty"`
	State        map[string][][]float64   `json:"state"`
}

type TrainingCheckpointConfig struct {
	NLayer    int `json:"n_layer"`
	NEmbd     int `json:"n_embd"`
	NHead     int `json:"n_head"`
	BlockSize int `json:"block_size"`
}

func ImportState(src map[string][][]float64) map[string][][]*Value {
	out := make(map[string][][]*Value, len(src))
	for name, mat := range src {
		rows := make([][]*Value, len(mat))
		for i, row := range mat {
			r := make([]*Value, len(row))
			for j, v := range row {
				r[j] = V(v)
			}
			rows[i] = r
		}
		out[name] = rows
	}
	return out
}

func LoadCheckpoint(path string) (TrainingCheckpoint, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return TrainingCheckpoint{}, err
	}
	var ckpt TrainingCheckpoint
	if err := json.Unmarshal(b, &ckpt); err != nil {
		return TrainingCheckpoint{}, err
	}
	if ckpt.Config.NLayer < 1 || ckpt.Config.NEmbd < 1 || ckpt.Config.NHead < 1 || ckpt.Config.BlockSize < 2 {
		return TrainingCheckpoint{}, fmt.Errorf("invalid checkpoint config")
	}
	if ckpt.Config.NEmbd%ckpt.Config.NHead != 0 {
		return TrainingCheckpoint{}, fmt.Errorf("invalid checkpoint: n_embd must be divisible by n_head")
	}
	return ckpt, nil
}

func BuildGPT(state map[string][][]*Value, nLayer, nEmbd, nHead int) func(tokenID, posID int, keys, values [][][]*Value) []*Value {
	headDim := nEmbd / nHead
	return func(tokenID, posID int, keys, values [][][]*Value) []*Value {
		tokEmb := state["wte"][tokenID]
		posEmb := state["wpe"][posID]
		x := make([]*Value, len(tokEmb))
		for i := range tokEmb {
			x[i] = Add(tokEmb[i], posEmb[i])
		}
		x = rmsnorm(x)

		for li := 0; li < nLayer; li++ {
			xResidual := x
			x = rmsnorm(x)
			q := linear(x, state[fmt.Sprintf("layer%d.attn_wq", li)])
			k := linear(x, state[fmt.Sprintf("layer%d.attn_wk", li)])
			v := linear(x, state[fmt.Sprintf("layer%d.attn_wv", li)])
			keys[li] = append(keys[li], k)
			values[li] = append(values[li], v)

			xAttn := make([]*Value, 0, nEmbd)
			for h := 0; h < nHead; h++ {
				hs := h * headDim
				qH := q[hs : hs+headDim]

				kH := make([][]*Value, len(keys[li]))
				vH := make([][]*Value, len(values[li]))
				for t := 0; t < len(keys[li]); t++ {
					kH[t] = keys[li][t][hs : hs+headDim]
					vH[t] = values[li][t][hs : hs+headDim]
				}

				attnLogits := make([]*Value, len(kH))
				for t := 0; t < len(kH); t++ {
					score := V(0)
					for j := 0; j < headDim; j++ {
						score = Add(score, Mul(qH[j], kH[t][j]))
					}
					attnLogits[t] = Div(score, V(math.Sqrt(float64(headDim))))
				}
				attnWeights := softmax(attnLogits)

				headOut := make([]*Value, headDim)
				for j := 0; j < headDim; j++ {
					s := V(0)
					for t := 0; t < len(vH); t++ {
						s = Add(s, Mul(attnWeights[t], vH[t][j]))
					}
					headOut[j] = s
				}
				xAttn = append(xAttn, headOut...)
			}

			x = linear(xAttn, state[fmt.Sprintf("layer%d.attn_wo", li)])
			for i := range x {
				x[i] = Add(x[i], xResidual[i])
			}

			xResidual = x
			x = rmsnorm(x)
			x = linear(x, state[fmt.Sprintf("layer%d.mlp_fc1", li)])
			for i := range x {
				x[i] = ReLU(x[i])
			}
			x = linear(x, state[fmt.Sprintf("layer%d.mlp_fc2", li)])
			for i := range x {
				x[i] = Add(x[i], xResidual[i])
			}
		}

		return linear(x, state["lm_head"])
	}
}

// Sampling functions
func SampleWeighted(weights []float64) int {
	sum := 0.0
	for _, w := range weights {
		sum += w
	}
	r := rand.Float64() * sum
	running := 0.0
	for i, w := range weights {
		running += w
		if r <= running {
			return i
		}
	}
	return len(weights) - 1
}

func SoftmaxFloat(logits []float64) []float64 {
	maxLogit := -math.MaxFloat64
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}
	sum := 0.0
	out := make([]float64, len(logits))
	for i, l := range logits {
		out[i] = math.Exp(l - maxLogit)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func NextTokenWeights(logits []*Value, temperature float64, topK int, topP float64, recent map[int]bool, repetitionPenalty float64) []float64 {
	l := make([]float64, len(logits))
	for i, v := range logits {
		l[i] = v.Data
		if recent[i] {
			if l[i] >= 0 {
				l[i] /= repetitionPenalty
			} else {
				l[i] *= repetitionPenalty
			}
		}
		l[i] /= temperature
	}
	w := SoftmaxFloat(l)
	if topK > 0 {
		w = ApplyTopK(w, topK)
	}
	if topP > 0 && topP < 1.0 {
		w = ApplyTopP(w, topP)
	}
	return w
}

func ApplyTopK(weights []float64, k int) []float64 {
	if k >= len(weights) {
		return weights
	}
	type kv struct {
		i int
		w float64
	}
	arr := make([]kv, len(weights))
	for i, w := range weights {
		arr[i] = kv{i, w}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].w > arr[j].w })
	out := make([]float64, len(weights))
	for i := 0; i < k; i++ {
		out[arr[i].i] = arr[i].w
	}
	return out
}

func ApplyTopP(weights []float64, p float64) []float64 {
	type kv struct {
		i int
		w float64
	}
	arr := make([]kv, len(weights))
	for i, w := range weights {
		arr[i] = kv{i, w}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].w > arr[j].w })
	out := make([]float64, len(weights))
	sum := 0.0
	for i := 0; i < len(arr); i++ {
		sum += arr[i].w
		out[arr[i].i] = arr[i].w
		if sum >= p {
			break
		}
	}
	return out
}
