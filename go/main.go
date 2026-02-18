package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	tiktoken "github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

type DatasetRecord struct {
	RecordType  string   `json:"record_type"`
	Text        string   `json:"text,omitempty"`
	Question    string   `json:"question,omitempty"`
	Answer      string   `json:"answer,omitempty"`
	Input       string   `json:"input,omitempty"`
	Output      string   `json:"output,omitempty"`
	Task        string   `json:"task,omitempty"`
	Actions     []string `json:"actions,omitempty"`
	Result      string   `json:"result,omitempty"`
	Prompt      string   `json:"prompt,omitempty"`
	Chosen      string   `json:"chosen,omitempty"`
	Rejected    string   `json:"rejected,omitempty"`
	ID          string   `json:"id,omitempty"`
	Instruction string   `json:"instruction,omitempty"`
	Response    string   `json:"response,omitempty"`
	Context     string   `json:"context,omitempty"`
}

func copyExampleDataset(path string, cause error) error {
	examplePath := "assistant_dataset_seed.jsonl"
	b, err := os.ReadFile(examplePath)
	if err != nil {
		return cause
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return cause
	}
	return nil
}

func normalize(s string) string {
	return strings.TrimSpace(s)
}

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

func validateAndExtractDoc(rec DatasetRecord) (string, error) {
	if normalize(rec.ID) == "" {
		return "", fmt.Errorf("missing required field: id")
	}
	rt := normalize(rec.RecordType)
	if rt == "" {
		return "", fmt.Errorf("missing required field: record_type")
	}
	switch rt {
	case "knowledge":
		text := normalize(rec.Text)
		if text == "" {
			return "", fmt.Errorf("knowledge requires non-empty text")
		}
		return text, nil
	case "memory":
		text := normalize(rec.Text)
		if text == "" {
			return "", fmt.Errorf("memory requires non-empty text")
		}
		return text, nil
	case "qa":
		q := normalize(rec.Question)
		a := normalize(rec.Answer)
		if q == "" || a == "" {
			return "", fmt.Errorf("qa requires non-empty question and answer")
		}
		return joinNonEmpty("Question: "+q, "Answer: "+a), nil
	case "chat":
		in := normalize(rec.Input)
		out := normalize(rec.Output)
		if in == "" || out == "" {
			return "", fmt.Errorf("chat requires non-empty input and output")
		}
		return joinNonEmpty("User: "+in, "Assistant: "+out), nil
	case "trajectory":
		task := normalize(rec.Task)
		result := normalize(rec.Result)
		if task == "" || result == "" {
			return "", fmt.Errorf("trajectory requires non-empty task and result")
		}
		actions := ""
		if len(rec.Actions) > 0 {
			clean := make([]string, 0, len(rec.Actions))
			for _, a := range rec.Actions {
				a = normalize(a)
				if a != "" {
					clean = append(clean, a)
				}
			}
			if len(clean) > 0 {
				actions = "Actions: " + strings.Join(clean, " | ")
			}
		}
		return joinNonEmpty("Task: "+task, actions, "Result: "+result), nil
	case "preference":
		prompt := normalize(rec.Prompt)
		chosen := normalize(rec.Chosen)
		rejected := normalize(rec.Rejected)
		if prompt == "" || chosen == "" || rejected == "" {
			return "", fmt.Errorf("preference requires non-empty prompt, chosen, and rejected")
		}
		// Use prompt+chosen for training text; rejected is still required for dataset quality checks.
		return joinNonEmpty("Prompt: "+prompt, "Preferred: "+chosen), nil
	default:
		return "", fmt.Errorf("unsupported record_type: %s", rt)
	}
}

func parseDatasetLine(line string, lineNo int) (DatasetRecord, string, error) {
	var rec DatasetRecord
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		return DatasetRecord{}, "", fmt.Errorf("invalid JSON at line %d: %w", lineNo, err)
	}
	// Compatibility path for raw Dolly-style rows (instruction/response/context).
	// We synthesize required schema fields so raw public datasets can be used directly.
	if normalize(rec.ID) == "" && normalize(rec.RecordType) == "" {
		inst := normalize(rec.Instruction)
		resp := normalize(rec.Response)
		if inst != "" && resp != "" {
			rec.ID = fmt.Sprintf("dolly-auto-%d", lineNo)
			rec.RecordType = "chat"
			if ctx := normalize(rec.Context); ctx != "" {
				rec.Input = inst + "\nContext: " + ctx
			} else {
				rec.Input = inst
			}
			rec.Output = resp
		}
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

func validateDatasetJSONL(path string) (map[string]int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	counts := map[string]int{}
	seenIDs := map[string]int{}
	lineNo := 0
	valid := 0
	s := bufio.NewScanner(f)
	for s.Scan() {
		lineNo++
		line := normalize(s.Text())
		if line == "" {
			continue
		}
		rec, _, err := parseDatasetLine(line, lineNo)
		if err != nil {
			return nil, valid, err
		}
		id := normalize(rec.ID)
		if prev, ok := seenIDs[id]; ok {
			return nil, valid, fmt.Errorf("line %d (id=%s): duplicate id also used at line %d", lineNo, id, prev)
		}
		seenIDs[id] = lineNo
		counts[normalize(rec.RecordType)]++
		valid++
	}
	if err := s.Err(); err != nil {
		return nil, valid, err
	}
	return counts, valid, nil
}

func V(x float64) *Value {
	return &Value{Data: x}
}

func Add(a, b *Value) *Value {
	return &Value{Data: a.Data + b.Data, Children: []*Value{a, b}, LocalGrads: []float64{1, 1}}
}

func Sub(a, b *Value) *Value {
	return Add(a, Neg(b))
}

func Mul(a, b *Value) *Value {
	return &Value{Data: a.Data * b.Data, Children: []*Value{a, b}, LocalGrads: []float64{b.Data, a.Data}}
}

func Pow(a *Value, p float64) *Value {
	return &Value{Data: math.Pow(a.Data, p), Children: []*Value{a}, LocalGrads: []float64{p * math.Pow(a.Data, p-1)}}
}

func Div(a, b *Value) *Value {
	return Mul(a, Pow(b, -1))
}

func Neg(a *Value) *Value {
	return Mul(a, V(-1))
}

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

func maybeDownloadDefaultJSONL(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	return copyExampleDataset(path, fmt.Errorf("dataset not found at %s", path))
}

func loadDocsJSONL(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var docs []string
	s := bufio.NewScanner(f)
	lineNo := 0
	for s.Scan() {
		lineNo++
		line := normalize(s.Text())
		if line == "" {
			continue
		}
		_, doc, err := parseDatasetLine(line, lineNo)
		if err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

func sampleWeighted(weights []float64) int {
	total := 0.0
	for _, w := range weights {
		if w > 0 && !math.IsNaN(w) && !math.IsInf(w, 0) {
			total += w
		}
	}
	if total <= 0 {
		return rand.Intn(len(weights))
	}
	r := rand.Float64() * total
	acc := 0.0
	for i, w := range weights {
		if w > 0 && !math.IsNaN(w) && !math.IsInf(w, 0) {
			acc += w
		}
		if r <= acc {
			return i
		}
	}
	return len(weights) - 1
}

func softmaxFloat(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	expVals := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		ev := math.Exp(v - maxV)
		expVals[i] = ev
		sum += ev
	}
	if sum <= 0 {
		return expVals
	}
	for i := range expVals {
		expVals[i] /= sum
	}
	return expVals
}

func applyTopK(weights []float64, k int) []float64 {
	if k <= 0 || k >= len(weights) {
		return weights
	}
	type kv struct {
		i int
		w float64
	}
	arr := make([]kv, len(weights))
	for i, w := range weights {
		arr[i] = kv{i: i, w: w}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].w > arr[j].w })
	keep := map[int]bool{}
	for i := 0; i < k && i < len(arr); i++ {
		keep[arr[i].i] = true
	}
	out := make([]float64, len(weights))
	for i, w := range weights {
		if keep[i] {
			out[i] = w
		}
	}
	return out
}

func applyTopP(weights []float64, p float64) []float64 {
	if p <= 0 || p >= 1 {
		return weights
	}
	type kv struct {
		i int
		w float64
	}
	arr := make([]kv, len(weights))
	for i, w := range weights {
		arr[i] = kv{i: i, w: w}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].w > arr[j].w })
	cum := 0.0
	keep := map[int]bool{}
	for _, it := range arr {
		if it.w <= 0 {
			continue
		}
		keep[it.i] = true
		cum += it.w
		if cum >= p {
			break
		}
	}
	out := make([]float64, len(weights))
	for i, w := range weights {
		if keep[i] {
			out[i] = w
		}
	}
	return out
}

func nextTokenWeights(logits []*Value, temperature float64, topK int, topP float64, recent map[int]bool, repetitionPenalty float64) []float64 {
	if temperature <= 0 {
		temperature = 1.0
	}
	raw := make([]float64, len(logits))
	for i, l := range logits {
		v := l.Data
		if repetitionPenalty > 1 && recent[i] {
			v /= repetitionPenalty
		}
		raw[i] = v / temperature
	}
	w := softmaxFloat(raw)
	w = applyTopK(w, topK)
	w = applyTopP(w, topP)
	return w
}

func evalLoss(gpt func(tokenID, posID int, keys, values [][][]*Value) []*Value, docs [][]int, nLayer, blockSize, bosID, maxDocs int) float64 {
	if len(docs) == 0 {
		return math.Inf(1)
	}
	steps := len(docs)
	if maxDocs > 0 && maxDocs < steps {
		steps = maxDocs
	}
	indices := rand.Perm(len(docs))
	if steps < len(indices) {
		indices = indices[:steps]
	}
	total := 0.0
	count := 0
	for _, idx := range indices {
		doc := docs[idx]
		tokens := make([]int, 0, len(doc)+2)
		tokens = append(tokens, bosID)
		tokens = append(tokens, doc...)
		tokens = append(tokens, bosID)
		n := len(tokens) - 1
		if n > blockSize {
			start := rand.Intn(n - blockSize + 1)
			tokens = tokens[start : start+blockSize+1]
			n = blockSize
		}
		if n <= 0 {
			continue
		}
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		for posID := 0; posID < n; posID++ {
			tokenID, targetID := tokens[posID], tokens[posID+1]
			logits := gpt(tokenID, posID, keys, values)
			weights := softmaxFloat(func() []float64 {
				tmp := make([]float64, len(logits))
				for i, l := range logits {
					tmp[i] = l.Data
				}
				return tmp
			}())
			p := weights[targetID]
			if p < 1e-12 {
				p = 1e-12
			}
			total += -math.Log(p)
			count++
		}
	}
	if count == 0 {
		return math.Inf(1)
	}
	return total / float64(count)
}

func envInt(name string, def int) int {
	v := normalize(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func envFloat(name string, def float64) float64 {
	v := normalize(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return n
}

func envBool(name string, def bool) bool {
	v := strings.ToLower(normalize(os.Getenv(name)))
	if v == "" {
		return def
	}
	switch v {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	default:
		return def
	}
}

func readLinuxMemInfo() (totalKB uint64, availableKB uint64, ok bool) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0, false
	}
	lines := strings.Split(string(b), "\n")
	var t, a uint64
	for _, ln := range lines {
		f := strings.Fields(ln)
		if len(f) < 2 {
			continue
		}
		switch f[0] {
		case "MemTotal:":
			if v, err := strconv.ParseUint(f[1], 10, 64); err == nil {
				t = v
			}
		case "MemAvailable:":
			if v, err := strconv.ParseUint(f[1], 10, 64); err == nil {
				a = v
			}
		}
	}
	if t == 0 {
		return 0, 0, false
	}
	return t, a, true
}

func previewRunes(s string, max int) string {
	rs := []rune(s)
	if len(rs) <= max {
		return s
	}
	return string(rs[:max]) + "..."
}

func slugifyName(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return "run"
	}
	var b strings.Builder
	prevDash := false
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if ok {
			b.WriteRune(r)
			prevDash = false
			continue
		}
		if !prevDash {
			b.WriteRune('-')
			prevDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "run"
	}
	return out
}

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

type tokenizerRuntime struct {
	mode        string
	bpeEncoding string
	bpe         *tiktoken.Tiktoken
	bpeToLocal  map[int]int
	localToBPE  []int
	charToLocal map[rune]int
	localToChar []rune
	unkID       int
	bosID       int
}

func runesToStrings(rs []rune) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = string(r)
	}
	return out
}

func runOutput(name string, args ...string) string {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func gpuCheck() {
	fmt.Println("GPU readiness check")
	fmt.Println("trainer backend: CPU kernels (GPU math kernels not wired yet)")
	smi := runOutput("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
	if smi == "" {
		fmt.Println("nvidia-smi: not available (NVIDIA runtime not detected)")
		fmt.Println("recommended for this project now: TRAIN_DEVICE=cpu")
		return
	}
	fmt.Println("nvidia-smi:", strings.Split(smi, "\n")[0])
	nvcc := runOutput("nvcc", "--version")
	if nvcc == "" {
		fmt.Println("nvcc: not found (CUDA toolkit missing or not in PATH)")
	} else {
		lines := strings.Split(nvcc, "\n")
		fmt.Println("nvcc:", lines[len(lines)-1])
	}
	fmt.Println("status: GPU present, but training still executes on CPU in current codebase")
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

func (t tokenizerRuntime) vocabSize() int {
	if t.mode == "bpe_cl100k" {
		return len(t.localToBPE) + 2
	}
	return len(t.localToChar) + 1
}

func (t tokenizerRuntime) encodeDoc(doc string) []int {
	if t.mode == "bpe_cl100k" {
		raw := t.bpe.EncodeOrdinary(doc)
		out := make([]int, 0, len(raw))
		for _, id := range raw {
			if local, ok := t.bpeToLocal[id]; ok {
				out = append(out, local)
			} else {
				out = append(out, t.unkID)
			}
		}
		return out
	}
	out := make([]int, 0, len(doc))
	for _, r := range doc {
		if id, ok := t.charToLocal[r]; ok {
			out = append(out, id)
		}
	}
	return out
}

func (t tokenizerRuntime) decodeTokens(tokens []int) string {
	if t.mode == "bpe_cl100k" {
		raw := make([]int, 0, len(tokens))
		for _, local := range tokens {
			if local >= 0 && local < len(t.localToBPE) {
				raw = append(raw, t.localToBPE[local])
			}
		}
		return t.bpe.Decode(raw)
	}
	out := make([]rune, 0, len(tokens))
	for _, id := range tokens {
		if id >= 0 && id < len(t.localToChar) {
			out = append(out, t.localToChar[id])
		}
	}
	return string(out)
}

func buildTokenizer(mode, bpeEncoding string, targetVocab int, trainDocs []string) (tokenizerRuntime, [][]int, error) {
	if strings.EqualFold(mode, "bpe") {
		encName := strings.TrimSpace(bpeEncoding)
		if encName == "" {
			encName = "cl100k_base"
		}
		enc, err := tiktoken.GetEncoding(encName)
		if err != nil {
			return tokenizerRuntime{}, nil, err
		}
		if targetVocab < 512 {
			targetVocab = 512
		}
		rawDocs := make([][]int, 0, len(trainDocs))
		freq := map[int]int{}
		for _, doc := range trainDocs {
			ids := enc.EncodeOrdinary(doc)
			rawDocs = append(rawDocs, ids)
			for _, id := range ids {
				freq[id]++
			}
		}
		type kv struct {
			id  int
			cnt int
		}
		arr := make([]kv, 0, len(freq))
		for id, cnt := range freq {
			arr = append(arr, kv{id: id, cnt: cnt})
		}
		sort.Slice(arr, func(i, j int) bool {
			if arr[i].cnt == arr[j].cnt {
				return arr[i].id < arr[j].id
			}
			return arr[i].cnt > arr[j].cnt
		})
		keep := targetVocab - 2 // reserve UNK + BOS
		if keep > len(arr) {
			keep = len(arr)
		}
		localToBPE := make([]int, 0, keep)
		bpeToLocal := make(map[int]int, keep)
		for i := 0; i < keep; i++ {
			id := arr[i].id
			bpeToLocal[id] = len(localToBPE)
			localToBPE = append(localToBPE, id)
		}
		tok := tokenizerRuntime{
			mode:        "bpe_cl100k",
			bpeEncoding: encName,
			bpe:         enc,
			bpeToLocal:  bpeToLocal,
			localToBPE:  localToBPE,
			unkID:       len(localToBPE),
			bosID:       len(localToBPE) + 1,
		}
		tokenized := make([][]int, 0, len(rawDocs))
		for _, raw := range rawDocs {
			mapped := make([]int, 0, len(raw))
			for _, id := range raw {
				if local, ok := bpeToLocal[id]; ok {
					mapped = append(mapped, local)
				} else {
					mapped = append(mapped, tok.unkID)
				}
			}
			tokenized = append(tokenized, mapped)
		}
		return tok, tokenized, nil
	}

	charset := map[rune]bool{}
	for _, d := range trainDocs {
		for _, r := range d {
			charset[r] = true
		}
	}
	uchars := make([]rune, 0, len(charset))
	for r := range charset {
		uchars = append(uchars, r)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	charToLocal := make(map[rune]int, len(uchars))
	for i, r := range uchars {
		charToLocal[r] = i
	}
	tok := tokenizerRuntime{
		mode:        "char",
		charToLocal: charToLocal,
		localToChar: uchars,
		bosID:       len(uchars),
	}
	tokenized := make([][]int, 0, len(trainDocs))
	for _, d := range trainDocs {
		tokenized = append(tokenized, tok.encodeDoc(d))
	}
	return tok, tokenized, nil
}

func tokenizerFromCheckpoint(ckpt TrainingCheckpoint) (tokenizerRuntime, error) {
	if ckpt.Tokenization == "bpe_cl100k" || len(ckpt.BPETokenIDs) > 0 {
		encName := strings.TrimSpace(ckpt.BPEEncoding)
		if encName == "" {
			encName = "cl100k_base"
		}
		enc, err := tiktoken.GetEncoding(encName)
		if err != nil {
			return tokenizerRuntime{}, err
		}
		localToBPE := append([]int(nil), ckpt.BPETokenIDs...)
		bpeToLocal := make(map[int]int, len(localToBPE))
		for i, id := range localToBPE {
			bpeToLocal[id] = i
		}
		return tokenizerRuntime{
			mode:        "bpe_cl100k",
			bpeEncoding: encName,
			bpe:         enc,
			bpeToLocal:  bpeToLocal,
			localToBPE:  localToBPE,
			unkID:       len(localToBPE),
			bosID:       len(localToBPE) + 1,
		}, nil
	}
	uchars, err := stringsToRunes(ckpt.Vocab)
	if err != nil {
		return tokenizerRuntime{}, err
	}
	if len(uchars) == 0 {
		return tokenizerRuntime{}, fmt.Errorf("checkpoint has empty character vocab")
	}
	charToLocal := make(map[rune]int, len(uchars))
	for i, r := range uchars {
		charToLocal[r] = i
	}
	return tokenizerRuntime{
		mode:        "char",
		charToLocal: charToLocal,
		localToChar: uchars,
		bosID:       len(uchars),
	}, nil
}

func exportState(state map[string][][]*Value) map[string][][]float64 {
	out := make(map[string][][]float64, len(state))
	for name, mat := range state {
		rows := make([][]float64, len(mat))
		for i, row := range mat {
			r := make([]float64, len(row))
			for j, v := range row {
				r[j] = v.Data
			}
			rows[i] = r
		}
		out[name] = rows
	}
	return out
}

func importState(src map[string][][]float64) map[string][][]*Value {
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

func saveCheckpoint(path string, ckpt TrainingCheckpoint) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(ckpt, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func loadCheckpoint(path string) (TrainingCheckpoint, error) {
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
	if ckpt.Tokenization == "bpe_cl100k" || len(ckpt.BPETokenIDs) > 0 {
		if len(ckpt.BPETokenIDs) == 0 {
			return TrainingCheckpoint{}, fmt.Errorf("invalid checkpoint: bpe tokenization requires bpe_token_ids")
		}
	} else if len(ckpt.Vocab) == 0 {
		return TrainingCheckpoint{}, fmt.Errorf("invalid checkpoint: empty vocab")
	}
	return ckpt, nil
}

func buildGPT(state map[string][][]*Value, nLayer, nEmbd, nHead int) func(tokenID, posID int, keys, values [][][]*Value) []*Value {
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

func runChatOnce(checkpointPath, prompt string) error {
	ckpt, err := loadCheckpoint(checkpointPath)
	if err != nil {
		return err
	}
	tokenizer, err := tokenizerFromCheckpoint(ckpt)
	if err != nil {
		return err
	}
	BOS := tokenizer.bosID

	state := importState(ckpt.State)
	nLayer := ckpt.Config.NLayer
	nEmbd := ckpt.Config.NEmbd
	nHead := ckpt.Config.NHead
	blockSize := ckpt.Config.BlockSize
	gpt := buildGPT(state, nLayer, nEmbd, nHead)

	temp := envFloat("CHAT_TEMPERATURE", 0.35)
	if temp <= 0 {
		temp = 0.35
	}
	maxNew := envInt("CHAT_MAX_NEW_TOKENS", 180)
	if maxNew < 1 {
		maxNew = 180
	}
	topK := envInt("CHAT_TOP_K", 20)
	topP := envFloat("CHAT_TOP_P", 0.92)
	if topP <= 0 || topP > 1 {
		topP = 0.92
	}
	repetitionPenalty := envFloat("CHAT_REPETITION_PENALTY", 1.15)
	if repetitionPenalty < 1.0 {
		repetitionPenalty = 1.0
	}
	minNew := envInt("CHAT_MIN_NEW_TOKENS", 20)
	if minNew < 0 {
		minNew = 0
	}
	repeatLastN := envInt("CHAT_REPEAT_LAST_N", 64)
	if repeatLastN < 1 {
		repeatLastN = 64
	}

	chatPrompt := strings.TrimSpace(prompt)
	lower := strings.ToLower(chatPrompt)
	if !strings.Contains(lower, "assistant:") {
		chatPrompt = "User: " + chatPrompt + "\nAssistant:"
	}
	promptTokens := tokenizer.encodeDoc(chatPrompt)
	if len(promptTokens) > blockSize-1 {
		promptTokens = promptTokens[len(promptTokens)-(blockSize-1):]
	}
	keys := make([][][]*Value, nLayer)
	values := make([][][]*Value, nLayer)
	tokenID := BOS
	pos := 0
	for _, nextID := range promptTokens {
		if pos >= blockSize {
			break
		}
		_ = gpt(tokenID, pos, keys, values)
		tokenID = nextID
		pos++
	}

	out := make([]int, 0, maxNew)
	recent := make([]int, 0, repeatLastN)
	stopSeqs := []string{"\nUser:", "\nPrompt:", "\nTask:", "\nContext:"}
	for pos < blockSize && len(out) < maxNew {
		logits := gpt(tokenID, pos, keys, values)
		recentSet := map[int]bool{}
		for _, id := range recent {
			recentSet[id] = true
		}
		weights := nextTokenWeights(logits, temp, topK, topP, recentSet, repetitionPenalty)
		if len(out) < minNew && BOS >= 0 && BOS < len(weights) {
			weights[BOS] = 0
		}
		tokenID = sampleWeighted(weights)
		if tokenID == BOS {
			break
		}
		out = append(out, tokenID)
		recent = append(recent, tokenID)
		if len(recent) > repeatLastN {
			recent = recent[len(recent)-repeatLastN:]
		}
		if len(out) >= minNew {
			partial := tokenizer.decodeTokens(out)
			for _, stop := range stopSeqs {
				if i := strings.Index(partial, stop); i >= 0 {
					partial = partial[:i]
					fmt.Print(strings.TrimSpace(strings.TrimPrefix(partial, "Assistant:")))
					return nil
				}
			}
		}
		pos++
	}
	decoded := strings.TrimSpace(tokenizer.decodeTokens(out))
	decoded = strings.TrimSpace(strings.TrimPrefix(decoded, "Assistant:"))
	fmt.Print(decoded)
	return nil
}

func main() {
	rand.Seed(42)
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())

	datasetPath := normalize(os.Getenv("DATASET_PATH"))
	if datasetPath == "" {
		datasetPath = "datasets/raw/databricks-dolly-15k.jsonl"
	}
	if len(os.Args) > 1 && os.Args[1] == "chat-once" {
		if len(os.Args) < 4 {
			panic("usage: go run . chat-once <checkpoint_path> <prompt>")
		}
		checkpointPath := normalize(os.Args[2])
		prompt := strings.Join(os.Args[3:], " ")
		if err := runChatOnce(checkpointPath, prompt); err != nil {
			panic(err)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "gpu-check" {
		gpuCheck()
		return
	}
	if err := maybeDownloadDefaultJSONL(datasetPath); err != nil {
		panic(err)
	}
	if len(os.Args) > 1 && os.Args[1] == "validate-dataset" {
		validatePath := datasetPath
		if len(os.Args) > 2 {
			validatePath = normalize(os.Args[2])
		}
		if validatePath == "" {
			panic("validate-dataset requires a non-empty dataset path")
		}
		counts, total, err := validateDatasetJSONL(validatePath)
		if err != nil {
			panic(err)
		}
		fmt.Printf("dataset valid: %s\n", validatePath)
		fmt.Printf("records: %d\n", total)
		keys := make([]string, 0, len(counts))
		for k := range counts {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Printf("- %s: %d\n", k, counts[k])
		}
		return
	}
	docs, err := loadDocsJSONL(datasetPath)
	if err != nil {
		panic(err)
	}
	if len(docs) == 0 {
		panic("assistant_dataset_train.jsonl loaded but no documents found")
	}
	rand.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	valSplit := envFloat("VAL_SPLIT", 0.10)
	if valSplit <= 0 || valSplit >= 0.5 {
		valSplit = 0.10
	}
	valCount := int(float64(len(docs)) * valSplit)
	if valCount < 1 {
		valCount = 1
	}
	if valCount >= len(docs) {
		valCount = len(docs) - 1
	}
	trainDocsText := docs[:len(docs)-valCount]
	valDocsText := docs[len(docs)-valCount:]
	fmt.Printf("dataset: %s | num docs: %d (train=%d val=%d)\n", datasetPath, len(docs), len(trainDocsText), len(valDocsText))

	tokenizerMode := strings.ToLower(normalize(os.Getenv("TOKENIZER")))
	if tokenizerMode == "" {
		tokenizerMode = "bpe"
	}
	bpeEncoding := normalize(os.Getenv("BPE_ENCODING"))
	if bpeEncoding == "" {
		bpeEncoding = "cl100k_base"
	}
	tokenVocabSize := envInt("TOKEN_VOCAB_SIZE", 2048)

	tokenizer, trainDocsTokens, err := buildTokenizer(tokenizerMode, bpeEncoding, tokenVocabSize, trainDocsText)
	if err != nil {
		fmt.Printf("[warn] tokenizer setup failed for mode=%s (%v), falling back to character mode\n", tokenizerMode, err)
		tokenizer, trainDocsTokens, err = buildTokenizer("char", "", tokenVocabSize, trainDocsText)
		if err != nil {
			panic(err)
		}
	}
	valDocsTokens := make([][]int, 0, len(valDocsText))
	for _, d := range valDocsText {
		valDocsTokens = append(valDocsTokens, tokenizer.encodeDoc(d))
	}
	if len(trainDocsTokens) == 0 {
		panic("tokenized train docs are empty")
	}
	BOS := tokenizer.bosID
	vocabSize := tokenizer.vocabSize()
	fmt.Printf("tokenizer: %s\n", tokenizer.mode)
	if tokenizer.mode == "bpe_cl100k" {
		fmt.Printf("bpe encoding: %s | local bpe vocab: %d (+UNK +BOS)\n", tokenizer.bpeEncoding, len(tokenizer.localToBPE))
	}
	fmt.Printf("vocab size: %d\n", vocabSize)

	nLayer := envInt("N_LAYER", 1)
	nEmbd := envInt("N_EMBD", 48)
	blockSize := envInt("BLOCK_SIZE", 96)
	nHead := envInt("N_HEAD", 4)
	if nLayer < 1 || nEmbd < 1 || blockSize < 2 || nHead < 1 {
		panic("invalid hyperparameters: ensure N_LAYER>=1, N_EMBD>=1, BLOCK_SIZE>=2, N_HEAD>=1")
	}
	if nEmbd%nHead != 0 {
		panic("invalid hyperparameters: N_EMBD must be divisible by N_HEAD")
	}
	requestedDevice := strings.ToLower(normalize(os.Getenv("TRAIN_DEVICE")))
	if requestedDevice == "" {
		requestedDevice = "cpu"
	}
	activeDevice := requestedDevice
	if requestedDevice != "cpu" && requestedDevice != "gpu" {
		panic("invalid TRAIN_DEVICE: use cpu or gpu")
	}
	if requestedDevice == "gpu" {
		activeDevice = "cpu"
		fmt.Println("[warn] TRAIN_DEVICE=gpu requested, but this Go trainer currently runs on CPU kernels. Falling back to CPU.")
	}
	fmt.Printf("device: requested=%s active=%s\n", requestedDevice, activeDevice)
	fmt.Printf("config: n_layer=%d n_embd=%d n_head=%d block_size=%d\n", nLayer, nEmbd, nHead, blockSize)
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

	var params []*Value
	for _, mat := range state {
		for _, row := range mat {
			for _, p := range row {
				params = append(params, p)
			}
		}
	}
	fmt.Printf("num params: %d\n", len(params))

	gpt := buildGPT(state, nLayer, nEmbd, nHead)

	learningRate := envFloat("LEARNING_RATE", 0.004)
	beta1 := envFloat("BETA1", 0.85)
	beta2 := envFloat("BETA2", 0.99)
	epsAdam := envFloat("EPS_ADAM", 1e-8)
	m := make([]float64, len(params))
	v := make([]float64, len(params))

	numSteps := envInt("NUM_STEPS", 800)
	verbose := envBool("VERBOSE", false)
	logLevel := strings.ToLower(normalize(os.Getenv("LOG_LEVEL")))
	if logLevel == "" {
		logLevel = "info"
	}
	debugMode := logLevel == "debug"
	metricInterval := envInt("METRIC_INTERVAL", 25)
	if metricInterval < 1 {
		metricInterval = 1
	}
	if numSteps < 1 {
		panic("invalid NUM_STEPS: must be >=1")
	}
	fmt.Printf("optimizer: lr=%.5f beta1=%.3f beta2=%.3f eps=%.1e steps=%d\n", learningRate, beta1, beta2, epsAdam, numSteps)
	trainDocSampling := strings.ToLower(normalize(os.Getenv("TRAIN_DOC_SAMPLING")))
	if trainDocSampling == "" {
		trainDocSampling = "random"
	}
	if trainDocSampling != "random" && trainDocSampling != "sequential" {
		trainDocSampling = "random"
	}
	fmt.Printf("sampling: train_docs=%s eval_docs=random eval_windows=random\n", trainDocSampling)
	evalInterval := envInt("EVAL_INTERVAL", 50)
	if evalInterval < 1 {
		evalInterval = 100
	}
	evalSteps := envInt("EVAL_STEPS", 16)
	if evalSteps < 1 {
		evalSteps = 64
	}
	earlyStopPatience := envInt("EARLY_STOP_PATIENCE", 8)
	if earlyStopPatience < 1 {
		earlyStopPatience = 8
	}
	earlyStopMinDelta := envFloat("EARLY_STOP_MIN_DELTA", 0.0005)
	if earlyStopMinDelta <= 0 {
		earlyStopMinDelta = 0.0005
	}
	fmt.Printf("validation: interval=%d eval_steps=%d patience=%d min_delta=%.6f\n", evalInterval, evalSteps, earlyStopPatience, earlyStopMinDelta)

	trainStart := time.Now()
	bestValLoss := math.Inf(1)
	bestState := exportState(state)
	noImproveCount := 0
	actualSteps := 0
	for step := 0; step < numSteps; step++ {
		docIdx := step % len(trainDocsTokens)
		if trainDocSampling == "random" {
			docIdx = rand.Intn(len(trainDocsTokens))
		}
		doc := trainDocsText[docIdx]
		docTok := trainDocsTokens[docIdx]
		tokens := make([]int, 0, len(docTok)+2)
		tokens = append(tokens, BOS)
		tokens = append(tokens, docTok...)
		tokens = append(tokens, BOS)

		n := len(tokens) - 1
		if n > blockSize {
			start := rand.Intn(n - blockSize + 1)
			tokens = tokens[start : start+blockSize+1]
			n = blockSize
		}

		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		losses := make([]*Value, 0, n)

		for posID := 0; posID < n; posID++ {
			tokenID, targetID := tokens[posID], tokens[posID+1]
			logits := gpt(tokenID, posID, keys, values)
			probs := softmax(logits)
			lossT := Neg(Log(probs[targetID]))
			losses = append(losses, lossT)
		}

		loss := V(0)
		for _, lt := range losses {
			loss = Add(loss, lt)
		}
		loss = Mul(V(1/float64(n)), loss)

		Backward(loss)

		lrT := learningRate * (1 - float64(step)/float64(numSteps))
		for i, p := range params {
			m[i] = beta1*m[i] + (1-beta1)*p.Grad
			v[i] = beta2*v[i] + (1-beta2)*p.Grad*p.Grad
			mHat := m[i] / (1 - math.Pow(beta1, float64(step+1)))
			vHat := v[i] / (1 - math.Pow(beta2, float64(step+1)))
			p.Data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.Grad = 0
		}
		actualSteps = step + 1

		if verbose {
			elapsed := time.Since(trainStart).Seconds()
			if elapsed <= 0 {
				elapsed = 1e-9
			}
			stepsPerSec := float64(step+1) / elapsed
			remainingSteps := numSteps - (step + 1)
			etaSec := float64(remainingSteps) / stepsPerSec
			mem := &runtime.MemStats{}
			runtime.ReadMemStats(mem)
			sysUsedPct := -1.0
			sysAvailMB := -1.0
			if totalKB, availableKB, ok := readLinuxMemInfo(); ok {
				usedKB := totalKB - availableKB
				sysUsedPct = (float64(usedKB) / float64(totalKB)) * 100.0
				sysAvailMB = float64(availableKB) / 1024.0
			}
			if (step+1)%metricInterval == 0 || step == 0 || step+1 == numSteps {
				fmt.Printf(
					"[step] %d/%d loss=%.4f lr=%.6f seq_len=%d doc_chars=%d steps_per_sec=%.3f elapsed=%s eta=%s heap_alloc_mb=%.2f runtime_sys_mb=%.2f sys_ram_used_pct=%.2f sys_ram_avail_mb=%.2f gc=%d goroutines=%d\n",
					step+1,
					numSteps,
					loss.Data,
					lrT,
					n,
					len([]rune(doc)),
					stepsPerSec,
					time.Since(trainStart).Truncate(time.Second).String(),
					time.Duration(etaSec*float64(time.Second)).Truncate(time.Second).String(),
					float64(mem.Alloc)/1024.0/1024.0,
					float64(mem.Sys)/1024.0/1024.0,
					sysUsedPct,
					sysAvailMB,
					mem.NumGC,
					runtime.NumGoroutine(),
				)
				if debugMode {
					fmt.Printf(
						"[debug] step=%d doc_preview=%q tokens_with_bos=%d learning_rate_t=%.6f\n",
						step+1,
						previewRunes(doc, 120),
						len(tokens),
						lrT,
					)
				}
			}
		} else {
			fmt.Printf("step %4d / %4d | loss %.4f\r", step+1, numSteps, loss.Data)
		}

		if (step+1)%evalInterval == 0 || step+1 == numSteps {
			valLoss := evalLoss(gpt, valDocsTokens, nLayer, blockSize, BOS, evalSteps)
			improved := bestValLoss-valLoss > earlyStopMinDelta
			if improved {
				bestValLoss = valLoss
				bestState = exportState(state)
				noImproveCount = 0
			} else {
				noImproveCount++
			}
			fmt.Printf("[eval] step=%d train_loss=%.4f val_loss=%.4f best_val=%.4f improved=%t patience=%d/%d\n",
				step+1, loss.Data, valLoss, bestValLoss, improved, noImproveCount, earlyStopPatience)
			if noImproveCount >= earlyStopPatience {
				fmt.Printf("[system] early stopping triggered at step %d\n", step+1)
				break
			}
		}
	}

	temperature := envFloat("TEMPERATURE", 0.5)
	sampleCount := envInt("SAMPLE_COUNT", 20)
	if temperature <= 0 {
		panic("invalid TEMPERATURE: must be > 0")
	}
	if sampleCount < 1 {
		panic("invalid SAMPLE_COUNT: must be >=1")
	}
	topK := envInt("TOP_K", 40)
	topP := envFloat("TOP_P", 0.9)
	if topP <= 0 || topP > 1 {
		topP = 0.9
	}
	repetitionPenalty := envFloat("REPETITION_PENALTY", 1.1)
	if repetitionPenalty < 1.0 {
		repetitionPenalty = 1.0
	}
	minNew := envInt("MIN_NEW_TOKENS", 24)
	if minNew < 0 {
		minNew = 0
	}
	sampleMaxNew := envInt("SAMPLE_MAX_NEW_TOKENS", 160)
	if sampleMaxNew < 1 {
		sampleMaxNew = 160
	}
	repeatLastN := envInt("REPEAT_LAST_N", 64)
	if repeatLastN < 1 {
		repeatLastN = 64
	}
	fmt.Println("\n--- inference (generated samples) ---")
	for sampleIdx := 0; sampleIdx < sampleCount; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		tokenID := BOS
		sample := make([]int, 0, sampleMaxNew)
		recent := make([]int, 0, repeatLastN)

		for posID := 0; posID < blockSize && len(sample) < sampleMaxNew; posID++ {
			logits := gpt(tokenID, posID, keys, values)
			recentSet := map[int]bool{}
			for _, id := range recent {
				recentSet[id] = true
			}
			weights := nextTokenWeights(logits, temperature, topK, topP, recentSet, repetitionPenalty)
			if len(sample) < minNew && BOS >= 0 && BOS < len(weights) {
				weights[BOS] = 0
			}
			tokenID = sampleWeighted(weights)
			if tokenID == BOS {
				break
			}
			sample = append(sample, tokenID)
			recent = append(recent, tokenID)
			if len(recent) > repeatLastN {
				recent = recent[len(recent)-repeatLastN:]
			}
		}

		fmt.Printf("sample %2d: %s\n", sampleIdx+1, tokenizer.decodeTokens(sample))
	}

	modelOut := normalize(os.Getenv("MODEL_OUT_PATH"))
	if modelOut == "" {
		if err := os.MkdirAll("models", 0o755); err != nil {
			panic(err)
		}
		runName := normalize(os.Getenv("RUN_NAME"))
		if runName == "" {
			runName = time.Now().Format("20060102_150405")
		}
		runName = slugifyName(runName)
		modelOut = filepath.Join("models", fmt.Sprintf("ckpt_%s_step%04d_valloss%.4f.json", runName, actualSteps, bestValLoss))
	}
	ckpt := TrainingCheckpoint{
		Version:   2,
		CreatedAt: time.Now().Format(time.RFC3339),
		Config: TrainingCheckpointConfig{
			NLayer:    nLayer,
			NEmbd:     nEmbd,
			NHead:     nHead,
			BlockSize: blockSize,
		},
		State: exportState(state),
	}
	if tokenizer.mode == "bpe_cl100k" {
		ckpt.Tokenization = tokenizer.mode
		ckpt.BPEEncoding = tokenizer.bpeEncoding
		ckpt.BPETokenIDs = append([]int(nil), tokenizer.localToBPE...)
	} else {
		ckpt.Tokenization = "char"
		ckpt.Vocab = runesToStrings(tokenizer.localToChar)
	}
	if err := saveCheckpoint(modelOut, ckpt); err != nil {
		fmt.Printf("[model] failed to save checkpoint: %v\n", err)
	} else {
		fmt.Printf("[model] checkpoint saved: %s\n", modelOut)
		latestPath := filepath.Join(filepath.Dir(modelOut), "latest_checkpoint.json")
		if b, err := os.ReadFile(modelOut); err == nil {
			if err := os.WriteFile(latestPath, b, 0o644); err == nil {
				fmt.Printf("[model] latest checkpoint: %s\n", latestPath)
			}
		}
	}
	bestOut := normalize(os.Getenv("MODEL_BEST_PATH"))
	if bestOut == "" {
		bestOut = filepath.Join("models", "best_checkpoint.json")
	}
	bestCkpt := ckpt
	bestCkpt.State = bestState
	if err := saveCheckpoint(bestOut, bestCkpt); err == nil {
		fmt.Printf("[model] best checkpoint saved: %s (best_val=%.4f steps=%d)\n", bestOut, bestValLoss, actualSteps)
	}
}
