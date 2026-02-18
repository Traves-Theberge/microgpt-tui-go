package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

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
	url := "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
	resp, err := http.Get(url)
	if err != nil {
		return copyExampleDataset(path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return copyExampleDataset(path, fmt.Errorf("failed download: %s", resp.Status))
	}
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	scanner := bufio.NewScanner(resp.Body)
	writer := bufio.NewWriter(out)
	bootstrapIdx := 0
	for scanner.Scan() {
		name := strings.TrimSpace(scanner.Text())
		if name == "" {
			continue
		}
		bootstrapIdx++
		rec := DatasetRecord{
			ID:         fmt.Sprintf("bootstrap-%d", bootstrapIdx),
			RecordType: "knowledge",
			Text:       name,
		}
		b, err := json.Marshal(rec)
		if err != nil {
			return err
		}
		if _, err := writer.WriteString(string(b) + "\n"); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return writer.Flush()
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

type TrainingCheckpoint struct {
	Version   int                      `json:"version"`
	CreatedAt string                   `json:"created_at"`
	Config    TrainingCheckpointConfig `json:"config"`
	Vocab     []string                 `json:"vocab"`
	State     map[string][][]float64   `json:"state"`
}

type TrainingCheckpointConfig struct {
	NLayer    int `json:"n_layer"`
	NEmbd     int `json:"n_embd"`
	NHead     int `json:"n_head"`
	BlockSize int `json:"block_size"`
}

func runesToStrings(rs []rune) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = string(r)
	}
	return out
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
	if len(ckpt.Vocab) == 0 {
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
	uchars, err := stringsToRunes(ckpt.Vocab)
	if err != nil {
		return err
	}
	stoi := make(map[rune]int, len(uchars))
	for i, r := range uchars {
		stoi[r] = i
	}
	BOS := len(uchars)

	state := importState(ckpt.State)
	nLayer := ckpt.Config.NLayer
	nEmbd := ckpt.Config.NEmbd
	nHead := ckpt.Config.NHead
	blockSize := ckpt.Config.BlockSize
	gpt := buildGPT(state, nLayer, nEmbd, nHead)

	temp := envFloat("CHAT_TEMPERATURE", 0.6)
	if temp <= 0 {
		temp = 0.6
	}
	maxNew := envInt("CHAT_MAX_NEW_TOKENS", 180)
	if maxNew < 1 {
		maxNew = 180
	}

	promptRunes := []rune(prompt)
	if len(promptRunes) > blockSize-1 {
		promptRunes = promptRunes[len(promptRunes)-(blockSize-1):]
	}
	keys := make([][][]*Value, nLayer)
	values := make([][][]*Value, nLayer)
	tokenID := BOS
	pos := 0
	for _, r := range promptRunes {
		nextID, ok := stoi[r]
		if !ok {
			continue
		}
		if pos >= blockSize {
			break
		}
		_ = gpt(tokenID, pos, keys, values)
		tokenID = nextID
		pos++
	}

	out := make([]rune, 0, maxNew)
	for pos < blockSize && len(out) < maxNew {
		logits := gpt(tokenID, pos, keys, values)
		scaled := make([]*Value, len(logits))
		for i, l := range logits {
			scaled[i] = Div(l, V(temp))
		}
		probs := softmax(scaled)
		weights := make([]float64, len(probs))
		for i, p := range probs {
			weights[i] = p.Data
		}
		tokenID = sampleWeighted(weights)
		if tokenID == BOS {
			break
		}
		out = append(out, uchars[tokenID])
		pos++
	}
	fmt.Print(string(out))
	return nil
}

func main() {
	rand.Seed(42)

	datasetPath := normalize(os.Getenv("DATASET_PATH"))
	if datasetPath == "" {
		datasetPath = "assistant_dataset_train.jsonl"
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
	fmt.Printf("dataset: %s | num docs: %d\n", datasetPath, len(docs))

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

	nLayer := envInt("N_LAYER", 1)
	nEmbd := envInt("N_EMBD", 16)
	blockSize := envInt("BLOCK_SIZE", 16)
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

	learningRate := envFloat("LEARNING_RATE", 0.01)
	beta1 := envFloat("BETA1", 0.85)
	beta2 := envFloat("BETA2", 0.99)
	epsAdam := envFloat("EPS_ADAM", 1e-8)
	m := make([]float64, len(params))
	v := make([]float64, len(params))

	numSteps := envInt("NUM_STEPS", 1000)
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
	trainStart := time.Now()
	for step := 0; step < numSteps; step++ {
		doc := docs[step%len(docs)]
		tokens := []int{BOS}
		for _, ch := range doc {
			tokens = append(tokens, stoi[ch])
		}
		tokens = append(tokens, BOS)

		n := len(tokens) - 1
		if n > blockSize {
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
					len(doc),
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
	}

	temperature := envFloat("TEMPERATURE", 0.5)
	sampleCount := envInt("SAMPLE_COUNT", 20)
	if temperature <= 0 {
		panic("invalid TEMPERATURE: must be > 0")
	}
	if sampleCount < 1 {
		panic("invalid SAMPLE_COUNT: must be >=1")
	}
	fmt.Println("\n--- inference (generated samples) ---")
	for sampleIdx := 0; sampleIdx < sampleCount; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		tokenID := BOS
		sample := make([]rune, 0, blockSize)

		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(tokenID, posID, keys, values)
			scaled := make([]*Value, len(logits))
			for i, l := range logits {
				scaled[i] = Div(l, V(temperature))
			}
			probs := softmax(scaled)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.Data
			}
			tokenID = sampleWeighted(weights)
			if tokenID == BOS {
				break
			}
			sample = append(sample, uchars[tokenID])
		}

		fmt.Printf("sample %2d: %s\n", sampleIdx+1, string(sample))
	}

	modelOut := normalize(os.Getenv("MODEL_OUT_PATH"))
	if modelOut == "" {
		if err := os.MkdirAll("models", 0o755); err != nil {
			panic(err)
		}
		modelOut = filepath.Join("models", fmt.Sprintf("checkpoint_%s.json", time.Now().Format("20060102_150405")))
	}
	ckpt := TrainingCheckpoint{
		Version:   1,
		CreatedAt: time.Now().Format(time.RFC3339),
		Config: TrainingCheckpointConfig{
			NLayer:    nLayer,
			NEmbd:     nEmbd,
			NHead:     nHead,
			BlockSize: blockSize,
		},
		Vocab: runesToStrings(uchars),
		State: exportState(state),
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
}
