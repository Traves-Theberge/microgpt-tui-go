package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/lipgloss"
)

type fieldType int

const (
	fieldString fieldType = iota
	fieldInt
	fieldFloat
	fieldBool
	fieldChoice
)

type cfgField struct {
	Key     string
	Label   string
	Type    fieldType
	Value   string
	Desc    string
	Choices []string
}

type preset struct {
	name        string
	description string
	values      map[string]string
}

type stepMetrics struct {
	step        int
	total       int
	loss        string
	lr          string
	seqLen      string
	docChars    string
	stepsPerSec string
	elapsed     string
	eta         string
	heapMB      string
	runtimeMB   string
	sysRAMPct   string
	sysRAMAvail string
	gc          string
	goroutines  string
}

type evalMetrics struct {
	step         int
	trainLoss    float64
	valLoss      float64
	bestVal      float64
	improved     bool
	patienceUsed int
	patienceMax  int
}

type sysStats struct {
	cpuPct     float64
	memUsedMB  int64
	memFreeMB  int64
	memTotalMB int64
	procRSSKB  int64
	pid        int
}

type cpuSample struct {
	total uint64
	idle  uint64
}

type styles struct {
	title      lipgloss.Style
	tab        lipgloss.Style
	tabActive  lipgloss.Style
	panel      lipgloss.Style
	panelTitle lipgloss.Style
	selected   lipgloss.Style
	dim        lipgloss.Style
	ok         lipgloss.Style
	warn       lipgloss.Style
	graphLoss  lipgloss.Style
	graphSPS   lipgloss.Style
	graphCPU   lipgloss.Style
	graphEval  lipgloss.Style
	graphMem   lipgloss.Style
	splash     lipgloss.Style
	splashText lipgloss.Style
}

type monitorMetric struct {
	Name   string
	Series []float64
	Color  lipgloss.Style
	What   string
	Why    string
	Read   string
}

type model struct {
	width   int
	height  int
	styles  styles
	tabs    []string
	tabIdx  int
	presets []preset

	fields       []cfgField
	fieldIdx     int
	editing      bool
	editor       textinput.Model
	status       string
	running      bool
	latestModel  string
	debugCount   int
	monitorMode  int
	monitorIdx   int
	monitorFocus bool

	cmd          *exec.Cmd
	pid          int
	projectRoot  string
	trainLogPath string
	latestLog    string
	metricsPath  string
	evalPath     string
	runMetaPath  string
	runTag       string
	trainLogFile *os.File
	latestFile   *os.File
	metricsFile  *os.File
	evalFile     *os.File
	lineCh       chan string
	doneCh       chan error

	logs        []string
	trainView   viewport.Model
	logView     viewport.Model
	monitorView viewport.Model
	lastStep    stepMetrics
	lastEval    evalMetrics
	sys         sysStats
	prevCPUS    cpuSample
	spin        spinner.Model
	help        help.Model
	keys        keyMap
	runs        []string
	models      []string
	lastError   string
	gpuInfo     string
	gpuReady    bool

	chatView        viewport.Model
	chatRawLines    []string
	chatLines       []string
	chatPromptInput textinput.Model
	chatPathInput   textinput.Model
	chatEditingPath bool
	chatTyping      bool
	chatModelIdx    int
	chatTemp        float64
	chatMaxTokens   int
	chatWaiting     bool

	datasetPicker      bool
	datasetPickerInput textinput.Model
	datasetCandidates  []string
	datasetFiltered    []string
	datasetPickIdx     int

	lossSeries []float64
	valSeries  []float64
	gapSeries  []float64
	spsSeries  []float64
	tokSeries  []float64
	lrSeries   []float64
	heapSeries []float64
	rtmSeries  []float64
	gorSeries  []float64
	gcSeries   []float64
	pplSeries  []float64
	cpuSeries  []float64
	ramSeries  []float64
	rssSeries  []float64

	lossAnimSeries []float64
	valAnimSeries  []float64
	gapAnimSeries  []float64
	spsAnimSeries  []float64
	tokAnimSeries  []float64
	cpuAnimSeries  []float64
	ramAnimSeries  []float64
	rssAnimSeries  []float64
	lossAnim       float64
	lossVel        float64
	valAnim        float64
	valVel         float64
	gapAnim        float64
	gapVel         float64
	spsAnim        float64
	spsVel         float64
	tokAnim        float64
	tokVel         float64
	cpuAnim        float64
	cpuVel         float64
	ramAnim        float64
	ramVel         float64
	rssAnim        float64
	rssVel         float64
	animPrimed     bool

	splashActive      bool
	splashStarted     time.Time
	splashMinDuration time.Duration
	splashProgress    float64
	splashProgressVel float64
	splashGlow        float64
	splashGlowVel     float64
	splashSpring      harmonica.Spring
	graphSpring       harmonica.Spring
}

type keyMap struct {
	Start    key.Binding
	Stop     key.Binding
	Quit     key.Binding
	TabNext  key.Binding
	TabPrev  key.Binding
	Up       key.Binding
	Down     key.Binding
	Edit     key.Binding
	Apply    key.Binding
	Cancel   key.Binding
	Cycle    key.Binding
	Preset1  key.Binding
	Preset2  key.Binding
	Preset3  key.Binding
	Refresh  key.Binding
	ClearLog key.Binding
	Path     key.Binding
	Latest   key.Binding
	TempUp   key.Binding
	TempDown key.Binding
	TokUp    key.Binding
	TokDown  key.Binding
	FilePick key.Binding
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Start, k.Stop, k.TabNext, k.Edit, k.Quit}
}

func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Start, k.Stop, k.ClearLog, k.Refresh, k.Quit},
		{k.TabNext, k.TabPrev, k.Up, k.Down},
		{k.Edit, k.Apply, k.Cancel, k.Cycle, k.FilePick},
		{k.Preset1, k.Preset2, k.Preset3},
		{k.Path, k.Latest, k.TempDown, k.TempUp, k.TokDown, k.TokUp},
	}
}

var stepRE = regexp.MustCompile(`\[step\]\s+(\d+)/(\d+)\s+loss=([^\s]+)\s+lr=([^\s]+)\s+seq_len=(\d+)\s+doc_chars=(\d+)\s+steps_per_sec=([^\s]+)\s+elapsed=([^\s]+)\s+eta=([^\s]+)\s+heap_alloc_mb=([^\s]+)\s+runtime_sys_mb=([^\s]+)\s+sys_ram_used_pct=([^\s]+)\s+sys_ram_avail_mb=([^\s]+)\s+gc=(\d+)\s+goroutines=(\d+)`)
var evalRE = regexp.MustCompile(`\[eval\]\s+step=(\d+)\s+train_loss=([^\s]+)\s+val_loss=([^\s]+)\s+best_val=([^\s]+)\s+improved=(true|false)\s+patience=(\d+)/(\d+)`)

type sysTickMsg struct {
	stats sysStats
	next  cpuSample
	ts    time.Time
}

type lineMsg string
type doneMsg struct{ err error }
type refreshMsg struct{}
type animTickMsg struct{ ts time.Time }
type chatResponseMsg struct {
	text string
	err  error
}

func defaultStyles() styles {
	brand := lipgloss.AdaptiveColor{Light: "26", Dark: "81"}
	subtle := lipgloss.AdaptiveColor{Light: "245", Dark: "244"}
	border := lipgloss.AdaptiveColor{Light: "250", Dark: "238"}
	return styles{
		title:      lipgloss.NewStyle().Bold(true).Foreground(brand),
		tab:        lipgloss.NewStyle().Padding(0, 1).Foreground(subtle),
		tabActive:  lipgloss.NewStyle().Padding(0, 1).Bold(true).Foreground(lipgloss.Color("15")).Background(brand),
		panel:      lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(border).Padding(0, 1),
		panelTitle: lipgloss.NewStyle().Bold(true).Foreground(brand),
		selected:   lipgloss.NewStyle().Bold(true).Foreground(brand),
		dim:        lipgloss.NewStyle().Foreground(subtle),
		ok:         lipgloss.NewStyle().Foreground(lipgloss.Color("42")).Bold(true),
		warn:       lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true),
		graphLoss:  lipgloss.NewStyle().Foreground(lipgloss.Color("203")),
		graphSPS:   lipgloss.NewStyle().Foreground(lipgloss.Color("86")),
		graphCPU:   lipgloss.NewStyle().Foreground(lipgloss.Color("111")),
		graphEval:  lipgloss.NewStyle().Foreground(lipgloss.Color("214")),
		graphMem:   lipgloss.NewStyle().Foreground(lipgloss.Color("69")),
		splash:     lipgloss.NewStyle().Border(lipgloss.DoubleBorder()).BorderForeground(brand).Padding(1, 3),
		splashText: lipgloss.NewStyle().Bold(true).Foreground(brand),
	}
}

func defaultFields(root string) []cfgField {
	return []cfgField{
		{Key: "DATASET_PATH", Label: "Dataset Path", Type: fieldString, Value: "datasets/raw/databricks-dolly-15k.jsonl", Desc: "JSONL training dataset path"},
		{Key: "TOKENIZER", Label: "Tokenizer", Type: fieldChoice, Value: "bpe", Desc: "bpe (default) or char fallback", Choices: []string{"bpe", "char"}},
		{Key: "BPE_ENCODING", Label: "BPE Encoding", Type: fieldString, Value: "cl100k_base", Desc: "BPE encoding family"},
		{Key: "TOKEN_VOCAB_SIZE", Label: "Local Token Vocab", Type: fieldInt, Value: "2048", Desc: "Top BPE tokens to keep (+UNK,+BOS)"},
		{Key: "N_LAYER", Label: "Layers", Type: fieldInt, Value: "1", Desc: "Transformer layer count"},
		{Key: "N_EMBD", Label: "Embedding Size", Type: fieldInt, Value: "48", Desc: "Embedding width"},
		{Key: "N_HEAD", Label: "Attention Heads", Type: fieldInt, Value: "4", Desc: "Head count"},
		{Key: "BLOCK_SIZE", Label: "Block Size", Type: fieldInt, Value: "96", Desc: "Max sequence length"},
		{Key: "NUM_STEPS", Label: "Training Steps", Type: fieldInt, Value: "800", Desc: "Optimizer steps"},
		{Key: "LEARNING_RATE", Label: "Learning Rate", Type: fieldFloat, Value: "0.004", Desc: "Initial learning rate"},
		{Key: "BETA1", Label: "Adam Beta1", Type: fieldFloat, Value: "0.85", Desc: "Adam momentum term"},
		{Key: "BETA2", Label: "Adam Beta2", Type: fieldFloat, Value: "0.99", Desc: "Adam variance term"},
		{Key: "EPS_ADAM", Label: "Adam Epsilon", Type: fieldFloat, Value: "1e-8", Desc: "Adam stability epsilon"},
		{Key: "VAL_SPLIT", Label: "Validation Split", Type: fieldFloat, Value: "0.10", Desc: "Fraction of docs reserved for validation"},
		{Key: "EVAL_INTERVAL", Label: "Eval Interval", Type: fieldInt, Value: "50", Desc: "Run validation every N steps"},
		{Key: "EVAL_STEPS", Label: "Eval Docs", Type: fieldInt, Value: "16", Desc: "Docs used per validation pass"},
		{Key: "EARLY_STOP_PATIENCE", Label: "Early Stop Patience", Type: fieldInt, Value: "8", Desc: "Validation intervals without improvement before stop"},
		{Key: "EARLY_STOP_MIN_DELTA", Label: "Early Stop Min Delta", Type: fieldFloat, Value: "0.0005", Desc: "Minimum val-loss improvement to reset patience"},
		{Key: "TEMPERATURE", Label: "Sample Temperature", Type: fieldFloat, Value: "0.60", Desc: "Generation randomness"},
		{Key: "SAMPLE_COUNT", Label: "Sample Count", Type: fieldInt, Value: "8", Desc: "Number of output samples"},
		{Key: "SAMPLE_MAX_NEW_TOKENS", Label: "Sample Max New Tokens", Type: fieldInt, Value: "160", Desc: "Max generated tokens per sample"},
		{Key: "TOP_K", Label: "Top-K", Type: fieldInt, Value: "40", Desc: "Limit next-token candidates to top K"},
		{Key: "TOP_P", Label: "Top-P", Type: fieldFloat, Value: "0.90", Desc: "Nucleus sampling cumulative probability"},
		{Key: "REPETITION_PENALTY", Label: "Repetition Penalty", Type: fieldFloat, Value: "1.10", Desc: "Penalize recently seen tokens"},
		{Key: "MIN_NEW_TOKENS", Label: "Min New Tokens", Type: fieldInt, Value: "24", Desc: "Try to avoid immediate EOS"},
		{Key: "REPEAT_LAST_N", Label: "Repeat Window", Type: fieldInt, Value: "64", Desc: "Recent token window for repetition penalty"},
		{Key: "TRAIN_DEVICE", Label: "Train Device", Type: fieldChoice, Value: "cpu", Desc: "Requested compute device", Choices: []string{"cpu", "gpu"}},
		{Key: "METRIC_INTERVAL", Label: "Metric Interval", Type: fieldInt, Value: "1", Desc: "Verbose metric log cadence"},
		{Key: "LOG_LEVEL", Label: "Log Level", Type: fieldChoice, Value: "debug", Desc: "info or debug", Choices: []string{"info", "debug"}},
		{Key: "VERBOSE", Label: "Verbose Metrics", Type: fieldBool, Value: "true", Desc: "Always on for dashboard"},
		{Key: "MODEL_OUT_PATH", Label: "Model Output", Type: fieldString, Value: "", Desc: "Optional custom checkpoint path (empty = auto naming)"},
	}
}

func defaultPresets() []preset {
	return []preset{
		{name: "fast", description: "quick smoke run", values: map[string]string{"TOKENIZER": "bpe", "TOKEN_VOCAB_SIZE": "1536", "N_LAYER": "1", "N_EMBD": "32", "N_HEAD": "4", "BLOCK_SIZE": "64", "NUM_STEPS": "400", "LEARNING_RATE": "0.0045", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "4"}},
		{name: "balanced", description: "laptop-safe baseline", values: map[string]string{"TOKENIZER": "bpe", "TOKEN_VOCAB_SIZE": "2048", "N_LAYER": "1", "N_EMBD": "48", "N_HEAD": "4", "BLOCK_SIZE": "96", "NUM_STEPS": "800", "LEARNING_RATE": "0.004", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "8"}},
		{name: "max", description: "stronger run", values: map[string]string{"TOKENIZER": "bpe", "TOKEN_VOCAB_SIZE": "3072", "N_LAYER": "2", "N_EMBD": "64", "N_HEAD": "4", "BLOCK_SIZE": "128", "NUM_STEPS": "1800", "LEARNING_RATE": "0.0038", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "10"}},
	}
}

func initialModel() model {
	root := projectRoot()
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("81"))

	ed := textinput.New()
	ed.CharLimit = 512
	ed.Width = 36

	prompt := textinput.New()
	prompt.Placeholder = "Type your prompt and press Enter"
	prompt.CharLimit = 1200
	prompt.Blur()

	pathIn := textinput.New()
	pathIn.CharLimit = 600
	pathIn.SetValue(filepath.Join(root, "models", "latest_checkpoint.json"))

	dsPick := textinput.New()
	dsPick.CharLimit = 240
	dsPick.Placeholder = "filter datasets"
	dsPick.Width = 46

	logVP := viewport.New(100, 16)
	logVP.SetContent("logs will appear here")
	trainVP := viewport.New(100, 16)
	trainVP.SetContent("train will appear here")
	monVP := viewport.New(100, 20)
	monVP.SetContent("monitor will appear here")
	chatVP := viewport.New(100, 16)
	chatVP.SetContent("chat will appear here")

	m := model{
		styles:             defaultStyles(),
		tabs:               []string{"Train", "Monitor", "Logs", "Runs", "Models", "Chat"},
		tabIdx:             0,
		presets:            defaultPresets(),
		fields:             defaultFields(root),
		fieldIdx:           0,
		editor:             ed,
		status:             "idle",
		lineCh:             make(chan string, 4096),
		doneCh:             make(chan error, 1),
		spin:               sp,
		trainView:          trainVP,
		logView:            logVP,
		monitorView:        monVP,
		chatView:           chatVP,
		chatPromptInput:    prompt,
		chatPathInput:      pathIn,
		datasetPickerInput: dsPick,
		chatTemp:           0.60,
		chatMaxTokens:      220,
		help:               help.New(),
		projectRoot:        root,
		splashActive:       true,
		splashStarted:      time.Now(),
		splashMinDuration:  2200 * time.Millisecond,
		splashSpring:       harmonica.NewSpring(harmonica.FPS(30), 8.0, 0.72),
		graphSpring:        harmonica.NewSpring(harmonica.FPS(30), 6.0, 1.0),
		keys: keyMap{
			Start:    key.NewBinding(key.WithKeys("s"), key.WithHelp("s", "start")),
			Stop:     key.NewBinding(key.WithKeys("x"), key.WithHelp("x", "stop")),
			Quit:     key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
			TabNext:  key.NewBinding(key.WithKeys("tab", "l"), key.WithHelp("tab/l", "next tab")),
			TabPrev:  key.NewBinding(key.WithKeys("shift+tab", "h"), key.WithHelp("shift+tab/h", "prev tab")),
			Up:       key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("up/k", "up")),
			Down:     key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("down/j", "down")),
			Edit:     key.NewBinding(key.WithKeys("e"), key.WithHelp("e", "edit")),
			Apply:    key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "apply/send")),
			Cancel:   key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel edit")),
			Cycle:    key.NewBinding(key.WithKeys("space"), key.WithHelp("space", "cycle toggle")),
			Preset1:  key.NewBinding(key.WithKeys("1"), key.WithHelp("1", "preset fast")),
			Preset2:  key.NewBinding(key.WithKeys("2"), key.WithHelp("2", "preset balanced")),
			Preset3:  key.NewBinding(key.WithKeys("3"), key.WithHelp("3", "preset max")),
			Refresh:  key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "refresh lists")),
			ClearLog: key.NewBinding(key.WithKeys("c"), key.WithHelp("c", "clear logs/chat")),
			Path:     key.NewBinding(key.WithKeys("p"), key.WithHelp("p", "edit chat path")),
			Latest:   key.NewBinding(key.WithKeys("L"), key.WithHelp("L", "latest checkpoint")),
			TempDown: key.NewBinding(key.WithKeys("["), key.WithHelp("[", "chat temp -0.05")),
			TempUp:   key.NewBinding(key.WithKeys("]"), key.WithHelp("]", "chat temp +0.05")),
			TokDown:  key.NewBinding(key.WithKeys("-"), key.WithHelp("-", "chat tokens -10")),
			TokUp:    key.NewBinding(key.WithKeys("="), key.WithHelp("=", "chat tokens +10")),
			FilePick: key.NewBinding(key.WithKeys("f"), key.WithHelp("f", "dataset picker")),
		},
	}
	m.addChatLine("[system] chat ready")
	m.addChatLine("[system] checkpoint: " + m.chatPathInput.Value())
	m.refreshLists()
	m.detectGPU()
	m.datasetCandidates = listDatasetCandidates(root)
	m.applyDatasetFilter()
	return m
}

func projectRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		cwd, err := os.Getwd()
		if err != nil {
			return "."
		}
		return cwd
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../.."))
}

func (m model) Init() tea.Cmd {
	return tea.Batch(m.spin.Tick, waitLineCmd(m.lineCh), waitDoneCmd(m.doneCh), sysTickCmd(m.pid, m.prevCPUS), refreshCmd(), animTickCmd())
}

func waitLineCmd(ch <-chan string) tea.Cmd {
	return func() tea.Msg {
		ln := <-ch
		return lineMsg(ln)
	}
}

func waitDoneCmd(ch <-chan error) tea.Cmd {
	return func() tea.Msg {
		err := <-ch
		return doneMsg{err: err}
	}
}

func sysTickCmd(pid int, prev cpuSample) tea.Cmd {
	return tea.Tick(1*time.Second, func(now time.Time) tea.Msg {
		stats, next := sampleSystem(pid, prev)
		return sysTickMsg{stats: stats, next: next, ts: now}
	})
}

func refreshCmd() tea.Cmd {
	return tea.Tick(2*time.Second, func(time.Time) tea.Msg { return refreshMsg{} })
}

func animTickCmd() tea.Cmd {
	return tea.Tick(time.Second/30, func(ts time.Time) tea.Msg { return animTickMsg{ts: ts} })
}

func callChatCmd(root, checkpointPath, prompt string, temperature float64, maxNew int) tea.Cmd {
	return func() tea.Msg {
		cmd := exec.Command("go", "run", ".", "chat-once", checkpointPath, prompt)
		cmd.Dir = root
		cmd.Env = append(os.Environ(),
			"CHAT_TEMPERATURE="+fmt.Sprintf("%.2f", temperature),
			"CHAT_MAX_NEW_TOKENS="+strconv.Itoa(maxNew),
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			return chatResponseMsg{err: fmt.Errorf("%v | %s", err, strings.TrimSpace(string(out)))}
		}
		return chatResponseMsg{text: strings.TrimSpace(string(out))}
	}
}

func (m *model) addChatLine(line string) {
	m.chatRawLines = append(m.chatRawLines, line)
	if len(m.chatRawLines) > 2500 {
		m.chatRawLines = m.chatRawLines[len(m.chatRawLines)-2500:]
	}
	m.rebuildChatView()
}

func (m *model) rebuildChatView() {
	contentW := max(44, m.width-4)
	leftW := max(42, int(float64(contentW)*0.66))
	chatW := max(34, leftW-6)
	maxW := max(20, chatW-2)
	out := make([]string, 0, len(m.chatRawLines)*2)
	for _, ln := range m.chatRawLines {
		out = append(out, wrapText(ln, maxW)...)
	}
	m.chatLines = out
	m.chatView.SetContent(strings.Join(m.chatLines, "\n"))
	m.chatView.GotoBottom()
}

func (m *model) fieldByKey(key string) *cfgField {
	for i := range m.fields {
		if m.fields[i].Key == key {
			return &m.fields[i]
		}
	}
	return nil
}

func (m *model) applyPreset(idx int) {
	if idx < 0 || idx >= len(m.presets) {
		return
	}
	for k, v := range m.presets[idx].values {
		if f := m.fieldByKey(k); f != nil {
			f.Value = v
		}
	}
	m.appendLog("[system] preset loaded: " + m.presets[idx].name)
}

func (m *model) cycleField(i int) {
	if i < 0 || i >= len(m.fields) {
		return
	}
	f := &m.fields[i]
	switch f.Type {
	case fieldBool:
		if strings.EqualFold(f.Value, "true") {
			f.Value = "false"
		} else {
			f.Value = "true"
		}
	case fieldChoice:
		if len(f.Choices) == 0 {
			return
		}
		idx := 0
		for j, c := range f.Choices {
			if c == f.Value {
				idx = j
				break
			}
		}
		f.Value = f.Choices[(idx+1)%len(f.Choices)]
	}
}

func (m *model) startEdit() {
	if m.running || m.tabIdx != 0 {
		return
	}
	f := m.fields[m.fieldIdx]
	m.editing = true
	m.editor.SetValue(f.Value)
	m.editor.Placeholder = f.Label
	m.editor.Focus()
}

func (m *model) applyEdit() {
	if !m.editing {
		return
	}
	m.fields[m.fieldIdx].Value = strings.TrimSpace(m.editor.Value())
	m.editing = false
	m.editor.Blur()
}

func (m *model) cancelEdit() {
	if !m.editing {
		return
	}
	m.editing = false
	m.editor.Blur()
}

func (m *model) validateFields() error {
	intMin := map[string]int{
		"TOKEN_VOCAB_SIZE":      512,
		"N_LAYER":               1,
		"N_EMBD":                1,
		"N_HEAD":                1,
		"BLOCK_SIZE":            2,
		"NUM_STEPS":             1,
		"SAMPLE_COUNT":          1,
		"SAMPLE_MAX_NEW_TOKENS": 1,
		"TOP_K":                 0,
		"MIN_NEW_TOKENS":        0,
		"REPEAT_LAST_N":         1,
		"EVAL_INTERVAL":         1,
		"EVAL_STEPS":            1,
		"EARLY_STOP_PATIENCE":   1,
		"METRIC_INTERVAL":       1,
	}
	floatMinExclusive := map[string]float64{"LEARNING_RATE": 0, "EPS_ADAM": 0, "TEMPERATURE": 0, "VAL_SPLIT": 0, "EARLY_STOP_MIN_DELTA": 0, "TOP_P": 0}
	floatMinInclusive := map[string]float64{"BETA1": 0, "BETA2": 0}
	floatMaxInclusive := map[string]float64{"BETA1": 1, "BETA2": 1}

	vals := map[string]string{}
	for _, f := range m.fields {
		vals[f.Key] = strings.TrimSpace(f.Value)
	}
	for k, mn := range intMin {
		n, err := strconv.Atoi(vals[k])
		if err != nil || n < mn {
			return fmt.Errorf("%s must be an integer >= %d", k, mn)
		}
	}
	for k, mn := range floatMinExclusive {
		n, err := strconv.ParseFloat(vals[k], 64)
		if err != nil || n <= mn {
			return fmt.Errorf("%s must be > %v", k, mn)
		}
	}
	for k, mn := range floatMinInclusive {
		n, err := strconv.ParseFloat(vals[k], 64)
		if err != nil || n < mn {
			return fmt.Errorf("%s must be >= %v", k, mn)
		}
	}
	for k, mx := range floatMaxInclusive {
		n, _ := strconv.ParseFloat(vals[k], 64)
		if n > mx {
			return fmt.Errorf("%s must be <= %v", k, mx)
		}
	}
	nEmbd, _ := strconv.Atoi(vals["N_EMBD"])
	nHead, _ := strconv.Atoi(vals["N_HEAD"])
	if nEmbd%nHead != 0 {
		return fmt.Errorf("N_EMBD must be divisible by N_HEAD")
	}
	if vals["TOKENIZER"] != "bpe" && vals["TOKENIZER"] != "char" {
		return fmt.Errorf("TOKENIZER must be bpe or char")
	}
	valSplit, _ := strconv.ParseFloat(vals["VAL_SPLIT"], 64)
	if valSplit >= 0.5 {
		return fmt.Errorf("VAL_SPLIT must be < 0.5")
	}
	topP, _ := strconv.ParseFloat(vals["TOP_P"], 64)
	if topP > 1 {
		return fmt.Errorf("TOP_P must be <= 1")
	}
	dev := strings.ToLower(vals["TRAIN_DEVICE"])
	if dev != "cpu" && dev != "gpu" {
		return fmt.Errorf("TRAIN_DEVICE must be cpu or gpu")
	}
	if vals["DATASET_PATH"] == "" {
		return fmt.Errorf("DATASET_PATH cannot be empty")
	}
	return nil
}

func (m *model) envMap() map[string]string {
	out := map[string]string{}
	for _, f := range m.fields {
		out[f.Key] = strings.TrimSpace(f.Value)
	}
	return out
}

func (m *model) appendLog(line string) {
	m.logs = append(m.logs, line)
	if len(m.logs) > 3500 {
		m.logs = m.logs[len(m.logs)-3500:]
	}
	m.logView.SetContent(strings.Join(m.logs, "\n"))
	m.logView.GotoBottom()
	if m.trainLogFile != nil {
		_, _ = m.trainLogFile.WriteString(line + "\n")
	}
	if m.latestFile != nil {
		_, _ = m.latestFile.WriteString(line + "\n")
	}
}

func (m *model) appendSystemCSV(ts time.Time, stats sysStats) {
	if m.metricsFile == nil {
		return
	}
	stamp := ts.Format("2006-01-02T15:04:05")
	line := fmt.Sprintf("%s,%d,%.2f,%d,%d,%d,%d,%d,%d", stamp, stats.pid, stats.cpuPct, stats.procRSSKB, stats.memUsedMB, stats.memFreeMB, stats.memTotalMB, m.lastStep.step, m.lastStep.total)
	_, _ = m.metricsFile.WriteString(line + "\n")
}

func (m *model) appendEvalCSV(ts time.Time, em evalMetrics) {
	if m.evalFile == nil {
		return
	}
	stamp := ts.Format("2006-01-02T15:04:05")
	gap := em.valLoss - em.trainLoss
	line := fmt.Sprintf(
		"%s,%d,%.6f,%.6f,%.6f,%.6f,%t,%d,%d,%.2f,%d,%d,%d",
		stamp,
		em.step,
		em.trainLoss,
		em.valLoss,
		em.bestVal,
		gap,
		em.improved,
		em.patienceUsed,
		em.patienceMax,
		m.sys.cpuPct,
		m.sys.memUsedMB,
		m.sys.memTotalMB,
		m.sys.procRSSKB,
	)
	_, _ = m.evalFile.WriteString(line + "\n")
}

func (m *model) openRunLogs(presetName string) error {
	logRoot := filepath.Join(m.projectRoot, "logs")
	trainDir := filepath.Join(logRoot, "train")
	systemDir := filepath.Join(logRoot, "system")
	evalDir := filepath.Join(logRoot, "eval")
	runsDir := filepath.Join(logRoot, "runs")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(systemDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(evalDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	ts := time.Now().Format("20060102_150405")
	m.runTag = fmt.Sprintf("%s_%s", presetName, ts)
	m.trainLogPath = filepath.Join(trainDir, fmt.Sprintf("tui_train_%s.log", m.runTag))
	m.metricsPath = filepath.Join(systemDir, fmt.Sprintf("tui_system_metrics_%s.csv", m.runTag))
	m.evalPath = filepath.Join(evalDir, fmt.Sprintf("tui_eval_metrics_%s.csv", m.runTag))
	m.runMetaPath = filepath.Join(runsDir, fmt.Sprintf("run_%s.txt", m.runTag))
	m.latestLog = filepath.Join(logRoot, "train_latest.log")

	lf, err := os.OpenFile(m.trainLogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	lat, err := os.OpenFile(m.latestLog, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		_ = lf.Close()
		return err
	}
	mf, err := os.OpenFile(m.metricsPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		_ = lf.Close()
		_ = lat.Close()
		return err
	}
	ef, err := os.OpenFile(m.evalPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		_ = lf.Close()
		_ = lat.Close()
		_ = mf.Close()
		return err
	}
	m.trainLogFile = lf
	m.latestFile = lat
	m.metricsFile = mf
	m.evalFile = ef
	_, _ = m.metricsFile.WriteString("timestamp,pid,cpu_percent,proc_rss_kb,sys_mem_used_mb,sys_mem_free_mb,sys_mem_total_mb,step,step_total\n")
	_, _ = m.evalFile.WriteString("timestamp,step,train_loss,val_loss,best_val,generalization_gap,improved,patience_used,patience_max,cpu_percent,sys_mem_used_mb,sys_mem_total_mb,proc_rss_kb\n")
	_ = os.WriteFile(m.runMetaPath, []byte("run_tag="+m.runTag+"\n"), 0o644)
	return nil
}

func (m *model) closeRunLogs() {
	if m.trainLogFile != nil {
		_ = m.trainLogFile.Close()
		m.trainLogFile = nil
	}
	if m.latestFile != nil {
		_ = m.latestFile.Close()
		m.latestFile = nil
	}
	if m.metricsFile != nil {
		_ = m.metricsFile.Close()
		m.metricsFile = nil
	}
	if m.evalFile != nil {
		_ = m.evalFile.Close()
		m.evalFile = nil
	}
}

func (m *model) startTraining() {
	if m.running {
		m.appendLog("[system] training already running")
		return
	}
	if err := m.validateFields(); err != nil {
		m.lastError = err.Error()
		m.appendLog("[system] config validation failed: " + err.Error())
		return
	}
	env := m.envMap()
	presetName := "custom"
	if f := m.fieldByKey("NUM_STEPS"); f != nil {
		presetName = "steps" + f.Value
	}
	if err := m.openRunLogs(presetName); err != nil {
		m.lastError = err.Error()
		m.appendLog("[system] failed to open logs: " + err.Error())
		return
	}

	cmd := exec.Command("go", "run", ".")
	cmd.Dir = m.projectRoot
	cmd.Env = os.Environ()
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	if strings.TrimSpace(m.runTag) != "" {
		cmd.Env = append(cmd.Env, "RUN_NAME="+m.runTag)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		m.appendLog("[system] failed stdout pipe: " + err.Error())
		m.closeRunLogs()
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		m.appendLog("[system] failed stderr pipe: " + err.Error())
		m.closeRunLogs()
		return
	}

	m.appendLog("[system] starting training with current dashboard configuration")
	if err := cmd.Start(); err != nil {
		m.appendLog("[system] failed start: " + err.Error())
		m.closeRunLogs()
		return
	}
	m.cmd = cmd
	m.pid = cmd.Process.Pid
	m.running = true
	m.status = "running"
	m.lastStep = stepMetrics{}
	m.lastEval = evalMetrics{}
	m.latestModel = ""
	m.debugCount = 0
	m.monitorMode = 0
	m.monitorIdx = 0
	m.monitorFocus = false
	m.lossSeries = nil
	m.valSeries = nil
	m.gapSeries = nil
	m.spsSeries = nil
	m.tokSeries = nil
	m.lrSeries = nil
	m.heapSeries = nil
	m.rtmSeries = nil
	m.gorSeries = nil
	m.gcSeries = nil
	m.pplSeries = nil
	m.cpuSeries = nil
	m.ramSeries = nil
	m.rssSeries = nil
	m.lossAnimSeries = nil
	m.valAnimSeries = nil
	m.gapAnimSeries = nil
	m.spsAnimSeries = nil
	m.tokAnimSeries = nil
	m.cpuAnimSeries = nil
	m.ramAnimSeries = nil
	m.rssAnimSeries = nil
	m.animPrimed = false
	m.appendLog(fmt.Sprintf("[system] pid=%d", m.pid))
	m.appendLog("[system] run tag: " + m.runTag)
	m.appendLog("[system] train log: " + m.trainLogPath)
	m.appendLog("[system] system metrics: " + m.metricsPath)
	m.appendLog("[system] eval metrics: " + m.evalPath)
	m.appendLog("[system] run meta: " + m.runMetaPath)
	if m.runMetaPath != "" {
		var meta strings.Builder
		meta.WriteString("run_tag=" + m.runTag + "\n")
		meta.WriteString("started_at=" + time.Now().Format(time.RFC3339) + "\n")
		meta.WriteString("dataset=" + env["DATASET_PATH"] + "\n")
		meta.WriteString("train_log=" + m.trainLogPath + "\n")
		meta.WriteString("system_metrics=" + m.metricsPath + "\n")
		meta.WriteString("eval_metrics=" + m.evalPath + "\n")
		for _, f := range m.fields {
			meta.WriteString(f.Key + "=" + strings.TrimSpace(f.Value) + "\n")
		}
		_ = os.WriteFile(m.runMetaPath, []byte(meta.String()), 0o644)
	}

	pump := func(sc *bufio.Scanner) {
		for sc.Scan() {
			m.lineCh <- sc.Text()
		}
	}
	go pump(bufio.NewScanner(stdout))
	go pump(bufio.NewScanner(stderr))
	go func() { m.doneCh <- cmd.Wait() }()
}

func (m *model) stopTraining() {
	if !m.running || m.cmd == nil || m.cmd.Process == nil {
		m.appendLog("[system] no active training process")
		return
	}
	m.appendLog("[system] stop requested")
	_ = m.cmd.Process.Signal(syscall.SIGINT)
	go func(proc *os.Process) {
		time.Sleep(800 * time.Millisecond)
		_ = proc.Kill()
	}(m.cmd.Process)
}

func (m *model) refreshLists() {
	m.runs = buildRunSummary(filepath.Join(m.projectRoot, "logs"))
	m.models = listFilesByPattern(filepath.Join(m.projectRoot, "models"), "*.json", 16)
	if len(m.models) == 0 {
		m.chatModelIdx = 0
	} else {
		curBase := filepath.Base(strings.TrimSpace(m.chatPathInput.Value()))
		for i, name := range m.models {
			if name == curBase {
				m.chatModelIdx = i
				break
			}
		}
		if m.chatModelIdx < 0 {
			m.chatModelIdx = 0
		}
		if m.chatModelIdx >= len(m.models) {
			m.chatModelIdx = len(m.models) - 1
		}
	}
	if m.latestModel == "" {
		latest := filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
		if _, err := os.Stat(latest); err == nil {
			m.latestModel = latest
		}
	}
	m.datasetCandidates = listDatasetCandidates(m.projectRoot)
	m.applyDatasetFilter()
	m.detectGPU()
}

func (m *model) detectGPU() {
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
	out, err := cmd.Output()
	if err != nil {
		m.gpuInfo = "NVIDIA GPU: not detected (CPU mode active)"
		m.gpuReady = false
		return
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		m.gpuInfo = "NVIDIA GPU: detected but details unavailable"
		m.gpuReady = true
		return
	}
	m.gpuInfo = "NVIDIA GPU: " + strings.TrimSpace(lines[0]) + " | trainer kernels: CPU today"
	m.gpuReady = true
}

func buildRunSummary(logRoot string) []string {
	train := listFilesByPattern(filepath.Join(logRoot, "train"), "tui_train_*.log", 8)
	system := listFilesByPattern(filepath.Join(logRoot, "system"), "tui_system_metrics_*.csv", 8)
	eval := listFilesByPattern(filepath.Join(logRoot, "eval"), "tui_eval_metrics_*.csv", 8)
	manifests := listFilesByPattern(filepath.Join(logRoot, "runs"), "run_*.txt", 8)
	lines := []string{}
	appendSection := func(name string, items []string) {
		lines = append(lines, name+":")
		if len(items) == 0 {
			lines = append(lines, "  (none)")
		} else {
			for _, it := range items {
				lines = append(lines, "  "+it)
			}
		}
		lines = append(lines, "")
	}
	appendSection("Train Logs", train)
	appendSection("System Metrics", system)
	appendSection("Eval Metrics", eval)
	appendSection("Run Manifests", manifests)
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func listDatasetCandidates(root string) []string {
	paths := []string{
		filepath.Join(root, "assistant_dataset_train.jsonl"),
		filepath.Join(root, "datasets"),
		filepath.Join(root, "data_sources"),
	}
	cands := make([]string, 0, 64)
	seen := map[string]bool{}
	add := func(path string) {
		if !strings.HasSuffix(strings.ToLower(path), ".jsonl") {
			return
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return
		}
		rel = filepath.Clean(rel)
		if rel == "." || rel == "" || seen[rel] {
			return
		}
		seen[rel] = true
		cands = append(cands, rel)
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		if !info.IsDir() {
			add(p)
			continue
		}
		_ = filepath.WalkDir(p, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				name := strings.ToLower(d.Name())
				if name == ".git" || name == "logs" || name == "models" {
					return filepath.SkipDir
				}
				return nil
			}
			add(path)
			return nil
		})
	}
	sort.Strings(cands)
	return cands
}

func (m *model) applyDatasetFilter() {
	q := strings.ToLower(strings.TrimSpace(m.datasetPickerInput.Value()))
	filtered := make([]string, 0, len(m.datasetCandidates))
	for _, p := range m.datasetCandidates {
		if q == "" || strings.Contains(strings.ToLower(p), q) {
			filtered = append(filtered, p)
		}
	}
	m.datasetFiltered = filtered
	if len(filtered) == 0 {
		m.datasetPickIdx = 0
		return
	}
	if m.datasetPickIdx < 0 {
		m.datasetPickIdx = 0
	}
	if m.datasetPickIdx >= len(filtered) {
		m.datasetPickIdx = len(filtered) - 1
	}
}

func (m *model) openDatasetPicker() {
	m.datasetPicker = true
	m.datasetPickIdx = 0
	m.datasetPickerInput.SetValue("")
	m.datasetPickerInput.Focus()
	m.applyDatasetFilter()
}

func (m *model) closeDatasetPicker() {
	m.datasetPicker = false
	m.datasetPickerInput.Blur()
}

func listFilesByPattern(dir, pattern string, limit int) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	type f struct {
		name string
		mod  time.Time
	}
	arr := make([]f, 0)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ok, _ := filepath.Match(pattern, e.Name())
		if !ok {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		arr = append(arr, f{name: e.Name(), mod: info.ModTime()})
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].mod.After(arr[j].mod) })
	if len(arr) > limit {
		arr = arr[:limit]
	}
	out := make([]string, 0, len(arr))
	for _, it := range arr {
		out = append(out, it.name)
	}
	return out
}

func (m *model) selectChatModelByIndex(idx int) {
	if len(m.models) == 0 {
		return
	}
	if idx < 0 {
		idx = 0
	}
	if idx >= len(m.models) {
		idx = len(m.models) - 1
	}
	m.chatModelIdx = idx
	full := filepath.Join(m.projectRoot, "models", m.models[m.chatModelIdx])
	m.chatPathInput.SetValue(full)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	m.spin, cmd = m.spin.Update(msg)
	cmds = append(cmds, cmd)

	if m.editing {
		m.editor, cmd = m.editor.Update(msg)
		cmds = append(cmds, cmd)
	}
	if m.datasetPicker {
		m.datasetPickerInput, cmd = m.datasetPickerInput.Update(msg)
		cmds = append(cmds, cmd)
	}

	if m.tabIdx == 5 {
		if m.chatEditingPath {
			m.chatPathInput, cmd = m.chatPathInput.Update(msg)
			cmds = append(cmds, cmd)
		} else if m.chatTyping {
			m.chatPromptInput, cmd = m.chatPromptInput.Update(msg)
			cmds = append(cmds, cmd)
		}
	}

	m.logView, cmd = m.logView.Update(msg)
	cmds = append(cmds, cmd)
	m.trainView, cmd = m.trainView.Update(msg)
	cmds = append(cmds, cmd)
	m.monitorView, cmd = m.monitorView.Update(msg)
	cmds = append(cmds, cmd)
	m.chatView, cmd = m.chatView.Update(msg)
	cmds = append(cmds, cmd)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.trainView.Width = max(60, m.width-8)
		m.trainView.Height = max(9, m.height-20)
		m.logView.Width = max(60, m.width-8)
		m.logView.Height = max(9, m.height-20)
		m.monitorView.Width = max(60, m.width-8)
		m.monitorView.Height = max(8, m.height-18)
		contentW := max(70, m.width-4)
		leftW := max(42, int(float64(contentW)*0.66))
		m.chatView.Width = max(34, leftW-6)
		m.chatView.Height = max(10, m.height-22)
		m.editor.Width = max(24, min(64, m.width/2))
		m.datasetPickerInput.Width = max(24, min(72, m.width/2))
		m.chatPromptInput.Width = max(28, m.width-12)
		m.chatPathInput.Width = max(28, m.width-12)
		m.rebuildChatView()

	case tea.KeyMsg:
		s := msg.String()
		if m.datasetPicker {
			switch s {
			case "ctrl+c":
				if m.running {
					m.stopTraining()
				}
				m.closeRunLogs()
				return m, tea.Quit
			case "esc":
				m.closeDatasetPicker()
			case "enter":
				if len(m.datasetFiltered) > 0 {
					p := m.datasetFiltered[m.datasetPickIdx]
					if f := m.fieldByKey("DATASET_PATH"); f != nil {
						f.Value = p
					}
					m.appendLog("[system] dataset selected: " + p)
				}
				m.closeDatasetPicker()
			case "up":
				m.datasetPickIdx = max(0, m.datasetPickIdx-1)
			case "down":
				m.datasetPickIdx = min(max(0, len(m.datasetFiltered)-1), m.datasetPickIdx+1)
			case "pgup", "ctrl+u":
				m.datasetPickIdx = max(0, m.datasetPickIdx-8)
			case "pgdown", "ctrl+d":
				m.datasetPickIdx = min(max(0, len(m.datasetFiltered)-1), m.datasetPickIdx+8)
			case "home":
				m.datasetPickIdx = 0
			case "end":
				m.datasetPickIdx = max(0, len(m.datasetFiltered)-1)
			default:
				m.applyDatasetFilter()
			}
			return m, tea.Batch(cmds...)
		}
		if m.editing {
			switch s {
			case "enter":
				m.applyEdit()
			case "esc":
				m.cancelEdit()
			}
			break
		}
		if m.tabIdx == 5 && !m.splashActive {
			if m.chatEditingPath {
				switch s {
				case "ctrl+c":
					if m.running {
						m.stopTraining()
					}
					m.closeRunLogs()
					return m, tea.Quit
				case "enter":
					m.chatEditingPath = false
					m.chatPathInput.Blur()
					m.chatPromptInput.Blur()
					m.addChatLine("[system] checkpoint set: " + strings.TrimSpace(m.chatPathInput.Value()))
				case "esc":
					m.chatEditingPath = false
					m.chatPathInput.Blur()
				}
				return m, tea.Batch(cmds...)
			}
			if m.chatTyping {
				switch s {
				case "ctrl+c":
					if m.running {
						m.stopTraining()
					}
					m.closeRunLogs()
					return m, tea.Quit
				case "enter":
					if m.chatWaiting {
						return m, tea.Batch(cmds...)
					}
					prompt := strings.TrimSpace(m.chatPromptInput.Value())
					if prompt == "" {
						m.chatTyping = false
						m.chatPromptInput.Blur()
						return m, tea.Batch(cmds...)
					}
					ckpt := strings.TrimSpace(m.chatPathInput.Value())
					if ckpt == "" {
						ckpt = filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
						m.chatPathInput.SetValue(ckpt)
					}
					if _, err := os.Stat(ckpt); err != nil {
						m.addChatLine("[system] checkpoint not found: " + ckpt)
						return m, tea.Batch(cmds...)
					}
					m.chatWaiting = true
					m.addChatLine("[you] " + prompt)
					m.addChatLine("[assistant] thinking...")
					m.chatPromptInput.SetValue("")
					m.chatTyping = false
					m.chatPromptInput.Blur()
					cmds = append(cmds, callChatCmd(m.projectRoot, ckpt, prompt, m.chatTemp, m.chatMaxTokens))
				case "esc":
					m.chatTyping = false
					m.chatPromptInput.Blur()
				}
				return m, tea.Batch(cmds...)
			}
		}

		switch s {
		case "q", "ctrl+c":
			if m.running {
				m.stopTraining()
			}
			m.closeRunLogs()
			return m, tea.Quit
		case "tab", "l":
			if m.splashActive {
				break
			}
			m.tabIdx = (m.tabIdx + 1) % len(m.tabs)
		case "shift+tab", "h":
			if m.splashActive {
				break
			}
			m.tabIdx = (m.tabIdx - 1 + len(m.tabs)) % len(m.tabs)
		case "j", "down":
			if m.splashActive {
				break
			}
			if m.tabIdx == 0 {
				m.fieldIdx = min(len(m.fields)-1, m.fieldIdx+1)
			} else if m.tabIdx == 1 {
				metrics := m.currentMonitorMetrics()
				if len(metrics) > 0 {
					m.monitorIdx = min(len(metrics)-1, m.monitorIdx+1)
				}
				m.monitorView.LineDown(1)
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.selectChatModelByIndex(m.chatModelIdx + 1)
			}
		case "k", "up":
			if m.splashActive {
				break
			}
			if m.tabIdx == 0 {
				m.fieldIdx = max(0, m.fieldIdx-1)
			} else if m.tabIdx == 1 {
				m.monitorIdx = max(0, m.monitorIdx-1)
				m.monitorView.LineUp(1)
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.selectChatModelByIndex(m.chatModelIdx - 1)
			}
		case "left":
			if m.splashActive {
				break
			}
			if m.tabIdx == 1 {
				m.monitorMode = (m.monitorMode + 3) % 4
				m.monitorIdx = 0
			}
		case "right":
			if m.splashActive {
				break
			}
			if m.tabIdx == 1 {
				m.monitorMode = (m.monitorMode + 1) % 4
				m.monitorIdx = 0
			}
		case "pgup", "ctrl+u":
			if m.tabIdx == 0 {
				m.trainView.PageUp()
			} else if m.tabIdx == 1 {
				m.monitorView.PageUp()
			} else if m.tabIdx == 2 {
				m.logView.PageUp()
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.chatView.PageUp()
			}
		case "pgdown", "ctrl+d":
			if m.tabIdx == 0 {
				m.trainView.PageDown()
			} else if m.tabIdx == 1 {
				m.monitorView.PageDown()
			} else if m.tabIdx == 2 {
				m.logView.PageDown()
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.chatView.PageDown()
			}
		case "home":
			if m.tabIdx == 0 {
				m.trainView.GotoTop()
			} else if m.tabIdx == 1 {
				m.monitorView.GotoTop()
			} else if m.tabIdx == 2 {
				m.logView.GotoTop()
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.chatView.GotoTop()
			}
		case "end":
			if m.tabIdx == 0 {
				m.trainView.GotoBottom()
			} else if m.tabIdx == 1 {
				m.monitorView.GotoBottom()
			} else if m.tabIdx == 2 {
				m.logView.GotoBottom()
			} else if m.tabIdx == 5 && !m.chatTyping && !m.chatEditingPath {
				m.chatView.GotoBottom()
			}
		case "e":
			if m.splashActive {
				break
			}
			if m.tabIdx == 0 {
				m.startEdit()
			}
		case "f":
			if m.splashActive {
				break
			}
			if m.tabIdx == 0 && m.fields[m.fieldIdx].Key == "DATASET_PATH" {
				m.openDatasetPicker()
			}
		case "space":
			if m.splashActive {
				m.splashActive = false
				break
			}
			if m.tabIdx == 0 {
				m.cycleField(m.fieldIdx)
			}
		case "enter":
			if m.splashActive {
				m.splashActive = false
				break
			}
			if m.tabIdx == 0 {
				m.startEdit()
			} else if m.tabIdx == 1 {
				m.monitorFocus = !m.monitorFocus
			} else if m.tabIdx == 5 {
				if !m.chatEditingPath && !m.chatTyping {
					m.chatTyping = true
					m.chatPromptInput.Focus()
				}
			}
		case "1":
			if m.splashActive {
				break
			}
			m.applyPreset(0)
		case "2":
			if m.splashActive {
				break
			}
			m.applyPreset(1)
		case "3":
			if m.splashActive {
				break
			}
			m.applyPreset(2)
		case "s":
			if m.splashActive {
				break
			}
			m.startTraining()
		case "x":
			if m.splashActive {
				break
			}
			m.stopTraining()
		case "c":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 {
				m.chatRawLines = nil
				m.chatLines = nil
				m.chatView.SetContent("")
			} else {
				m.logs = nil
				m.logView.SetContent("")
			}
		case "r":
			if m.splashActive {
				break
			}
			m.refreshLists()
		case "p":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 {
				m.chatEditingPath = !m.chatEditingPath
				if m.chatEditingPath {
					m.chatTyping = false
					m.chatPromptInput.Blur()
					m.chatPathInput.Focus()
				} else {
					m.chatPathInput.Blur()
					m.chatPromptInput.Blur()
				}
			}
		case "esc":
			if m.splashActive {
				m.splashActive = false
				break
			}
			if m.tabIdx == 1 {
				m.monitorFocus = false
			}
			if m.tabIdx == 5 {
				if m.chatEditingPath {
					m.chatEditingPath = false
					m.chatPathInput.Blur()
				}
				if m.chatTyping {
					m.chatTyping = false
					m.chatPromptInput.Blur()
				}
			}
		case "L":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 {
				latest := filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
				m.chatPathInput.SetValue(latest)
				for i, name := range m.models {
					if name == "latest_checkpoint.json" {
						m.chatModelIdx = i
						break
					}
				}
				m.addChatLine("[system] using latest checkpoint: " + latest)
			}
		case "]":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 && !m.chatWaiting {
				m.chatTemp = math.Min(1.8, m.chatTemp+0.05)
			}
		case "[":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 && !m.chatWaiting {
				m.chatTemp = math.Max(0.1, m.chatTemp-0.05)
			}
		case "=":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 && !m.chatWaiting {
				m.chatMaxTokens = min(800, m.chatMaxTokens+10)
			}
		case "-":
			if m.splashActive {
				break
			}
			if m.tabIdx == 5 && !m.chatWaiting {
				m.chatMaxTokens = max(20, m.chatMaxTokens-10)
			}
		}

	case lineMsg:
		line := string(msg)
		if line == "" {
			break
		}
		m.appendLog(line)
		if strings.HasPrefix(line, "[debug]") {
			m.debugCount++
		}
		if strings.HasPrefix(line, "[model] latest checkpoint: ") {
			m.latestModel = strings.TrimSpace(strings.TrimPrefix(line, "[model] latest checkpoint: "))
		}
		if strings.HasPrefix(line, "[model] checkpoint saved: ") && m.latestModel == "" {
			m.latestModel = strings.TrimSpace(strings.TrimPrefix(line, "[model] checkpoint saved: "))
		}
		if mt := stepRE.FindStringSubmatch(line); len(mt) == 16 {
			st, _ := strconv.Atoi(mt[1])
			tot, _ := strconv.Atoi(mt[2])
			m.lastStep = stepMetrics{step: st, total: tot, loss: mt[3], lr: mt[4], seqLen: mt[5], docChars: mt[6], stepsPerSec: mt[7], elapsed: mt[8], eta: mt[9], heapMB: mt[10], runtimeMB: mt[11], sysRAMPct: mt[12], sysRAMAvail: mt[13], gc: mt[14], goroutines: mt[15]}
			if loss, err := strconv.ParseFloat(mt[3], 64); err == nil {
				m.lossSeries = appendSeries(m.lossSeries, loss, 5000)
			}
			if sps, err := strconv.ParseFloat(mt[7], 64); err == nil {
				m.spsSeries = appendSeries(m.spsSeries, sps, 5000)
				if seqLen, err := strconv.ParseFloat(mt[5], 64); err == nil {
					m.tokSeries = appendSeries(m.tokSeries, sps*seqLen, 5000)
				}
			}
			if lr, err := strconv.ParseFloat(mt[4], 64); err == nil {
				m.lrSeries = appendSeries(m.lrSeries, lr, 5000)
			}
			if heap, err := strconv.ParseFloat(mt[10], 64); err == nil {
				m.heapSeries = appendSeries(m.heapSeries, heap, 5000)
			}
			if rtm, err := strconv.ParseFloat(mt[11], 64); err == nil {
				m.rtmSeries = appendSeries(m.rtmSeries, rtm, 5000)
			}
			if gor, err := strconv.ParseFloat(mt[15], 64); err == nil {
				m.gorSeries = appendSeries(m.gorSeries, gor, 5000)
			}
			if gc, err := strconv.ParseFloat(mt[14], 64); err == nil {
				m.gcSeries = appendSeries(m.gcSeries, gc, 5000)
			}
		}
		if ev := evalRE.FindStringSubmatch(line); len(ev) == 8 {
			step, _ := strconv.Atoi(ev[1])
			trainLoss, _ := strconv.ParseFloat(ev[2], 64)
			valLoss, _ := strconv.ParseFloat(ev[3], 64)
			bestVal, _ := strconv.ParseFloat(ev[4], 64)
			improved := strings.EqualFold(ev[5], "true")
			pUsed, _ := strconv.Atoi(ev[6])
			pMax, _ := strconv.Atoi(ev[7])
			m.lastEval = evalMetrics{
				step:         step,
				trainLoss:    trainLoss,
				valLoss:      valLoss,
				bestVal:      bestVal,
				improved:     improved,
				patienceUsed: pUsed,
				patienceMax:  pMax,
			}
			m.valSeries = appendSeries(m.valSeries, valLoss, 5000)
			m.gapSeries = appendSeries(m.gapSeries, valLoss-trainLoss, 5000)
			if valLoss < 20 {
				m.pplSeries = appendSeries(m.pplSeries, math.Exp(valLoss), 5000)
			}
			m.appendEvalCSV(time.Now(), m.lastEval)
		}
		cmds = append(cmds, waitLineCmd(m.lineCh))

	case doneMsg:
		m.running = false
		m.pid = 0
		if msg.err != nil {
			m.status = "error"
			m.lastError = msg.err.Error()
			m.appendLog("[system] process ended with error: " + msg.err.Error())
		} else {
			m.status = "completed"
			m.appendLog("[system] training completed")
		}
		m.closeRunLogs()
		m.refreshLists()
		cmds = append(cmds, waitDoneCmd(m.doneCh))

	case sysTickMsg:
		m.sys = msg.stats
		m.prevCPUS = msg.next
		m.appendSystemCSV(msg.ts, msg.stats)
		m.cpuSeries = appendSeries(m.cpuSeries, m.sys.cpuPct, 5000)
		m.ramSeries = appendSeries(m.ramSeries, float64(m.sys.memUsedMB), 5000)
		m.rssSeries = appendSeries(m.rssSeries, float64(m.sys.procRSSKB)/1024.0, 5000)
		cmds = append(cmds, sysTickCmd(m.pid, m.prevCPUS))

	case refreshMsg:
		m.refreshLists()
		cmds = append(cmds, refreshCmd())

	case animTickMsg:
		m.animateMetrics()
		if m.splashActive {
			m.splashProgress, m.splashProgressVel = m.splashSpring.Update(m.splashProgress, m.splashProgressVel, 1.0)
			glowTarget := 0.5 + 0.5*math.Sin(float64(msg.ts.UnixNano())/1e9*4.0)
			m.splashGlow, m.splashGlowVel = m.splashSpring.Update(m.splashGlow, m.splashGlowVel, glowTarget)
			if time.Since(m.splashStarted) >= m.splashMinDuration && m.splashProgress >= 0.995 {
				m.splashActive = false
			}
		}
		cmds = append(cmds, animTickCmd())

	case chatResponseMsg:
		m.chatWaiting = false
		if msg.err != nil {
			m.addChatLine("[system] inference error: " + msg.err.Error())
		} else {
			if len(m.chatLines) > 0 && strings.HasSuffix(m.chatLines[len(m.chatLines)-1], "thinking...") {
				m.chatLines = m.chatLines[:len(m.chatLines)-1]
			}
			text := msg.text
			if text == "" {
				text = "(no output)"
			}
			m.addChatLine("[assistant] " + text)
		}
	}

	return m, tea.Batch(cmds...)
}

func (m model) renderTabs() string {
	parts := make([]string, len(m.tabs))
	for i, t := range m.tabs {
		if i == m.tabIdx {
			parts[i] = m.styles.tabActive.Render(t)
		} else {
			parts[i] = m.styles.tab.Render(t)
		}
	}
	return strings.Join(parts, " ")
}

func (m model) progressBar(w int) string {
	if w < 10 {
		w = 10
	}
	step, total := m.displayProgress()
	ratio := 0.0
	if total > 0 {
		ratio = float64(step) / float64(total)
	}
	done := int(math.Round(ratio * float64(w)))
	if done < 0 {
		done = 0
	}
	if done > w {
		done = w
	}
	return strings.Repeat("#", done) + strings.Repeat("-", w-done)
}

func (m model) configuredSteps() int {
	if f := m.fieldByKey("NUM_STEPS"); f != nil {
		if n, err := strconv.Atoi(strings.TrimSpace(f.Value)); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

func (m model) displayProgress() (int, int) {
	if m.lastStep.total > 0 {
		return m.lastStep.step, m.lastStep.total
	}
	return 0, m.configuredSteps()
}

func (m model) panel(title string, lines []string, w int) string {
	return m.styles.panel.Width(panelInnerWidth(w)).Render(m.styles.panelTitle.Render(title) + "\n" + strings.Join(lines, "\n"))
}

func panelInnerWidth(total int) int {
	// Account for rounded border (2 cols) + horizontal padding (2 cols).
	return max(8, total-4)
}

func (m model) viewTrainTab(w, h int) string {
	maxRows := max(8, min(22, h/2))
	start := max(0, m.fieldIdx-maxRows/2)
	if start+maxRows > len(m.fields) {
		start = max(0, len(m.fields)-maxRows)
	}
	end := min(len(m.fields), start+maxRows)

	lines := make([]string, 0, (end-start)+3)
	for i := start; i < end; i++ {
		f := m.fields[i]
		line := fmt.Sprintf("  %-16s = %s", f.Key, f.Value)
		if i == m.fieldIdx {
			line = m.styles.selected.Render("> " + fmt.Sprintf("%-16s = %s", f.Key, f.Value))
		}
		lines = append(lines, line)
	}
	lines = append(lines, "", m.styles.dim.Render(fmt.Sprintf("Showing %d-%d of %d fields", start+1, end, len(m.fields))))
	cfg := m.panel("Config (All Variables)", lines, w)

	desc := m.fields[m.fieldIdx]
	detailLines := []string{"Selected: " + desc.Label, desc.Desc, "", "Presets: 1=fast 2=balanced 3=max", "Edit: e or enter | cycle: space (bool/choice)", "Run: s start | x stop"}
	guidance := fieldGuidance(desc)
	if w >= 72 {
		detailLines = append(detailLines, "")
		detailLines = append(detailLines, splitDetailColumns(guidance, max(26, w-6))...)
	} else {
		detailLines = append(detailLines, guidance...)
	}
	if m.editing {
		detailLines = append(detailLines, "", "Editing: "+m.editor.View(), "Enter=apply Esc=cancel")
	}
	if m.datasetPicker {
		maxPathW := max(18, w-12)
		pickLines := []string{
			"Dataset Search: " + m.datasetPickerInput.View(),
			"Enter=select Esc=cancel Up/Down=move PgUp/PgDn=scroll",
			"",
		}
		if len(m.datasetFiltered) == 0 {
			pickLines = append(pickLines, "(no matching .jsonl files)")
		} else {
			start := max(0, m.datasetPickIdx-4)
			if start+8 > len(m.datasetFiltered) {
				start = max(0, len(m.datasetFiltered)-8)
			}
			end := min(len(m.datasetFiltered), start+8)
			for i := start; i < end; i++ {
				item := truncateWithEllipsis(m.datasetFiltered[i], maxPathW)
				line := "  " + item
				if i == m.datasetPickIdx {
					line = m.styles.selected.Render("> " + item)
				}
				pickLines = append(pickLines, line)
			}
		}
		detailLines = append(detailLines, "", m.styles.panelTitle.Render("Dataset Picker"))
		detailLines = append(detailLines, pickLines...)
	}
	if m.lastError != "" {
		detailLines = append(detailLines, "", m.styles.warn.Render("Last Error: "+m.lastError))
	}
	info := m.panel("Field Detail", detailLines, w)
	topH := max(8, int(float64(h)*0.52))
	botH := max(8, h-topH-1)
	return lipgloss.JoinVertical(lipgloss.Top, fitHeight(cfg, topH), fitHeight(info, botH))
}

func splitDetailColumns(lines []string, totalW int) []string {
	clean := make([]string, 0, len(lines))
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			clean = append(clean, ln)
		}
	}
	if len(clean) == 0 || totalW < 48 {
		return lines
	}
	gap := 3
	colW := max(20, (totalW-gap)/2)
	left := make([]string, 0, (len(clean)+1)/2)
	right := make([]string, 0, len(clean)/2)
	for i, ln := range clean {
		wrapped := wrapText(ln, colW)
		if i%2 == 0 {
			left = append(left, wrapped...)
		} else {
			right = append(right, wrapped...)
		}
	}
	rows := max(len(left), len(right))
	out := make([]string, 0, rows)
	for i := 0; i < rows; i++ {
		l := ""
		r := ""
		if i < len(left) {
			l = left[i]
		}
		if i < len(right) {
			r = right[i]
		}
		out = append(out, fmt.Sprintf("%-*s%s%s", colW, l, strings.Repeat(" ", gap), r))
	}
	return out
}

func (m model) trainRightColumn(w, h int) string {
	step, total := m.displayProgress()
	evalLine := "Eval: pending"
	if m.lastEval.step > 0 {
		evalLine = fmt.Sprintf("Eval: step %d | val %.4f | best %.4f", m.lastEval.step, m.lastEval.valLoss, m.lastEval.bestVal)
	}
	summary := m.panel("Runtime Summary", []string{
		fmt.Sprintf("Status: %s", m.status),
		fmt.Sprintf("Step: %d/%d", step, total),
		fmt.Sprintf("Loss: %s | Steps/s: %s", nz(m.lastStep.loss, "-"), nz(m.lastStep.stepsPerSec, "-")),
		evalLine,
		"Loss graph: " + m.styles.graphLoss.Render(sparkline(m.lossSeries, max(10, w-16))),
		fmt.Sprintf("CPU %.1f%% | RAM %d/%dMB", m.sys.cpuPct, m.sys.memUsedMB, m.sys.memTotalMB),
		m.gpuInfo,
		"Latest model: " + pathOrDash(m.latestModel),
	}, w)

	lastLogs := []string{"Latest log lines:"}
	if len(m.logs) == 0 {
		lastLogs = append(lastLogs, m.styles.dim.Render("(no logs yet)"))
	} else {
		start := max(0, len(m.logs)-8)
		for _, ln := range m.logs[start:] {
			lastLogs = append(lastLogs, truncateWithEllipsis(ln, max(18, w-8)))
		}
	}
	activity := m.panel("Recent Activity", lastLogs, w)

	topH := max(9, int(float64(h)*0.44))
	botH := max(8, h-topH-1)
	return lipgloss.JoinVertical(lipgloss.Top, fitHeight(summary, topH), fitHeight(activity, botH))
}

func fieldGuidance(f cfgField) []string {
	switch f.Key {
	case "DATASET_PATH":
		return []string{
			"",
			"What this is: the training data file your model learns from.",
			"Why it matters: wrong file = bad outputs or schema errors.",
			"Safe start: keep `datasets/raw/databricks-dolly-15k.jsonl`.",
			"Tip: press `f` on this field to open dataset search/picker.",
		}
	case "TOKENIZER":
		return []string{
			"",
			"What this is: how text is split into tokens before training.",
			"`bpe` usually gives better language quality than `char`.",
			"Safe start: use `bpe`.",
		}
	case "BPE_ENCODING":
		return []string{
			"",
			"What this is: the BPE dictionary family.",
			"Use this only when TOKENIZER is `bpe`.",
			"Safe start: keep `cl100k_base`.",
		}
	case "TOKEN_VOCAB_SIZE":
		return []string{
			"",
			"What this is: number of subword tokens kept in local vocab.",
			"Why it matters: higher can improve wording, but increases RAM/time.",
			"Safe start (laptop): 2048. Next step: 3072, then 4096.",
		}
	case "N_LAYER":
		return []string{
			"",
			"What this is: number of transformer blocks (model depth).",
			"Why it matters: more layers can improve quality, but slows training heavily.",
			"Safe start: 1 on laptop, 2 if stable.",
		}
	case "N_EMBD":
		return []string{
			"",
			"What this is: embedding width (model capacity per token).",
			"Why it matters: bigger is stronger but costs RAM/CPU.",
			"Rule: must be divisible by N_HEAD.",
			"Safe start: 48.",
		}
	case "N_HEAD":
		return []string{
			"",
			"What this is: number of attention heads.",
			"Why it matters: controls how attention is split.",
			"Rule: must divide N_EMBD exactly.",
			"Safe start: 4.",
		}
	case "BLOCK_SIZE":
		return []string{
			"",
			"What this is: max context length (how much text model sees at once).",
			"Why it matters: higher gives more context but can explode compute cost.",
			"Safe start: 96. Try 128 only if run time is acceptable.",
		}
	case "NUM_STEPS":
		return []string{
			"",
			"What this is: total optimizer updates.",
			"Why it matters: more steps = longer training and possibly better fit.",
			"Safe start: 800. Increase slowly after a successful run.",
		}
	case "LEARNING_RATE":
		return []string{
			"",
			"What this is: step size for weight updates.",
			"Why it matters: too high can destabilize, too low can be very slow.",
			"Safe start: 0.004.",
		}
	case "BETA1", "BETA2":
		return []string{
			"",
			"What this is: Adam optimizer momentum settings.",
			"Why it matters: affects smoothing/stability of updates.",
			"Safe start: keep defaults unless you are debugging optimizer behavior.",
		}
	case "EPS_ADAM":
		return []string{
			"",
			"What this is: tiny Adam stability constant.",
			"Why it matters: prevents divide-by-zero style issues.",
			"Safe start: keep default `1e-8`.",
		}
	case "TEMPERATURE":
		return []string{
			"",
			"What this is: randomness for generated text.",
			"Why it matters: lower = more predictable, higher = more creative/noisy.",
			"Safe start: 0.60.",
		}
	case "SAMPLE_COUNT":
		return []string{
			"",
			"What this is: number of preview samples generated after training.",
			"Why it matters: more samples gives better inspection but takes longer.",
			"Safe start: 8.",
		}
	case "SAMPLE_MAX_NEW_TOKENS":
		return []string{
			"",
			"What this is: max length of each generated preview sample.",
			"Why it matters: longer outputs cost more time and can drift.",
			"Safe start: 160.",
		}
	case "TOP_K":
		return []string{
			"",
			"What this is: keep only top-K next-token choices.",
			"Why it matters: reduces wild/random outputs.",
			"Safe start: 40. Set 0 to disable.",
		}
	case "TOP_P":
		return []string{
			"",
			"What this is: nucleus sampling threshold.",
			"Why it matters: lower values reduce randomness and increase consistency.",
			"Safe start: 0.90.",
		}
	case "REPETITION_PENALTY":
		return []string{
			"",
			"What this is: penalty for repeating recent tokens.",
			"Why it matters: helps reduce loops and repeated phrases.",
			"Safe start: 1.10.",
		}
	case "MIN_NEW_TOKENS":
		return []string{
			"",
			"What this is: minimum tokens before model is allowed to stop.",
			"Why it matters: prevents ultra-short empty replies.",
			"Safe start: 24.",
		}
	case "REPEAT_LAST_N":
		return []string{
			"",
			"What this is: number of recent tokens tracked for repetition penalty.",
			"Why it matters: larger window catches more repeats but may over-penalize.",
			"Safe start: 64.",
		}
	case "VAL_SPLIT":
		return []string{
			"",
			"What this is: portion of dataset reserved for validation only.",
			"Why it matters: lets you monitor generalization instead of memorization.",
			"Safe start: 0.10.",
		}
	case "EVAL_INTERVAL":
		return []string{
			"",
			"What this is: how often validation runs during training.",
			"Why it matters: frequent eval gives faster feedback but adds overhead.",
			"Safe start: 50.",
		}
	case "EVAL_STEPS":
		return []string{
			"",
			"What this is: how many validation docs are sampled per eval run.",
			"Why it matters: larger sample is more accurate but slower.",
			"Safe start: 16.",
		}
	case "EARLY_STOP_PATIENCE":
		return []string{
			"",
			"What this is: number of eval checks allowed without improvement.",
			"Why it matters: automatically stops wasting time when model plateaus.",
			"Safe start: 8.",
		}
	case "EARLY_STOP_MIN_DELTA":
		return []string{
			"",
			"What this is: minimum validation-loss drop to count as real progress.",
			"Why it matters: ignores tiny noise-only changes.",
			"Safe start: 0.0005.",
		}
	case "TRAIN_DEVICE":
		return []string{
			"",
			"What this is: requested compute device.",
			"Current reality: this trainer runs CPU kernels today.",
			"Safe start: `cpu`.",
		}
	case "METRIC_INTERVAL":
		return []string{
			"",
			"What this is: how often per-step metric lines are printed.",
			"Why it matters: lower value gives more live feedback, but more log volume.",
			"Safe start: 1 for monitoring, 10+ for quieter logs.",
		}
	case "LOG_LEVEL":
		return []string{
			"",
			"What this is: log verbosity mode.",
			"`debug` includes deeper per-step diagnostics; `info` is cleaner.",
			"Safe start: `debug` while tuning, then `info` for longer runs.",
		}
	case "VERBOSE":
		return []string{
			"",
			"What this is: enables full runtime metric printing.",
			"Why it matters: required for rich TUI monitoring.",
			"Safe start: `true`.",
		}
	case "MODEL_OUT_PATH":
		return []string{
			"",
			"What this is: optional custom checkpoint file path.",
			"Why it matters: empty value enables automatic per-run naming.",
			"Safe start: leave empty for auto naming in `models/`.",
		}
	default:
		return nil
	}
}

func (m model) viewMonitorTab(w, h int) string {
	status := m.styles.ok.Render(strings.ToUpper(m.status))
	if m.running {
		status = m.styles.warn.Render(m.spin.View() + " RUNNING")
	}
	step, total := m.displayProgress()
	evalState := "no eval yet"
	if m.lastEval.step > 0 {
		evalState = fmt.Sprintf(
			"Eval step %d | train %.4f | val %.4f | best %.4f | gap %.4f | patience %d/%d | improved=%t",
			m.lastEval.step,
			m.lastEval.trainLoss,
			m.lastEval.valLoss,
			m.lastEval.bestVal,
			m.lastEval.valLoss-m.lastEval.trainLoss,
			m.lastEval.patienceUsed,
			m.lastEval.patienceMax,
			m.lastEval.improved,
		)
	}
	info := m.panel("Training Monitor", []string{
		"Status: " + status,
		fmt.Sprintf("Progress: %d/%d", step, total),
		m.progressBar(max(14, w-8)),
		fmt.Sprintf("Loss=%s | LR=%s | Steps/s=%s | ETA=%s", nz(m.lastStep.loss, "-"), nz(m.lastStep.lr, "-"), nz(m.lastStep.stepsPerSec, "-"), nz(m.lastStep.eta, "-")),
		evalState,
		fmt.Sprintf("CPU %.1f%% | RAM %d/%dMB free %dMB", m.sys.cpuPct, m.sys.memUsedMB, m.sys.memTotalMB, m.sys.memFreeMB),
		fmt.Sprintf("PID %d | RSS %dKB | Debug lines %d", m.sys.pid, m.sys.procRSSKB, m.debugCount),
		"Latest model: " + pathOrDash(m.latestModel),
		"Logs: train/ system/ eval under logs/",
	}, w)
	metrics := m.currentMonitorMetrics()
	idx := m.monitorIdx
	if idx >= len(metrics) && len(metrics) > 0 {
		idx = len(metrics) - 1
	}
	if idx < 0 {
		idx = 0
	}
	selected := monitorMetric{}
	if len(metrics) > 0 {
		selected = metrics[idx]
	}
	availW := max(48, w-2)
	rightW := min(40, max(26, availW/3))
	leftW := max(30, availW-rightW-2)
	if leftW < 34 {
		leftW = 34
		rightW = max(24, availW-leftW-2)
	}

	var graphWrap string
	if m.monitorFocus && selected.Name != "" {
		chartW := max(18, leftW-12)
		chartH := max(7, h-lipgloss.Height(info)-16)
		lines := []string{
			"Selected: " + selected.Name,
			"",
		}
		for _, ln := range lineChart(selected.Series, chartW, chartH) {
			lines = append(lines, selected.Color.Render(ln))
		}
		if latest, minV, maxV, ok := seriesStats(selected.Series); ok {
			delta := 0.0
			if len(selected.Series) >= 2 {
				delta = selected.Series[len(selected.Series)-1] - selected.Series[len(selected.Series)-2]
			}
			lines = append(lines, "")
			lines = append(lines, fmt.Sprintf("latest %.4f | delta %+0.4f | min %.4f | max %.4f | n=%d", latest, delta, minV, maxV, len(selected.Series)))
		}
		lines = append(lines, "")
		lines = append(lines, wrapText("What: "+selected.What, max(14, leftW-6))...)
		lines = append(lines, wrapText("Why: "+selected.Why, max(14, leftW-6))...)
		lines = append(lines, wrapText("How to read: "+selected.Read, max(14, leftW-6))...)
		lines = append(lines, "")
		lines = append(lines, m.styles.dim.Render("Press Enter or Esc to exit focus mode"))
		graphWrap = m.panel("Live Graphs (Metric Focus)", lines, leftW)
	} else {
		var graphRows []string
		if len(metrics) == 0 {
			graphRows = append(graphRows, "No metrics available yet.")
		} else {
			labelW := 24
			statsW := max(16, min(30, leftW/3))
			sparkW := max(8, leftW-labelW-statsW-8)
			for i, g := range metrics {
				name := truncateWithEllipsis(g.Name, labelW)
				prefix := "  "
				if i == idx {
					prefix = m.styles.selected.Render("> ")
				}
				sparkRaw := sparkline(g.Series, sparkW)
				spark := g.Color.Render(sparkRaw)
				stats := "n=0"
				if latest, _, _, ok := seriesStats(g.Series); ok {
					delta := 0.0
					if len(g.Series) >= 2 {
						delta = g.Series[len(g.Series)-1] - g.Series[len(g.Series)-2]
					}
					stats = fmt.Sprintf("l %.3f d %+0.3f n=%d", latest, delta, len(g.Series))
				}
				graphRows = append(graphRows, fmt.Sprintf("%s%-24s %s %s", prefix, name, spark, truncateWithEllipsis(stats, statsW)))
				// Add breathing room so adjacent spark rows don't visually overlap.
				graphRows = append(graphRows, "")
			}
		}
		graphRows = append(graphRows, "")
		graphRows = append(graphRows, m.styles.dim.Render(fmt.Sprintf("Showing %d metrics. Enter: focus selected metric. PgUp/PgDn/Home/End: scroll.", len(metrics))))
		graphWrap = m.panel("Live Graphs (Realtime + Historical)", graphRows, leftW)
	}
	modeNames := []string{"All", "Core", "Eval", "System"}
	modeName := "All"
	if m.monitorMode >= 0 && m.monitorMode < len(modeNames) {
		modeName = modeNames[m.monitorMode]
	}
	explorer := make([]string, 0, 28)
	explorer = append(explorer, "Category: "+modeName)
	explorer = append(explorer, wrapText("Use left/right to switch categories", max(12, rightW-6))...)
	explorer = append(explorer, wrapText("Use up/down to choose a metric", max(12, rightW-6))...)
	explorer = append(explorer, "")
	explorer = append(explorer, "Metrics:")
	for i, g := range metrics {
		line := "  " + truncateWithEllipsis(g.Name, max(10, rightW-8))
		if i == idx {
			line = m.styles.selected.Render("> " + truncateWithEllipsis(g.Name, max(10, rightW-9)))
		}
		explorer = append(explorer, line)
	}
	explorer = append(explorer, "")
	if selected.Name != "" {
		explorer = append(explorer, "Selected: "+selected.Name)
		explorer = append(explorer, wrapText("What: "+selected.What, max(14, rightW-6))...)
		explorer = append(explorer, wrapText("Why: "+selected.Why, max(14, rightW-6))...)
		explorer = append(explorer, wrapText("How to read: "+selected.Read, max(14, rightW-6))...)
	}
	explorePanel := m.panel("Metric Explorer", explorer, rightW)
	var mid string
	if w < 130 {
		mid = lipgloss.JoinVertical(lipgloss.Top, graphWrap, explorePanel)
	} else {
		mid = lipgloss.JoinHorizontal(lipgloss.Top, graphWrap, "  ", explorePanel)
	}
	midH := max(6, h-lipgloss.Height(info)-1)
	mv := m.monitorView
	mv.Width = max(30, availW)
	mv.Height = max(6, midH)
	mv.SetContent(mid)
	return lipgloss.JoinVertical(lipgloss.Top, info, mv.View())
}

func (m model) viewRunsTab(w int) string {
	lines := []string{"Recent run artifacts:"}
	if len(m.runs) == 0 {
		lines = append(lines, m.styles.dim.Render("(none yet)"))
	} else {
		for _, r := range m.runs {
			lines = append(lines, r)
		}
	}
	lines = append(lines, "", "Press r to refresh")
	return m.panel("Runs", lines, w)
}

func (m model) viewLogsTab(w, h int) string {
	lv := m.logView
	innerW := max(20, panelInnerWidth(w)-2)
	lv.Width = innerW
	lv.Height = max(4, h-5)

	// Wrap log lines to viewport width so terminal-side hard wrapping doesn't
	// inflate visual height and push the tab/header off-screen.
	wrapped := make([]string, 0, len(m.logs)*2)
	for _, ln := range m.logs {
		wrapped = append(wrapped, wrapText(ln, innerW)...)
	}
	lv.SetContent(strings.Join(wrapped, "\n"))
	lv.GotoBottom()

	body := m.styles.panel.Width(panelInnerWidth(w)).Render(m.styles.panelTitle.Render("Live Logs") + "\n" + lv.View())
	return fitHeight(body, h)
}

func (m model) viewModelsTab(w int) string {
	lines := []string{"Available model checkpoints:"}
	if len(m.models) == 0 {
		lines = append(lines, m.styles.dim.Render("(none yet)"))
	} else {
		for _, r := range m.models {
			lines = append(lines, "- "+r)
		}
	}
	lines = append(lines, "", "Latest from runtime: "+pathOrDash(m.latestModel))
	return m.panel("Models", lines, w)
}

func (m model) viewChatTab(w, h int) string {
	status := "ready"
	if m.chatWaiting {
		status = m.spin.View() + " generating..."
	}
	checkpoint := strings.TrimSpace(m.chatPathInput.Value())
	if checkpoint == "" {
		checkpoint = filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
	}
	modeBadge := m.styles.dim.Render("HOTKEYS")
	if m.chatTyping {
		modeBadge = m.styles.ok.Render("TYPING")
	}
	if m.chatEditingPath {
		modeBadge = m.styles.warn.Render("PATH EDIT")
	}
	modelLines := []string{"Selectable checkpoints (j/k):"}
	if len(m.models) == 0 {
		modelLines = append(modelLines, m.styles.dim.Render("(none found in models/)"))
	} else {
		start := max(0, m.chatModelIdx-4)
		if start+8 > len(m.models) {
			start = max(0, len(m.models)-8)
		}
		end := min(len(m.models), start+8)
		for i := start; i < end; i++ {
			name := m.models[i]
			if i == m.chatModelIdx {
				modelLines = append(modelLines, m.styles.selected.Render("> "+name))
			} else {
				modelLines = append(modelLines, "  "+name)
			}
		}
	}
	leftW := max(42, int(float64(w)*0.66))
	rightW := max(24, w-leftW-2)
	if rightW > 44 {
		rightW = 44
		leftW = max(42, w-rightW-2)
	}
	composerH := 4
	topH := max(8, h-composerH-1)

	cv := m.chatView
	cv.Width = max(34, leftW-6)
	cv.Height = max(6, topH-2)

	chat := m.styles.panel.Width(panelInnerWidth(leftW)).Render(m.styles.panelTitle.Render("Conversation") + "\n" + cv.View())
	controls := m.panel("Chat Lab", []string{
		"Status: " + status,
		"Mode: " + modeBadge,
		"Checkpoint:",
		checkpoint,
		fmt.Sprintf("Temperature: %.2f", m.chatTemp),
		fmt.Sprintf("Max tokens: %d", m.chatMaxTokens),
		"",
		"Enter: type/send",
		"Esc: exit typing",
		"p: edit checkpoint",
		"L: latest checkpoint",
		"[ ]: temp  - =: tokens",
		"c: clear chat",
	}, rightW)
	modelPanel := m.panel("Model Selector", modelLines, rightW)
	side := lipgloss.JoinVertical(lipgloss.Top, controls, modelPanel)
	side = fitHeight(side, topH)
	top := lipgloss.JoinHorizontal(lipgloss.Top, chat, "  ", side)

	inputLabel := "Prompt"
	input := m.chatPromptInput.View()
	if m.chatEditingPath {
		inputLabel = "Checkpoint Path (Enter to apply)"
		input = m.chatPathInput.View()
	} else if !m.chatTyping {
		inputLabel = "Prompt"
		input = m.styles.dim.Render("Press Enter to start typing")
	}
	composer := m.styles.panel.Width(panelInnerWidth(w)).Render(m.styles.panelTitle.Render(inputLabel) + "\n" + input)
	return lipgloss.JoinVertical(lipgloss.Top, top, composer)
}

func (m model) View() string {
	if m.width == 0 {
		return "loading..."
	}
	if m.splashActive {
		return m.viewSplash()
	}
	header := m.renderTabs()

	contentW := max(70, m.width-4)
	headerH := lipgloss.Height(header)
	footer := m.viewFooter(contentW)
	footerH := lipgloss.Height(footer)
	contentH := max(8, m.height-headerH-footerH-2)
	var content string
	switch m.tabIdx {
	case 0:
		var trainLayout string
		if contentW < 120 {
			trainLayout = lipgloss.JoinVertical(
				lipgloss.Top,
				m.viewTrainTab(contentW, max(10, int(float64(contentH)*0.58))),
				m.trainRightColumn(contentW, max(8, int(float64(contentH)*0.40))),
			)
		} else {
			leftW := max(38, int(float64(contentW)*0.60))
			rightW := max(30, contentW-leftW-2)
			left := m.viewTrainTab(leftW, contentH)
			right := m.trainRightColumn(rightW, contentH)
			trainLayout = lipgloss.JoinHorizontal(lipgloss.Top, left, "  ", right)
		}
		tv := m.trainView
		tv.Width = max(30, contentW)
		tv.Height = max(6, contentH)
		tv.SetContent(trainLayout)
		content = tv.View()
	case 1:
		content = m.viewMonitorTab(contentW, contentH)
	case 2:
		content = m.viewLogsTab(contentW, contentH)
	case 3:
		content = m.viewRunsTab(contentW)
	case 4:
		content = m.viewModelsTab(contentW)
	default:
		content = m.viewChatTab(contentW, contentH)
	}
	if m.tabIdx == 3 || m.tabIdx == 4 {
		content = fitHeight(content, contentH)
	}
	return lipgloss.JoinVertical(lipgloss.Left, header, "", content, footer)
}

func (m model) viewFooter(w int) string {
	base := []string{"[tab/h/l] switch tabs", "[s] start", "[x] stop", "[r] refresh", "[q] quit"}
	context := []string{}
	switch m.tabIdx {
	case 0:
		context = []string{"[j/k] select field", "[e/enter] edit", "[space] cycle bool/choice", "[f] dataset picker", "[1/2/3] presets", "[pgup/pgdown/home/end] scroll train"}
	case 1:
		context = []string{"[left/right] metric category", "[up/down] metric focus", "[enter] full graph focus", "[esc] exit focus", "[pgup/pgdown/home/end] scroll monitor"}
	case 2:
		context = []string{"[pgup/pgdown/home/end] scroll logs", "[c] clear logs"}
	case 3:
		context = []string{"Run artifacts grouped by train/system/eval + manifest"}
	case 4:
		context = []string{"Model list auto-refreshes from models/"}
	default:
		if m.chatTyping || m.chatEditingPath {
			context = []string{"Typing mode active: enter send/apply", "[esc] exit typing mode", "Hotkeys are locked while typing"}
		} else {
			context = []string{"[enter] start typing", "[j/k] select checkpoint", "[pgup/pgdown/home/end] scroll", "[p] checkpoint path", "[[ / ]] temp", "[-/=] tokens"}
		}
	}
	lines := []string{
		"Global: " + strings.Join(base, "  "),
		"Context: " + strings.Join(context, "  "),
	}
	body := []string{}
	for _, ln := range lines {
		body = append(body, wrapText(ln, max(20, w-8))...)
	}
	style := m.styles.panel.Copy().Padding(0, 1)
	return style.Width(panelInnerWidth(w)).Render(strings.Join(body, "\n"))
}

func pathOrDash(p string) string {
	if strings.TrimSpace(p) == "" {
		return "-"
	}
	return p
}

func nz(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func sampleSystem(pid int, prev cpuSample) (sysStats, cpuSample) {
	targetPID := resolveTrainerPID(pid)
	stats := sysStats{pid: targetPID}
	if total, idle, ok := readCPUStat(); ok {
		if prev.total > 0 && total > prev.total {
			dt := float64(total - prev.total)
			di := float64(idle - prev.idle)
			stats.cpuPct = (1.0 - di/dt) * 100.0
		}
		prev = cpuSample{total: total, idle: idle}
	}
	if total, used, free, ok := readMemMB(); ok {
		stats.memTotalMB = total
		stats.memUsedMB = used
		stats.memFreeMB = free
	}
	if targetPID > 0 {
		stats.procRSSKB = readProcRSSKB(targetPID)
	}
	return stats, prev
}

func resolveTrainerPID(pid int) int {
	if pid <= 0 {
		return 0
	}
	childrenPath := fmt.Sprintf("/proc/%d/task/%d/children", pid, pid)
	b, err := os.ReadFile(childrenPath)
	if err != nil {
		return pid
	}
	fields := strings.Fields(string(b))
	if len(fields) == 0 {
		return pid
	}
	childPID, err := strconv.Atoi(fields[len(fields)-1])
	if err != nil || childPID <= 0 {
		return pid
	}
	return childPID
}

func readCPUStat() (total, idle uint64, ok bool) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "cpu ") {
			f := strings.Fields(ln)
			if len(f) < 5 {
				return 0, 0, false
			}
			vals := make([]uint64, 0, len(f)-1)
			for _, s := range f[1:] {
				v, err := strconv.ParseUint(s, 10, 64)
				if err != nil {
					return 0, 0, false
				}
				vals = append(vals, v)
				total += v
			}
			if len(vals) >= 4 {
				idle = vals[3]
			}
			return total, idle, true
		}
	}
	return 0, 0, false
}

func readMemMB() (total, used, free int64, ok bool) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0, 0, false
	}
	var totalKB, availKB int64
	for _, ln := range strings.Split(string(b), "\n") {
		f := strings.Fields(ln)
		if len(f) < 2 {
			continue
		}
		switch f[0] {
		case "MemTotal:":
			v, _ := strconv.ParseInt(f[1], 10, 64)
			totalKB = v
		case "MemAvailable:":
			v, _ := strconv.ParseInt(f[1], 10, 64)
			availKB = v
		}
	}
	if totalKB == 0 {
		return 0, 0, 0, false
	}
	usedKB := totalKB - availKB
	return totalKB / 1024, usedKB / 1024, availKB / 1024, true
}

func readProcRSSKB(pid int) int64 {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "VmRSS:") {
			f := strings.Fields(ln)
			if len(f) >= 2 {
				v, _ := strconv.ParseInt(f[1], 10, 64)
				return v
			}
		}
	}
	return 0
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func appendSeries(series []float64, v float64, capN int) []float64 {
	series = append(series, v)
	if len(series) > capN {
		series = series[len(series)-capN:]
	}
	return series
}

func seriesStats(series []float64) (latest, minV, maxV float64, ok bool) {
	if len(series) == 0 {
		return 0, 0, 0, false
	}
	latest = series[len(series)-1]
	minV, maxV = series[0], series[0]
	for _, v := range series[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	return latest, minV, maxV, true
}

func (m model) currentMonitorMetrics() []monitorMetric {
	all := []monitorMetric{
		{Name: "Loss (Train)", Series: m.lossSeries, Color: m.styles.graphLoss, What: "Cross-entropy loss on training steps.", Why: "Primary optimization target.", Read: "Should trend down over time."},
		{Name: "Loss (Validation)", Series: m.valSeries, Color: m.styles.graphEval, What: "Loss on validation docs.", Why: "Tracks generalization quality.", Read: "Best metric for checkpoint quality."},
		{Name: "Generalization Gap", Series: m.gapSeries, Color: m.styles.warn, What: "Validation loss minus training loss.", Why: "Quick overfitting detector.", Read: "Keep small and stable."},
		{Name: "Throughput (Steps/sec)", Series: m.spsSeries, Color: m.styles.graphSPS, What: "Optimizer steps completed per second.", Why: "Measures training speed and stability.", Read: "Drops often map to CPU or memory pressure."},
		{Name: "Throughput (Tokens/sec)", Series: m.tokSeries, Color: m.styles.graphSPS, What: "Approximate tokens processed each second.", Why: "Best throughput metric when seq length changes.", Read: "Use for fair run-to-run speed comparison."},
		{Name: "Validation Perplexity", Series: m.pplSeries, Color: m.styles.graphEval, What: "Exp(val loss), an intuitive confidence metric.", Why: "Useful for comparing runs; lower usually means cleaner generations.", Read: "Lower is better. Very high values indicate unstable predictions."},
		{Name: "Learning Rate", Series: m.lrSeries, Color: m.styles.graphSPS, What: "Current optimizer step size.", Why: "Too high can diverge; too low can stall progress.", Read: "Use with loss trends: spikes in loss with high LR suggest instability."},
		{Name: "CPU (%)", Series: m.cpuSeries, Color: m.styles.graphCPU, What: "System-wide CPU utilization.", Why: "Confirms training is actively using compute.", Read: "Near 100% is expected for CPU training."},
		{Name: "RAM Used (MB)", Series: m.ramSeries, Color: m.styles.graphMem, What: "Total system RAM currently in use.", Why: "Prevents OOM crashes and swap thrashing.", Read: "Steady climb near max is a warning sign."},
		{Name: "Process RSS (MB)", Series: m.rssSeries, Color: m.styles.graphMem, What: "Trainer process resident memory.", Why: "Shows memory pressure from model/data configuration.", Read: "If it keeps rising, reduce model size or block size."},
	}
	switch m.monitorMode {
	case 1:
		return []monitorMetric{
			all[0], all[1], all[2], all[3], all[4],
		}
	case 2:
		return []monitorMetric{
			all[1], all[2], all[5], all[6],
		}
	case 3:
		return []monitorMetric{
			all[7], all[8], all[9],
		}
	default:
		return all
	}
}

func (m model) graphPanel(title string, series []float64, w int, st lipgloss.Style) string {
	height := 5
	if w < 48 {
		height = 4
	}
	width := max(16, w-16)
	graphLines := lineChart(series, width, height)
	for i := range graphLines {
		graphLines[i] = st.Render(graphLines[i])
	}
	graph := strings.Join(graphLines, "\n")
	if latest, minV, maxV, ok := seriesStats(series); ok {
		delta := 0.0
		if len(series) >= 2 {
			delta = series[len(series)-1] - series[len(series)-2]
		}
		sub := fmt.Sprintf("latest %.4f | delta %+0.4f | min %.4f | max %.4f | n=%d", latest, delta, minV, maxV, len(series))
		return m.panel(title, []string{graph, m.styles.dim.Render(sub)}, w)
	}
	return m.panel(title, []string{graph, m.styles.dim.Render("waiting for data...")}, w)
}

func lineChart(series []float64, width, height int) []string {
	if width < 8 {
		width = 8
	}
	if height < 3 {
		height = 3
	}
	if len(series) == 0 {
		return []string{strings.Repeat(".", width)}
	}
	sampled := make([]float64, 0, width)
	if len(series) <= width {
		sampled = append(sampled, series...)
	} else {
		step := float64(len(series)-1) / float64(width-1)
		for i := 0; i < width; i++ {
			idx := int(math.Round(float64(i) * step))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(series) {
				idx = len(series) - 1
			}
			sampled = append(sampled, series[idx])
		}
	}
	minV, maxV := sampled[0], sampled[0]
	for _, v := range sampled[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	grid := make([][]rune, height)
	for r := 0; r < height; r++ {
		row := make([]rune, width)
		for c := range row {
			row[c] = ' '
		}
		grid[r] = row
	}
	center := height / 2
	lastRow := center
	for x := 0; x < len(sampled); x++ {
		row := center
		if maxV > minV {
			ratio := (sampled[x] - minV) / (maxV - minV)
			row = height - 1 - int(math.Round(ratio*float64(height-1)))
		}
		if row < 0 {
			row = 0
		}
		if row >= height {
			row = height - 1
		}
		grid[row][x] = '●'
		if x > 0 {
			if row > lastRow {
				for rr := lastRow + 1; rr < row; rr++ {
					if grid[rr][x-1] == ' ' {
						grid[rr][x-1] = '│'
					}
				}
			} else if row < lastRow {
				for rr := row + 1; rr < lastRow; rr++ {
					if grid[rr][x-1] == ' ' {
						grid[rr][x-1] = '│'
					}
				}
			}
		}
		lastRow = row
	}
	lines := make([]string, 0, height)
	for r := 0; r < height; r++ {
		label := "         │"
		if r == 0 {
			label = fmt.Sprintf("%8.3f ┤", maxV)
		} else if r == height-1 {
			label = fmt.Sprintf("%8.3f ┤", minV)
		}
		lines = append(lines, label+string(grid[r]))
	}
	return lines
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func fitHeight(s string, h int) string {
	if h <= 0 {
		return s
	}
	lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for len(lines) < h {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func wrapText(s string, width int) []string {
	if width <= 1 {
		return []string{s}
	}
	paras := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	if len(paras) == 0 {
		return []string{""}
	}
	out := make([]string, 0, len(paras)*2)
	for _, p := range paras {
		if strings.TrimSpace(p) == "" {
			out = append(out, "")
			continue
		}
		words := strings.FieldsFunc(p, unicode.IsSpace)
		if len(words) == 0 {
			out = append(out, "")
			continue
		}
		cur := words[0]
		for _, w := range words[1:] {
			if len([]rune(w)) > width {
				if strings.TrimSpace(cur) != "" {
					out = append(out, cur)
				}
				rs := []rune(w)
				for len(rs) > width {
					out = append(out, string(rs[:width]))
					rs = rs[width:]
				}
				cur = string(rs)
				continue
			}
			if len([]rune(cur))+1+len([]rune(w)) <= width {
				cur += " " + w
			} else {
				out = append(out, cur)
				cur = w
			}
		}
		out = append(out, cur)
	}
	return out
}

func truncateWithEllipsis(s string, maxRunes int) string {
	if maxRunes < 4 {
		maxRunes = 4
	}
	rs := []rune(s)
	if len(rs) <= maxRunes {
		return s
	}
	return string(rs[:maxRunes-1]) + "…"
}

func (m *model) animateMetrics() {
	animate := func(src []float64, cur *float64, vel *float64, dst *[]float64) {
		if len(src) == 0 {
			return
		}
		target := src[len(src)-1]
		if !m.animPrimed {
			*cur = target
			*vel = 0
		}
		*cur, *vel = m.graphSpring.Update(*cur, *vel, target)
		*dst = appendSeries(*dst, *cur, 5000)
	}
	if len(m.lossSeries) > 0 {
		animate(m.lossSeries, &m.lossAnim, &m.lossVel, &m.lossAnimSeries)
	}
	animate(m.valSeries, &m.valAnim, &m.valVel, &m.valAnimSeries)
	animate(m.gapSeries, &m.gapAnim, &m.gapVel, &m.gapAnimSeries)
	animate(m.spsSeries, &m.spsAnim, &m.spsVel, &m.spsAnimSeries)
	animate(m.tokSeries, &m.tokAnim, &m.tokVel, &m.tokAnimSeries)
	animate(m.cpuSeries, &m.cpuAnim, &m.cpuVel, &m.cpuAnimSeries)
	animate(m.ramSeries, &m.ramAnim, &m.ramVel, &m.ramAnimSeries)
	animate(m.rssSeries, &m.rssAnim, &m.rssVel, &m.rssAnimSeries)
	m.animPrimed = true
}

func (m model) viewSplash() string {
	title := "MircoGPT - Go Edition"
	reveal := int(math.Round(float64(len(title)) * clamp01(m.splashProgress)))
	if reveal < 0 {
		reveal = 0
	}
	if reveal > len(title) {
		reveal = len(title)
	}
	head := m.styles.splashText.Render(title[:reveal]) + m.styles.dim.Render(title[reveal:])

	barW := max(24, min(56, m.width-20))
	done := int(math.Round(float64(barW) * clamp01(m.splashProgress)))
	if done < 0 {
		done = 0
	}
	if done > barW {
		done = barW
	}
	bar := "[" + strings.Repeat("=", done) + strings.Repeat(" ", barW-done) + "]"

	intensity := clamp01(0.3 + 0.7*m.splashGlow)
	pulseColor := lipgloss.Color("81")
	if intensity > 0.65 {
		pulseColor = lipgloss.Color("123")
	}

	t := time.Since(m.splashStarted).Seconds()
	orbitWidth := max(18, min(34, m.width-30))
	orbit := make([]rune, orbitWidth)
	for i := range orbit {
		orbit[i] = '·'
	}
	p1 := int((math.Sin(t*2.1)*0.5 + 0.5) * float64(orbitWidth-1))
	p2 := int((math.Cos(t*1.6+1.7)*0.5 + 0.5) * float64(orbitWidth-1))
	if p1 >= 0 && p1 < orbitWidth {
		orbit[p1] = '●'
	}
	if p2 >= 0 && p2 < orbitWidth {
		orbit[p2] = '◆'
	}
	orbitLine := lipgloss.NewStyle().Foreground(pulseColor).Render(string(orbit))

	waveW := max(24, min(56, m.width-20))
	var wb strings.Builder
	for i := 0; i < waveW; i++ {
		y := math.Sin((float64(i)*0.42)+(t*3.2)) * intensity
		switch {
		case y > 0.60:
			wb.WriteRune('█')
		case y > 0.25:
			wb.WriteRune('▓')
		case y > -0.1:
			wb.WriteRune('▒')
		default:
			wb.WriteRune('░')
		}
	}
	wave := lipgloss.NewStyle().Foreground(pulseColor).Render(wb.String())

	body := lipgloss.JoinVertical(
		lipgloss.Center,
		orbitLine,
		head,
		"",
		wave,
		bar,
		m.styles.dim.Render("Press Enter to skip"),
	)
	card := m.styles.splash.Render(body)
	return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, card)
}

func sparkline(series []float64, width int) string {
	if width < 4 {
		width = 4
	}
	if len(series) == 0 {
		return strings.Repeat(".", width)
	}
	sampled := make([]float64, 0, width)
	if len(series) <= width {
		sampled = append(sampled, series...)
	} else {
		step := float64(len(series)-1) / float64(width-1)
		for i := 0; i < width; i++ {
			idx := int(math.Round(float64(i) * step))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(series) {
				idx = len(series) - 1
			}
			sampled = append(sampled, series[idx])
		}
	}
	minV, maxV := sampled[0], sampled[0]
	for _, v := range sampled[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	chars := []rune("▁▂▃▄▅▆▇█")
	if maxV == minV {
		return strings.Repeat(string(chars[len(chars)-2]), width)
	}
	var b strings.Builder
	b.Grow(width)
	for i := 0; i < len(sampled); i++ {
		v := sampled[i]
		r := (v - minV) / (maxV - minV)
		pos := int(math.Round(r * float64(len(chars)-1)))
		if pos < 0 {
			pos = 0
		}
		if pos >= len(chars) {
			pos = len(chars) - 1
		}
		b.WriteRune(chars[pos])
	}
	for i := len(sampled); i < width; i++ {
		b.WriteRune(chars[0])
	}
	return b.String()
}

func main() {
	m := initialModel()
	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}
}
