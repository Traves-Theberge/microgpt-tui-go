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

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
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
}

type model struct {
	width   int
	height  int
	styles  styles
	tabs    []string
	tabIdx  int
	presets []preset

	fields      []cfgField
	fieldIdx    int
	editing     bool
	editor      textinput.Model
	status      string
	running     bool
	latestModel string
	debugCount  int

	cmd          *exec.Cmd
	pid          int
	projectRoot  string
	trainLogPath string
	latestLog    string
	metricsPath  string
	trainLogFile *os.File
	latestFile   *os.File
	metricsFile  *os.File
	lineCh       chan string
	doneCh       chan error

	logs      []string
	logView   viewport.Model
	lastStep  stepMetrics
	sys       sysStats
	prevCPUS  cpuSample
	spin      spinner.Model
	help      help.Model
	keys      keyMap
	runs      []string
	models    []string
	lastError string

	chatView        viewport.Model
	chatLines       []string
	chatPromptInput textinput.Model
	chatPathInput   textinput.Model
	chatEditingPath bool
	chatTemp        float64
	chatMaxTokens   int
	chatWaiting     bool

	lossSeries []float64
	spsSeries  []float64
	cpuSeries  []float64
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
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Start, k.Stop, k.TabNext, k.Edit, k.Quit}
}

func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Start, k.Stop, k.ClearLog, k.Refresh, k.Quit},
		{k.TabNext, k.TabPrev, k.Up, k.Down},
		{k.Edit, k.Apply, k.Cancel, k.Cycle},
		{k.Preset1, k.Preset2, k.Preset3},
		{k.Path, k.Latest, k.TempDown, k.TempUp, k.TokDown, k.TokUp},
	}
}

var stepRE = regexp.MustCompile(`\[step\]\s+(\d+)/(\d+)\s+loss=([^\s]+)\s+lr=([^\s]+)\s+seq_len=(\d+)\s+doc_chars=(\d+)\s+steps_per_sec=([^\s]+)\s+elapsed=([^\s]+)\s+eta=([^\s]+)\s+heap_alloc_mb=([^\s]+)\s+runtime_sys_mb=([^\s]+)\s+sys_ram_used_pct=([^\s]+)\s+sys_ram_avail_mb=([^\s]+)\s+gc=(\d+)\s+goroutines=(\d+)`)

type sysTickMsg struct {
	stats sysStats
	next  cpuSample
	ts    time.Time
}

type lineMsg string
type doneMsg struct{ err error }
type refreshMsg struct{}
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
	}
}

func defaultFields(root string) []cfgField {
	return []cfgField{
		{Key: "DATASET_PATH", Label: "Dataset Path", Type: fieldString, Value: "assistant_dataset_train.jsonl", Desc: "JSONL training dataset path"},
		{Key: "N_LAYER", Label: "Layers", Type: fieldInt, Value: "2", Desc: "Transformer layer count"},
		{Key: "N_EMBD", Label: "Embedding Size", Type: fieldInt, Value: "48", Desc: "Embedding width"},
		{Key: "N_HEAD", Label: "Attention Heads", Type: fieldInt, Value: "4", Desc: "Head count"},
		{Key: "BLOCK_SIZE", Label: "Block Size", Type: fieldInt, Value: "64", Desc: "Max sequence length"},
		{Key: "NUM_STEPS", Label: "Training Steps", Type: fieldInt, Value: "1200", Desc: "Optimizer steps"},
		{Key: "LEARNING_RATE", Label: "Learning Rate", Type: fieldFloat, Value: "0.008", Desc: "Initial learning rate"},
		{Key: "BETA1", Label: "Adam Beta1", Type: fieldFloat, Value: "0.85", Desc: "Adam momentum term"},
		{Key: "BETA2", Label: "Adam Beta2", Type: fieldFloat, Value: "0.99", Desc: "Adam variance term"},
		{Key: "EPS_ADAM", Label: "Adam Epsilon", Type: fieldFloat, Value: "1e-8", Desc: "Adam stability epsilon"},
		{Key: "TEMPERATURE", Label: "Sample Temperature", Type: fieldFloat, Value: "0.60", Desc: "Generation randomness"},
		{Key: "SAMPLE_COUNT", Label: "Sample Count", Type: fieldInt, Value: "12", Desc: "Number of output samples"},
		{Key: "TRAIN_DEVICE", Label: "Train Device", Type: fieldChoice, Value: "cpu", Desc: "Requested compute device", Choices: []string{"cpu", "gpu"}},
		{Key: "METRIC_INTERVAL", Label: "Metric Interval", Type: fieldInt, Value: "1", Desc: "Verbose metric log cadence"},
		{Key: "LOG_LEVEL", Label: "Log Level", Type: fieldChoice, Value: "debug", Desc: "info or debug", Choices: []string{"info", "debug"}},
		{Key: "VERBOSE", Label: "Verbose Metrics", Type: fieldBool, Value: "true", Desc: "Always on for dashboard"},
		{Key: "MODEL_OUT_PATH", Label: "Model Output", Type: fieldString, Value: filepath.Join(root, "models", "checkpoint_manual.json"), Desc: "Checkpoint output path"},
	}
}

func defaultPresets() []preset {
	return []preset{
		{name: "fast", description: "quick smoke run", values: map[string]string{"N_LAYER": "1", "N_EMBD": "32", "N_HEAD": "4", "BLOCK_SIZE": "48", "NUM_STEPS": "800", "LEARNING_RATE": "0.008", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "8"}},
		{name: "balanced", description: "recommended baseline", values: map[string]string{"N_LAYER": "2", "N_EMBD": "48", "N_HEAD": "4", "BLOCK_SIZE": "64", "NUM_STEPS": "1200", "LEARNING_RATE": "0.008", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "12"}},
		{name: "max", description: "heavier run", values: map[string]string{"N_LAYER": "3", "N_EMBD": "64", "N_HEAD": "4", "BLOCK_SIZE": "96", "NUM_STEPS": "1800", "LEARNING_RATE": "0.006", "TEMPERATURE": "0.6", "SAMPLE_COUNT": "12"}},
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
	prompt.Focus()

	pathIn := textinput.New()
	pathIn.CharLimit = 600
	pathIn.SetValue(filepath.Join(root, "models", "latest_checkpoint.json"))

	logVP := viewport.New(100, 16)
	logVP.SetContent("logs will appear here")
	chatVP := viewport.New(100, 16)
	chatVP.SetContent("chat will appear here")

	m := model{
		styles:          defaultStyles(),
		tabs:            []string{"Train", "Monitor", "Runs", "Models", "Chat"},
		tabIdx:          0,
		presets:         defaultPresets(),
		fields:          defaultFields(root),
		fieldIdx:        0,
		editor:          ed,
		status:          "idle",
		lineCh:          make(chan string, 4096),
		doneCh:          make(chan error, 1),
		spin:            sp,
		logView:         logVP,
		chatView:        chatVP,
		chatPromptInput: prompt,
		chatPathInput:   pathIn,
		chatTemp:        0.60,
		chatMaxTokens:   220,
		help:            help.New(),
		projectRoot:     root,
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
		},
	}
	m.addChatLine("[system] chat ready")
	m.addChatLine("[system] checkpoint: " + m.chatPathInput.Value())
	m.refreshLists()
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
	return tea.Batch(m.spin.Tick, waitLineCmd(m.lineCh), waitDoneCmd(m.doneCh), sysTickCmd(m.pid, m.prevCPUS), refreshCmd())
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
	m.chatLines = append(m.chatLines, line)
	if len(m.chatLines) > 2500 {
		m.chatLines = m.chatLines[len(m.chatLines)-2500:]
	}
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
	intMin := map[string]int{"N_LAYER": 1, "N_EMBD": 1, "N_HEAD": 1, "BLOCK_SIZE": 2, "NUM_STEPS": 1, "SAMPLE_COUNT": 1, "METRIC_INTERVAL": 1}
	floatMinExclusive := map[string]float64{"LEARNING_RATE": 0, "EPS_ADAM": 0, "TEMPERATURE": 0}
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
	dev := strings.ToLower(vals["TRAIN_DEVICE"])
	if dev != "cpu" && dev != "gpu" {
		return fmt.Errorf("TRAIN_DEVICE must be cpu or gpu")
	}
	if vals["DATASET_PATH"] == "" {
		return fmt.Errorf("DATASET_PATH cannot be empty")
	}
	if vals["MODEL_OUT_PATH"] == "" {
		return fmt.Errorf("MODEL_OUT_PATH cannot be empty")
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

func (m *model) openRunLogs(presetName string) error {
	if err := os.MkdirAll(filepath.Join(m.projectRoot, "logs"), 0o755); err != nil {
		return err
	}
	ts := time.Now().Format("20060102_150405")
	m.trainLogPath = filepath.Join(m.projectRoot, "logs", fmt.Sprintf("tui_train_%s_%s.log", presetName, ts))
	m.metricsPath = filepath.Join(m.projectRoot, "logs", fmt.Sprintf("tui_system_metrics_%s_%s.csv", presetName, ts))
	m.latestLog = filepath.Join(m.projectRoot, "logs", "train_latest.log")

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
	m.trainLogFile = lf
	m.latestFile = lat
	m.metricsFile = mf
	_, _ = m.metricsFile.WriteString("timestamp,pid,cpu_percent,proc_rss_kb,sys_mem_used_mb,sys_mem_free_mb,sys_mem_total_mb,step,step_total\n")
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
	m.latestModel = ""
	m.debugCount = 0
	m.appendLog(fmt.Sprintf("[system] pid=%d", m.pid))
	m.appendLog("[system] train log: " + m.trainLogPath)
	m.appendLog("[system] system metrics: " + m.metricsPath)

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
	m.runs = listFilesByPattern(filepath.Join(m.projectRoot, "logs"), "tui_train_*.log", 12)
	m.models = listFilesByPattern(filepath.Join(m.projectRoot, "models"), "*.json", 16)
	if m.latestModel == "" {
		latest := filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
		if _, err := os.Stat(latest); err == nil {
			m.latestModel = latest
		}
	}
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

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	m.spin, cmd = m.spin.Update(msg)
	cmds = append(cmds, cmd)

	if m.editing {
		m.editor, cmd = m.editor.Update(msg)
		cmds = append(cmds, cmd)
	}

	if m.tabIdx == 4 {
		if m.chatEditingPath {
			m.chatPathInput, cmd = m.chatPathInput.Update(msg)
			cmds = append(cmds, cmd)
		} else {
			m.chatPromptInput, cmd = m.chatPromptInput.Update(msg)
			cmds = append(cmds, cmd)
		}
	}

	m.logView, cmd = m.logView.Update(msg)
	cmds = append(cmds, cmd)
	m.chatView, cmd = m.chatView.Update(msg)
	cmds = append(cmds, cmd)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.logView.Width = max(60, m.width-8)
		m.logView.Height = max(9, m.height-20)
		m.chatView.Width = max(60, m.width-8)
		m.chatView.Height = max(9, m.height-20)
		m.editor.Width = max(24, min(64, m.width/2))
		m.chatPromptInput.Width = max(28, m.width-12)
		m.chatPathInput.Width = max(28, m.width-12)

	case tea.KeyMsg:
		s := msg.String()
		if m.editing {
			switch s {
			case "enter":
				m.applyEdit()
			case "esc":
				m.cancelEdit()
			}
			break
		}

		switch s {
		case "q", "ctrl+c":
			if m.running {
				m.stopTraining()
			}
			m.closeRunLogs()
			return m, tea.Quit
		case "tab", "l":
			m.tabIdx = (m.tabIdx + 1) % len(m.tabs)
		case "shift+tab", "h":
			m.tabIdx = (m.tabIdx - 1 + len(m.tabs)) % len(m.tabs)
		case "j", "down":
			if m.tabIdx == 0 {
				m.fieldIdx = min(len(m.fields)-1, m.fieldIdx+1)
			}
		case "k", "up":
			if m.tabIdx == 0 {
				m.fieldIdx = max(0, m.fieldIdx-1)
			}
		case "e":
			if m.tabIdx == 0 {
				m.startEdit()
			}
		case "space":
			if m.tabIdx == 0 {
				m.cycleField(m.fieldIdx)
			}
		case "enter":
			if m.tabIdx == 0 {
				m.startEdit()
			} else if m.tabIdx == 4 {
				if m.chatEditingPath {
					m.chatEditingPath = false
					m.chatPathInput.Blur()
					m.chatPromptInput.Focus()
					m.addChatLine("[system] checkpoint set: " + strings.TrimSpace(m.chatPathInput.Value()))
				} else {
					if m.chatWaiting {
						break
					}
					prompt := strings.TrimSpace(m.chatPromptInput.Value())
					if prompt == "" {
						break
					}
					ckpt := strings.TrimSpace(m.chatPathInput.Value())
					if ckpt == "" {
						ckpt = filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
						m.chatPathInput.SetValue(ckpt)
					}
					if _, err := os.Stat(ckpt); err != nil {
						m.addChatLine("[system] checkpoint not found: " + ckpt)
						break
					}
					m.chatWaiting = true
					m.addChatLine("[you] " + prompt)
					m.addChatLine("[assistant] thinking...")
					m.chatPromptInput.SetValue("")
					cmds = append(cmds, callChatCmd(m.projectRoot, ckpt, prompt, m.chatTemp, m.chatMaxTokens))
				}
			}
		case "1":
			m.applyPreset(0)
		case "2":
			m.applyPreset(1)
		case "3":
			m.applyPreset(2)
		case "s":
			m.startTraining()
		case "x":
			m.stopTraining()
		case "c":
			if m.tabIdx == 4 {
				m.chatLines = nil
				m.chatView.SetContent("")
			} else {
				m.logs = nil
				m.logView.SetContent("")
			}
		case "r":
			m.refreshLists()
		case "p":
			if m.tabIdx == 4 {
				m.chatEditingPath = !m.chatEditingPath
				if m.chatEditingPath {
					m.chatPathInput.Focus()
					m.chatPromptInput.Blur()
				} else {
					m.chatPathInput.Blur()
					m.chatPromptInput.Focus()
				}
			}
		case "L":
			if m.tabIdx == 4 {
				latest := filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
				m.chatPathInput.SetValue(latest)
				m.addChatLine("[system] using latest checkpoint: " + latest)
			}
		case "]":
			if m.tabIdx == 4 && !m.chatWaiting {
				m.chatTemp = math.Min(1.8, m.chatTemp+0.05)
			}
		case "[":
			if m.tabIdx == 4 && !m.chatWaiting {
				m.chatTemp = math.Max(0.1, m.chatTemp-0.05)
			}
		case "=":
			if m.tabIdx == 4 && !m.chatWaiting {
				m.chatMaxTokens = min(800, m.chatMaxTokens+10)
			}
		case "-":
			if m.tabIdx == 4 && !m.chatWaiting {
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
				m.lossSeries = appendSeries(m.lossSeries, loss, 80)
			}
			if sps, err := strconv.ParseFloat(mt[7], 64); err == nil {
				m.spsSeries = appendSeries(m.spsSeries, sps, 80)
			}
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
		m.cpuSeries = appendSeries(m.cpuSeries, m.sys.cpuPct, 80)
		cmds = append(cmds, sysTickCmd(m.pid, m.prevCPUS))

	case refreshMsg:
		m.refreshLists()
		cmds = append(cmds, refreshCmd())

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
	ratio := 0.0
	if m.lastStep.total > 0 {
		ratio = float64(m.lastStep.step) / float64(m.lastStep.total)
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

func (m model) panel(title string, lines []string, w int) string {
	return m.styles.panel.Width(w).Render(m.styles.panelTitle.Render(title) + "\n" + strings.Join(lines, "\n"))
}

func (m model) viewTrainTab(w int) string {
	maxRows := 12
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
	detailLines = append(detailLines, fieldGuidance(desc)...)
	if m.editing {
		detailLines = append(detailLines, "", "Editing: "+m.editor.View(), "Enter=apply Esc=cancel")
	}
	if m.lastError != "" {
		detailLines = append(detailLines, "", m.styles.warn.Render("Last Error: "+m.lastError))
	}
	info := m.panel("Field Detail", detailLines, w)
	return lipgloss.JoinVertical(lipgloss.Top, cfg, info)
}

func fieldGuidance(f cfgField) []string {
	switch f.Key {
	case "DATASET_PATH":
		return []string{"", "Expected: path to JSONL file", "Tip: use datasets/examples/*.jsonl to start quickly"}
	case "N_LAYER":
		return []string{"", "Range: integer >= 1", "Higher = better quality, slower training"}
	case "N_EMBD":
		return []string{"", "Range: integer >= 1", "Must be divisible by N_HEAD"}
	case "N_HEAD":
		return []string{"", "Range: integer >= 1", "Must divide N_EMBD evenly"}
	case "BLOCK_SIZE":
		return []string{"", "Range: integer >= 2", "Higher = longer context, more compute/memory"}
	case "NUM_STEPS":
		return []string{"", "Range: integer >= 1", "More steps improve fit but can overfit"}
	case "LEARNING_RATE":
		return []string{"", "Range: float > 0", "Lower if training is unstable"}
	case "BETA1", "BETA2":
		return []string{"", "Range: 0.0 to 1.0", "Adam momentum coefficients"}
	case "EPS_ADAM":
		return []string{"", "Range: float > 0", "Small numeric stability constant"}
	case "TEMPERATURE":
		return []string{"", "Range: float > 0", "Lower = more deterministic samples"}
	case "SAMPLE_COUNT":
		return []string{"", "Range: integer >= 1", "Number of sample outputs after training"}
	case "TRAIN_DEVICE":
		return []string{"", "Allowed: cpu, gpu", "Current engine falls back to CPU when gpu is requested"}
	case "METRIC_INTERVAL":
		return []string{"", "Range: integer >= 1", "How often [step] lines are emitted"}
	case "LOG_LEVEL":
		return []string{"", "Allowed: info, debug", "debug adds richer per-step diagnostics"}
	case "VERBOSE":
		return []string{"", "Allowed: true/false", "true recommended for TUI observability"}
	case "MODEL_OUT_PATH":
		return []string{"", "Expected: output path ending in .json", "Tip: models/checkpoint_<date>.json"}
	default:
		return nil
	}
}

func (m model) viewMonitorTab(w int) string {
	status := m.styles.ok.Render(strings.ToUpper(m.status))
	if m.running {
		status = m.styles.warn.Render(m.spin.View() + " RUNNING")
	}
	info := m.panel("Training Monitor", []string{
		"Status: " + status,
		fmt.Sprintf("Progress: %d/%d", m.lastStep.step, m.lastStep.total),
		m.progressBar(max(14, w-8)),
		fmt.Sprintf("Loss=%s | LR=%s | Steps/s=%s | ETA=%s", nz(m.lastStep.loss, "-"), nz(m.lastStep.lr, "-"), nz(m.lastStep.stepsPerSec, "-"), nz(m.lastStep.eta, "-")),
		fmt.Sprintf("CPU %.1f%% | RAM %d/%dMB free %dMB", m.sys.cpuPct, m.sys.memUsedMB, m.sys.memTotalMB, m.sys.memFreeMB),
		fmt.Sprintf("PID %d | RSS %dKB | Debug lines %d", m.sys.pid, m.sys.procRSSKB, m.debugCount),
		"Latest model: " + pathOrDash(m.latestModel),
	}, w)
	graphs := m.panel("Live Graphs", []string{
		"Loss      " + sparkline(m.lossSeries, max(14, w-18)),
		"Steps/sec " + sparkline(m.spsSeries, max(14, w-18)),
		"CPU %     " + sparkline(m.cpuSeries, max(14, w-18)),
	}, w)
	logPanel := m.styles.panel.Width(w).Render(m.styles.panelTitle.Render("Live Logs") + "\n" + m.logView.View())
	return lipgloss.JoinVertical(lipgloss.Top, info, graphs, logPanel)
}

func (m model) viewRunsTab(w int) string {
	lines := []string{"Recent run logs:"}
	if len(m.runs) == 0 {
		lines = append(lines, m.styles.dim.Render("(none yet)"))
	} else {
		for _, r := range m.runs {
			lines = append(lines, "- "+r)
		}
	}
	lines = append(lines, "", "Press r to refresh")
	return m.panel("Runs", lines, w)
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

func (m model) viewChatTab(w int) string {
	status := "ready"
	if m.chatWaiting {
		status = m.spin.View() + " generating..."
	}
	checkpoint := strings.TrimSpace(m.chatPathInput.Value())
	if checkpoint == "" {
		checkpoint = filepath.Join(m.projectRoot, "models", "latest_checkpoint.json")
	}
	info := m.panel("Chat Lab", []string{
		"Status: " + status,
		"Checkpoint: " + checkpoint,
		fmt.Sprintf("Temperature: %.2f | Max tokens: %d", m.chatTemp, m.chatMaxTokens),
		"Keys: enter=send, p=toggle path edit, L=latest checkpoint",
		"Keys: [ ] temp, - = tokens, c clear chat",
	}, w)
	chat := m.styles.panel.Width(w).Render(m.styles.panelTitle.Render("Conversation") + "\n" + m.chatView.View())
	inputLabel := "Prompt"
	input := m.chatPromptInput.View()
	if m.chatEditingPath {
		inputLabel = "Checkpoint Path (Enter to apply)"
		input = m.chatPathInput.View()
	}
	return lipgloss.JoinVertical(lipgloss.Top, info, chat, inputLabel+": "+input)
}

func (m model) View() string {
	if m.width == 0 {
		return "loading..."
	}
	title := m.styles.title.Render("MircoGPT-tui")
	header := title + "\n" + m.renderTabs()

	contentW := max(70, m.width-4)
	var content string
	switch m.tabIdx {
	case 0:
		leftW := max(34, int(float64(contentW)*0.58))
		rightW := max(26, contentW-leftW-2)
		mini := m.panel("Runtime Summary", []string{
			fmt.Sprintf("Status: %s", m.status),
			fmt.Sprintf("Step: %d/%d", m.lastStep.step, m.lastStep.total),
			fmt.Sprintf("Loss: %s | Steps/s: %s", nz(m.lastStep.loss, "-"), nz(m.lastStep.stepsPerSec, "-")),
			"Loss graph: " + sparkline(m.lossSeries, max(10, rightW-14)),
			fmt.Sprintf("CPU %.1f%% | RAM %d/%dMB", m.sys.cpuPct, m.sys.memUsedMB, m.sys.memTotalMB),
			"Latest model: " + pathOrDash(m.latestModel),
		}, rightW)
		content = lipgloss.JoinHorizontal(lipgloss.Top, m.viewTrainTab(leftW), "  ", mini)
	case 1:
		content = m.viewMonitorTab(contentW)
	case 2:
		content = m.viewRunsTab(contentW)
	case 3:
		content = m.viewModelsTab(contentW)
	default:
		content = m.viewChatTab(contentW)
	}
	quick := m.styles.dim.Render("Quick: tab/h/l switch tabs | s start | x stop | e edit | enter apply/send | p chat path | r refresh")
	return lipgloss.JoinVertical(lipgloss.Left, header, "", content, quick, m.help.View(m.keys))
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
	chars := []byte(" .:-=+*#%@")
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
		b.WriteByte(chars[pos])
	}
	for b.Len() < width {
		b.WriteByte(chars[0])
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
