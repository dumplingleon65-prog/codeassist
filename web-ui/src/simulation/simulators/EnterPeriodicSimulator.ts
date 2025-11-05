import * as monaco from "monaco-editor";
import type { HumanSimulator, SimulationAction, SimulationConfig, SimulationStats } from "../types";

export class EnterPeriodicSimulator implements HumanSimulator {
  private editor: monaco.editor.IStandaloneCodeEditor | null = null;
  private config: SimulationConfig;
  private stats: SimulationStats = {
    totalActions: 0,
    totalDurationMs: 0,
    episodesCreated: 0,
    agentSuggestionsReceived: 0,
  };
  private isRunning = false;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private startTime: number = 0;

  constructor(config: SimulationConfig) {
    this.config = config;
  }

  setEditor(editor: monaco.editor.IStandaloneCodeEditor | null) {
    this.editor = editor;
  }

  async start(): Promise<void> {
    if (this.isRunning || !this.editor) {
      return;
    }

    this.isRunning = true;
    this.startTime = Date.now();
    this.stats.episodesCreated += 1;
    
    console.log("Starting Enter Periodic Simulator...");

    // First, simulate clicking on the editor to focus it
    await this.simulateEditorClick();

    // Start the periodic Enter key simulation
    this.startPeriodicEnterSimulation();
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.stats.totalDurationMs += Date.now() - this.startTime;
    this.stats.lastRunTime = Date.now();
    
    console.log("Stopped Enter Periodic Simulator");
  }

  private startPeriodicEnterSimulation() {
    const intervalMs = this.config.intervalMs || 3000; // Default 3 seconds
    const maxActions = this.config.maxActions || 50; // Default max 50 actions
    const durationMs = this.config.durationMs || 300000; // Default 5 minutes

    let actionCount = 0;
    
    this.intervalId = setInterval(async () => {
      if (!this.isRunning || !this.editor) {
        this.stop();
        return;
      }

      // Check if we've reached max actions or duration
      if (actionCount >= maxActions || (Date.now() - this.startTime) >= durationMs) {
        this.stop();
        return;
      }

      // Execute Enter key press
      const action: SimulationAction = {
        type: "enter",
        timestamp: Date.now(),
      };

      await this.executeAction(action);
      actionCount++;
    }, intervalMs);
  }

  async executeAction(action: SimulationAction): Promise<void> {
    if (!this.editor) {
      return;
    }

    switch (action.type) {
      case "enter":
        await this.simulateEnterKey();
        break;
      case "type":
        if (action.payload?.text) {
          await this.simulateTyping(action.payload.text);
        }
        break;
      case "cursor_move":
        if (action.payload?.position) {
          await this.simulateCursorMove(action.payload.position);
        }
        break;
      case "wait":
        if (action.payload?.durationMs) {
          await this.simulateWait(action.payload.durationMs);
        }
        break;
    }

    this.stats.totalActions += 1;
  }

  private async simulateEnterKey(): Promise<void> {
    if (!this.editor) return;
    this.editor.trigger('keyboard', 'type', { text: '\n' });
  }

  private async simulateTyping(text: string): Promise<void> {
    if (!this.editor) return;
    
    const position = this.editor.getPosition();
    if (!position) return;

    const edit = {
      range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
      text: text,
    };

    this.editor.executeEdits("simulation-type", [edit]);
    
    // Move cursor after the inserted text
    const newPosition = new monaco.Position(position.lineNumber, position.column + text.length);
    this.editor.setPosition(newPosition);

    console.log(`Simulated typing: "${text}"`);
  }

  private async simulateCursorMove(position: { line: number; column: number }): Promise<void> {
    if (!this.editor) return;
    
    const newPosition = new monaco.Position(position.line, position.column);
    this.editor.setPosition(newPosition);
    
    console.log(`Moved cursor to line ${position.line}, column ${position.column}`);
  }

  private async simulateWait(durationMs: number): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, durationMs));
    console.log(`Waited for ${durationMs}ms`);
  }

  getStats(): SimulationStats {
    return { ...this.stats };
  }

  private async simulateEditorClick(): Promise<void> {
    if (!this.editor) return;
    
    // Focus the editor (simulates clicking on it)
    this.editor.focus();
    
    // Set cursor to after the last line of the editor
    const model = this.editor.getModel();
    if (model) {
      const lastLineNumber = model.getLineCount();
      const lastLineLength = model.getLineContent(lastLineNumber).length;
      const position = new monaco.Position(lastLineNumber, lastLineLength + 1);
      this.editor.setPosition(position);
    }
    
    console.log("Simulated click on editor - focused and positioned cursor after last line");
  }
}


