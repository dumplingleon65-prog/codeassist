import * as monaco from "monaco-editor";
import type { HumanSimulator, SimulationAction, SimulationConfig, SimulationStats } from "../types";

/**
 * OllamaCodeSimulator - A human simulator that generates code using Ollama FIM completion
 * 
 * This simulator uses Fill-in-Middle (FIM) prompting with the same base coder model
 * used in the policy_models to generate realistic code completions. It's designed to
 * simulate a human writing code with the cursor at a specific position.
 * 
 * Prerequisites:
 * 1. Ollama must be running locally (http://localhost:11434)
 * 2. The model must be pulled: `ollama pull qwen2.5-coder:0.5b-base`
 * 
 * The simulator uses FIM tokens to provide context before and after the cursor,
 * allowing the base model to generate appropriate code completions.
 */


export class OllamaCodeSimulator implements HumanSimulator {
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
  private ollamaUrl: string;
  private modelName: string;
  private isTyping = false; // Mutex to prevent concurrent typing operations
  private problemDescription?: string; // Problem description to include in prompts

  constructor(config: SimulationConfig) {
    this.config = config;
    this.problemDescription = config.problemDescription;
    // Use localhost for Ollama API - adjust port if needed
    this.ollamaUrl = "http://localhost:11434";
    // Use the same base coder model as policy_models for FIM completion
    // Note: Ensure this model is pulled with: ollama pull qwen2.5-coder:0.5b-base
    this.modelName = "qwen2.5-coder:0.5b-base";
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
    
    console.log("Starting Ollama Code Simulator...");

    // First, simulate clicking on the editor to focus it
    await this.simulateEditorClick();

    // Start the periodic code generation simulation
    this.startPeriodicCodeGeneration();
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    this.isTyping = false; // Reset typing state when stopping
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.stats.totalDurationMs += Date.now() - this.startTime;
    this.stats.lastRunTime = Date.now();
    
    console.log("Stopped Ollama Code Simulator");
  }

  private startPeriodicCodeGeneration() {
    const intervalMs = this.config.intervalMs || 3000;
    const maxActions = this.config.maxActions || 20;
    const duration = this.config.durationMs || 300000; // Default 5 minutes

    let actionCount = 0;
    
    this.intervalId = setInterval(async () => {
      if (!this.isRunning || !this.editor) {
        this.stop();
        return;
      }

      // Check if we've reached max actions or duration
      if (actionCount >= maxActions || (Date.now() - this.startTime) >= duration) {
        this.stop();
        return;
      }

      // Generate next line of code using Ollama
      const action: SimulationAction = {
        type: "ollama_generate",
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
      case "ollama_generate":
        await this.generateAndTypeCode();
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

  private async generateAndTypeCode(): Promise<void> {
    if (!this.editor) return;

    // Check if already typing - skip if so
    if (this.isTyping) {
      console.log("🔒 Skipping generation - already typing");
      return;
    }

    try {
      // Set typing lock
      this.isTyping = true;
      console.log("🔒 Acquired typing lock");

      // Get current context from the editor
      const model = this.editor.getModel();
      if (!model) return;

      const position = this.editor.getPosition();
      if (!position) return;

      // Get text before cursor for context
      const beforeCursor = model.getValueInRange({
        startLineNumber: 1,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: position.column,
      });

      // Get text after cursor for context
      const afterCursor = model.getValueInRange({
        startLineNumber: position.lineNumber,
        startColumn: position.column,
        endLineNumber: model.getLineCount(),
        endColumn: model.getLineContent(model.getLineCount()).length + 1,
      });

      // Create FIM prompt with prefix and suffix
      const prompt = this.createFIMPrompt(beforeCursor, afterCursor);
      
      // Generate completion and type it with simulated delay
      await this.generateCodeCompletion(prompt);
    } catch (error) {
      console.error("Error generating code with Ollama:", error);
      // Fallback to pressing Enter if API fails
      await this.simulateTyping("\n");
    } finally {
      // Always release the typing lock
      this.isTyping = false;
      console.log("🔓 Released typing lock");
    }
  }

  private createFIMPrompt(prefix: string, suffix: string): string {
    // Include problem description in the prompt if available
    let promptPrefix = prefix;
    // TODO: We're limited to adding as a comment since we are using the base model, explore 
    // adding problem description in the prompt if we switch to APIs
    if (this.problemDescription) {
      const problemBlock = `\n'''\nProblem: ${this.problemDescription}\n'''\n\n`;
      // Add problem description at the beginning of the prefix
      promptPrefix = problemBlock + prefix;
    }
    
    // Use the same FIM template as the state-service
    return `<|fim_prefix|>${promptPrefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
  }

  private async generateCodeCompletion(prompt: string): Promise<void> {
    try {
      console.log('🤖 Starting completion...');
      
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.modelName,
          prompt: prompt,
          stream: false, // Use regular API, not streaming
          options: {
            temperature: 0.1,
            top_p: 1.0,
            num_predict: 50,
            stop: [
              "<|fim_prefix|>",
              "<|fim_suffix|>", 
              "<|fim_middle|>",
              "<|endoftext|>",
              "\n",
              "\r\n"
            ],
            repeat_penalty: 1.0,
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      let fullResponse = data.response || '';

      // Clean the response
      fullResponse = fullResponse.replace(/<\|fim_prefix\|>/g, '');
      fullResponse = fullResponse.replace(/<\|fim_suffix\|>/g, '');
      fullResponse = fullResponse.replace(/<\|fim_middle\|>/g, '');
      fullResponse = fullResponse.replace(/<\|endoftext\|>/g, '');

      if (fullResponse.trim()) {
        console.log(`🤖 Ollama generated: "${fullResponse.replace(/\n/g, '\\n')}"`);
        // Simulate typing the complete response with realistic delay
        await this.simulateTypingWithDelay(fullResponse);
      } else {
        // Fallback if nothing was generated
        await this.simulateTyping('\n');
        console.log(`🤖 Ollama fallback: newline`);
      }

    } catch (error) {
      console.error('Error calling Ollama:', error);
      // Fallback to regular typing
      await this.simulateTyping('\n');
    }
  }

  private async simulateTypingWithDelay(text: string): Promise<void> {
    if (!this.editor) return;
    
    // The isTyping mutex is already handled in generateAndTypeCode()
    // This method should only be called when it's safe to type
    
    for (const token of text.split('')) {
      this.editor.trigger('simulation', 'type', { text: token });
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log(`Simulated typing: "${text.replace(/\n/g, '\\n')}"`);
  }

  private async simulateTyping(text: string): Promise<void> {
    if (!this.editor) return;
    
    const position = this.editor.getPosition();
    if (!position) return;

    const edit = {
      range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
      text: text,
    };

    this.editor.executeEdits("simulation-ollama-type", [edit]);
    
    // Move cursor after the inserted text
    const lines = text.split('\n');
    let newLineNumber = position.lineNumber;
    let newColumn = position.column;
    
    if (lines.length > 1) {
      newLineNumber += lines.length - 1;
      newColumn = lines[lines.length - 1].length + 1;
    } else {
      newColumn += text.length;
    }
    
    const newPosition = new monaco.Position(newLineNumber, newColumn);
    this.editor.setPosition(newPosition);

    console.log(`Simulated typing: "${text.replace(/\n/g, '\\n')}"`);
  }

  private async simulateCursorMove(position: { line: number; column: number }): Promise<void> {
    if (!this.editor) return;
    
    const newPosition = new monaco.Position(position.line, position.column);
    this.editor.setPosition(newPosition);
    
    console.log(`Moved cursor to line ${position.line}, column ${position.column}`);
  }

  private async simulateWait(duration: number): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, duration));
    console.log(`Waited for ${duration}ms`);
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
